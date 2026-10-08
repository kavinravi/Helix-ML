"""Inspect CSV/asset datasets and persist an immutable evaluation protocol.

Everything written here is private evaluator input. Mount only a materialized
fold in training; never expose this directory or the original CSV to an agent.
"""
import argparse
from array import array
import ast
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import random
import shutil
import sqlite3
import sys
from statistics import fmean, median
import tempfile

METRICS = {"accuracy", "auroc", "log_loss", "rmse", "mae", "silhouette", "davies_bouldin", "trustworthiness"}
MAX_CSV = 2_000_000_000  # Storage/upload limit, not a RAM or row limit.
MAX_ASSETS = 2 * 1024 ** 3


def dump(path, value):
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, allow_nan=False)


def write_csv(path, columns, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def asset_path(root, name):
    path = PurePosixPath(name)
    if (not name or path.is_absolute() or "\\" in name or "\0" in name
            or any(part in {"", ".", ".."} or part.startswith(".") for part in name.split("/"))):
        raise ValueError(f"Asset paths must be relative, without traversal: {name!r}")
    candidate = root
    for part in path.parts:
        candidate /= part
        if candidate.is_symlink():
            raise ValueError(f"Asset symlinks are not supported: {name!r}")
    if not candidate.is_file() or not candidate.resolve().is_relative_to(root):
        raise ValueError(f"Missing or outside-dataset asset: {name!r}")
    if path.parts[0] in {"train.csv", "validation.csv", "manifest.json", "classes.json"}:
        raise ValueError(f"Asset name is reserved by the evaluator: {name!r}")
    return candidate


def time_value(value):
    try:
        result = float(value)
        if math.isfinite(result):
            return result
    except ValueError:
        pass
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed.replace(tzinfo=parsed.tzinfo or timezone.utc).timestamp()
    except (ValueError, OverflowError):
        raise ValueError("Time values must be finite numbers or ISO 8601 timestamps.") from None


def column_schema(rows, columns, assets):
    count, present, numeric = 0, [0] * len(columns), [True] * len(columns)
    for row in rows:
        count += 1
        for i, column in enumerate(columns):
            value = row[column]
            if value.strip():
                present[i] += 1
                if numeric[i]:
                    try:
                        numeric[i] = math.isfinite(float(value))
                    except ValueError:
                        numeric[i] = False
    return [{"name": column, "type": "asset" if column in assets else "number" if present[i] and numeric[i] else "text",
             "missing": count - present[i]} for i, column in enumerate(columns)]


def memory_limit(kind):
    default = 512 if kind == "PREP" else 1200 if os.environ.get("HELIX_HOSTED_WORKER") else 3072
    try:
        value = int(os.environ.get(f"HELIX_{kind}_MEMORY_MB") or default)
        if not 128 <= value <= 65536:
            raise ValueError()
        return value * 1024 ** 2
    except ValueError:
        raise ValueError(f"HELIX_{kind}_MEMORY_MB must be an integer from 128 to 65536.") from None


class SplitRows:
    """Compact label/group codes needed for splitting; feature rows stay on disk."""
    def __init__(self, columns):
        self.count, self.memory_bytes = 0, 0
        self.codes = {c: array("I") for c in columns}
        self.values = {c: [] for c in columns}
        self.lookup = {c: {} for c in columns}

    def add(self, row):
        for column, codes in self.codes.items():
            value = row[column]
            lookup = self.lookup[column]
            if value not in lookup:
                lookup[value] = len(lookup)
                self.values[column].append(value)
                self.memory_bytes += sys.getsizeof(value) + 112
            codes.append(lookup[value])
            self.memory_bytes += codes.itemsize
        self.count += 1

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        if not 0 <= index < self.count:
            raise IndexError(index)
        return {c: self.values[c][codes[index]] for c, codes in self.codes.items()}


def check_preparation_memory(task, rows, asset_metadata_bytes=0):
    fits = (task.get("folds", 3) if task.get("validation") == "cv" else 1) * len(task.get("seeds", [42]))
    # ponytail: indices remain JSON for compatibility with existing runs. Budget their
    # Python/Node copies; use binary partition files when this becomes the bottleneck.
    estimate = 64 * 1024 ** 2 + rows.memory_bytes + asset_metadata_bytes + len(rows) * (96 + 48 * (fits + 1))
    limit = memory_limit("PREP")
    if estimate > limit:
        raise ValueError(f"Preparing {len(rows):,} rows with {fits} validation fits needs about {math.ceil(estimate / 1024 ** 2):,} MiB for split metadata; this worker allows {limit // 1024 ** 2:,} MiB. Use fewer folds/seeds or a worker with more preparation memory.")
    return estimate


def hash_json(value):
    digest = hashlib.sha256()
    for chunk in json.JSONEncoder(sort_keys=True).iterencode(value):
        digest.update(chunk.encode())
    return digest.hexdigest()


def file_hash(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def estimate_resources(manifest, task, workspace=None):
    """Conservative working-set estimates, not a guarantee for arbitrary generated code."""
    n, features = manifest["rows"], manifest["features"]
    text = [c["name"] for c in manifest["schema"] if c["name"] in features and c["type"] == "text"]
    string_bytes = sum(manifest.get("featureStringBytes", {}).get(c, 0) for c in text)
    matrix_bytes = n * len(features) * 8 + n * len(text) * 64 + string_bytes * 4
    if manifest.get("featureEncoding") == "mixed-v1":
        matrix_bytes += string_bytes * 16 + n * len(text) * 16
    outputs = len(manifest["classes"]) if task["metric"] == "log_loss" else (task.get("dimensions") or 2) if manifest["taskType"] == "reduction" else 1
    if manifest["taskType"] == "reduction" and task.get("reductionMode") == "variance":
        outputs = len(features)
    output_bytes = n * outputs * 16
    base = 192 * 1024 ** 2
    estimate = base + matrix_bytes * 4 + n * 32 + output_bytes * 2
    # ponytail: recognize common allocations via AST, not arbitrary Python data flow.
    # The container enforces unknown allocations; extend estimates when a real model needs it.
    calls, incremental, chunked = [], False, False
    if workspace:
        for path in Path(workspace).rglob("*.py"):
            tree = ast.parse(path.read_text())
            aliases = {alias.asname or alias.name: alias.name for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) for alias in node.names}
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                name = aliases.get(node.func.id, node.func.id) if isinstance(node.func, ast.Name) else node.func.attr if isinstance(node.func, ast.Attribute) else ""
                params = {}
                for arg in node.keywords:
                    try:
                        params[arg.arg] = ast.literal_eval(arg.value)
                    except (ValueError, TypeError):
                        pass
                calls.append((name, params))
                incremental |= name == "partial_fit"
                chunked |= name == "read_csv" and isinstance(params.get("chunksize"), int) and 0 < params["chunksize"] <= 10000
    else:
        calls = [(task.get("model", "").split(".")[-1], {})]
    notes = ["Estimated working memory; learned model size and generated code can use more."]
    if incremental and chunked:
        estimate = base + min(n, 10000) * math.ceil(matrix_bytes / n) * 6 + output_bytes * 2
        notes.append("Chunked CSV input and partial_fit detected; data-buffer estimate uses at most 10,000 rows.")
    pairwise = {"KernelPCA", "MDS", "Isomap", "SpectralClustering", "SpectralEmbedding", "GaussianProcessRegressor", "GaussianProcessClassifier", "AgglomerativeClustering"}
    model_bytes = 0
    for name, params in calls:
        if name in pairwise:
            model_bytes += n * n * 8 * 2
            notes.append(f"{name}: budgeted two dense pairwise matrices.")
        if name == "UMAP":
            neighbors = params.get("n_neighbors", 15)
            neighbors = neighbors if isinstance(neighbors, int) and neighbors > 0 else 15
            model_bytes += n * min(n, neighbors) * 96
            notes.append(f"UMAP: budgeted a {neighbors}-neighbor graph and optimization buffers.")
        if name == "KMeans" and params.get("algorithm") == "elkan":
            clusters = params.get("n_clusters", 8)
            model_bytes += n * (clusters if isinstance(clusters, int) and clusters > 0 else 8) * 8
            notes.append("Elkan KMeans: budgeted per-row cluster bounds; Lloyd uses less memory.")
    estimate += model_bytes
    scorer = base + matrix_bytes * 6 + output_bytes * 2 + n * 64 if manifest["taskType"] in {"clustering", "reduction"} else 0
    return {"limitBytes": memory_limit("TRAIN"), "estimatedFitBytes": estimate, "estimatedScoringBytes": scorer,
            "rows": n, "features": len(features), "notes": notes}


def check_fit_memory(evaluation, workspace):
    manifest = json.loads((evaluation / "manifest.json").read_text())
    task = json.loads((evaluation / "task.json").read_text())
    resources = estimate_resources(manifest, task, workspace)
    limit = resources["limitBytes"]
    for label, key in [("This candidate", "estimatedFitBytes"), ("Trusted evaluation", "estimatedScoringBytes")]:
        if resources[key] > limit:
            raise ValueError(f"{label} is estimated to need {math.ceil(resources[key] / 1024 ** 2):,} MiB for {resources['rows']:,} rows × {resources['features']} features; this worker provides {limit // 1024 ** 2:,} MiB. Use a smaller-memory model or chunked CSV input with partial_fit, exclude unused features, or use a worker with more RAM. " + " ".join(resources["notes"]))
    return resources


def inspect_dataset(task, columns_only=False):
    source = Path(task["dataset"]).expanduser().resolve()
    if source.is_dir():
        source = next((source / name for name in ("train.csv", "labels.csv", "metadata.csv") if (source / name).is_file()), source)
    if not source.is_file() or source.suffix.lower() != ".csv":
        raise ValueError("Choose a CSV, or a folder containing train.csv, labels.csv, or metadata.csv.")
    initial = source.stat()
    if initial.st_size > MAX_CSV:
        raise ValueError("Each CSV must be smaller than the 2 GB upload/storage limit.")
    with source.open(newline="", encoding="utf-8-sig") as handle:
        columns = csv.DictReader(handle).fieldnames
    if not columns or any(not c.strip() for c in columns) or len(set(columns)) != len(columns):
        raise ValueError("CSV column names must be nonempty and unique.")
    if len(columns) > 200:
        raise ValueError("The CPU MVP supports up to 200 CSV columns.")
    if columns_only:
        return columns
    learning = task.get("learning", "supervised")
    if learning not in {"supervised", "clustering", "reduction"}:
        raise ValueError("Choose a supported learning mode.")
    unsupervised, metric = learning != "supervised", task["metric"]
    target = "" if unsupervised else task["target"]
    if not unsupervised and target not in columns:
        raise ValueError(f"Target column {target!r} is absent. Columns: {', '.join(columns)}")
    allowed = {"silhouette", "davies_bouldin"} if learning == "clustering" else {"trustworthiness"} if learning == "reduction" else {"accuracy", "auroc", "log_loss", "rmse", "mae"}
    if metric not in allowed:
        raise ValueError("Choose a metric for the selected learning mode.")
    classification = metric in {"accuracy", "auroc", "log_loss"}
    strategy = task.get("splitStrategy", "independent")
    if strategy not in {"independent", "group", "time"}:
        raise ValueError("Choose independent rows, grouped rows, or chronological evaluation.")
    split_column = task.get("groupColumn" if strategy == "group" else "timeColumn", "") if strategy != "independent" else ""
    if strategy != "independent" and (split_column not in columns or split_column == target):
        raise ValueError("Select a non-target group/time column with a value in every row.")
    excluded = task.get("excludedColumns", [])
    if not isinstance(excluded, list) or any(c not in columns for c in excluded):
        raise ValueError("Excluded columns must exist in the CSV.")
    if target and target in excluded:
        raise ValueError("The target cannot also be excluded.")
    features = [c for c in columns if c not in {target, split_column} and c not in excluded]
    if not features:
        raise ValueError("Include at least one input column after excluding target, group/time, and excluded columns.")
    assets = task.get("assetColumns", [])
    if not isinstance(assets, list) or len(set(assets)) != len(assets) or any(c not in features for c in assets):
        raise ValueError("Asset columns must be unique input column names.")
    rows = SplitRows(([target] if classification else []) + ([split_column] if split_column else []))
    files, sizes = {}, {c: 0 for c in features}
    total, asset_memory, duplicate_rows = 0, 0, 0
    with tempfile.TemporaryDirectory(prefix="helix-inspect-") as temporary:
        database = sqlite3.connect(str(Path(temporary) / "duplicates.sqlite"))
        try:
            database.executescript("PRAGMA journal_mode=OFF; PRAGMA cache_size=-8192; PRAGMA temp_store=FILE; CREATE TABLE inputs (digest BLOB PRIMARY KEY, grp TEXT) WITHOUT ROWID; CREATE TABLE full_rows (digest BLOB PRIMARY KEY) WITHOUT ROWID;")

            def checked_rows():
                nonlocal total, asset_memory, duplicate_rows
                with source.open(newline="", encoding="utf-8-sig") as handle:
                    for row in csv.DictReader(handle):
                        if None in row or any(v is None for v in row.values()):
                            raise ValueError("Every CSV row must have the same number of fields as the header.")
                        if target and not row[target].strip():
                            raise ValueError("Every row must have a target. Remove or label the missing targets first.")
                        if target and not classification:
                            try:
                                if not math.isfinite(float(row[target])):
                                    raise ValueError()
                            except ValueError:
                                raise ValueError("Regression targets must be finite numbers.") from None
                        if split_column and not row[split_column].strip():
                            raise ValueError("Select a non-target group/time column with a value in every row.")
                        if strategy == "time":
                            time_value(row[split_column])
                        rows.add(row)
                        if classification and len(rows.values[target]) > 1000:
                            raise ValueError("The CPU MVP supports up to 1,000 classes.")
                        for column in assets:
                            name = row[column]
                            if name not in files:
                                path = asset_path(source.parent, name)
                                if path == source:
                                    raise ValueError("The labeled CSV cannot also be an input asset.")
                                size = path.stat().st_size
                                total += size
                                if total > MAX_ASSETS:
                                    raise ValueError("Referenced assets exceed the 2 GiB dataset storage limit.")
                                files[name] = {"bytes": size, "sha256": file_hash(path)}
                                asset_memory += sys.getsizeof(name) + 512
                        if len(rows) % 1024 == 0:
                            check_preparation_memory(task, rows, asset_memory)
                        for column in features:
                            sizes[column] += len(row[column].encode("utf-8"))
                        if strategy != "independent":
                            digest = hashlib.sha256(json.dumps([row[c] for c in columns], ensure_ascii=False).encode()).digest()
                            duplicate_rows += database.execute("INSERT OR IGNORE INTO full_rows VALUES (?)", (digest,)).rowcount == 0
                        digest = hashlib.sha256(json.dumps([files[row[c]]["sha256"] if c in assets else row[c] for c in features], ensure_ascii=False).encode()).digest()
                        group = row[split_column] if strategy == "group" else ""
                        if database.execute("INSERT OR IGNORE INTO inputs VALUES (?,?)", (digest, group)).rowcount == 0:
                            if strategy == "independent":
                                raise ValueError("Found repeated input rows. Deduplicate them or choose a group column to avoid leakage.")
                            if strategy == "group" and database.execute("SELECT grp FROM inputs WHERE digest=?", (digest,)).fetchone()[0] != group:
                                raise ValueError("Identical inputs occur in different groups. Put duplicates in the same group or deduplicate them.")
                        yield row
            schema = column_schema(checked_rows(), columns, assets)
        finally:
            database.close()
    if len(rows) < 10:
        raise ValueError("At least 10 rows are needed; classification and CV may need more.")
    preparation_bytes = check_preparation_memory(task, rows, asset_memory)
    classes = sorted(rows.values[target]) if classification else []
    if classification and len(classes) < 2:
        raise ValueError("Classification needs at least two classes.")
    if metric == "auroc" and len(classes) != 2:
        raise ValueError("AUROC supports binary classification; choose accuracy or log loss for multiple classes.")
    if unsupervised:
        if assets:
            raise ValueError("Clustering and dimensionality reduction accept numeric, categorical, and text CSV columns. Image/audio asset files still need numeric embeddings; exclude asset paths or supply embeddings.")
        numeric_only = all(item["type"] == "number" for item in schema if item["name"] in features)
        dimensions = task.get("dimensions", 2)
        if learning == "reduction" and task.get("reductionMode", "dimensions") == "dimensions" and (type(dimensions) is not int or dimensions < 1 or (numeric_only and dimensions >= len(features))):
            raise ValueError("Output dimensions must be at least 1 and fewer than the number of encoded input features.")
    with source.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256")
    source_hash = digest.hexdigest()
    for chunk in json.JSONEncoder(sort_keys=True).iterencode(files):
        digest.update(chunk.encode())
    current = source.stat()
    if (initial.st_ino, initial.st_size, initial.st_mtime_ns) != (current.st_ino, current.st_size, current.st_mtime_ns):
        raise ValueError("The dataset changed during inspection. Retry with a stable CSV.")
    manifest = {"version": 2, "fingerprint": digest.hexdigest(), "sourceSha256": source_hash, "rows": len(rows), "target": target,
                "metric": metric, "taskType": learning if unsupervised else "classification" if classification else "regression",
                "dimensions": task.get("dimensions", 2), "reductionMode": task.get("reductionMode", "dimensions"),
                "varianceTarget": task.get("varianceTarget", .95),
                "features": features, "classes": classes, "schema": schema, "assetColumns": assets,
                "assets": files, "assetBytes": total, "duplicateRows": duplicate_rows,
                "splitStrategy": strategy, "splitColumn": split_column, "featureStringBytes": sizes,
                "warnings": ["Confirm that rows are independent; hidden repeated entities cannot be detected automatically."] if strategy == "independent" else []}
    if unsupervised:
        manifest["featureEncoding"] = "mixed-v1"
    manifest["resources"] = estimate_resources(manifest, task)
    manifest["resources"]["preparationBytes"] = preparation_bytes
    return source, rows, manifest


def make_protocol(task, rows, manifest):
    seeds, method, folds = task["seeds"], task["validation"], task["folds"]
    test_fraction, holdout_fraction = task.get("testFraction", .2), task.get("holdoutFraction", .2)
    if type(test_fraction) not in (int, float) or not 0 <= test_fraction < 1:
        raise ValueError("Test split must be at least 0% and less than 100%.")
    if type(holdout_fraction) not in (int, float) or not 0 < holdout_fraction < 1:
        raise ValueError("Validation split must be greater than 0% and less than 100% of development data.")
    if (not seeds or len(seeds) > 5 or any(type(s) is not int or not 0 <= s <= 2 ** 32 - 1 for s in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError("Provide one to five unique unsigned 32-bit seeds.")
    if method not in {"holdout", "cv"} or folds not in {3, 5, 10}:
        raise ValueError("Choose holdout or CV with 3, 5, or 10 folds.")
    target, strategy, column = manifest["target"], manifest["splitStrategy"], manifest["splitColumn"]
    classes = manifest["classes"]

    def buckets(ids):
        if not classes:
            return [list(ids)]
        groups = {c: [] for c in classes}
        for i in ids:
            groups[rows[i][target]].append(i)
        return list(groups.values())

    def partition(ids, seed, fraction):
        rng = random.Random(seed)
        if strategy == "time":
            times = sorted({time_value(rows[i][column]) for i in ids})
            if len(times) < 2:
                raise ValueError("Chronological evaluation needs multiple distinct timestamps in each partition.")
            cutoff = times[-max(1, math.ceil(len(times) * fraction))]
            return ([i for i in ids if time_value(rows[i][column]) < cutoff],
                    [i for i in ids if time_value(rows[i][column]) >= cutoff])
        if strategy == "group":
            groups = sorted({rows[i][column] for i in ids})
            if len(groups) < 2:
                raise ValueError("Need more distinct groups for development and held-out partitions.")
            rng.shuffle(groups)
            held = set(groups[:max(1, math.ceil(len(groups) * fraction))])
            return ([i for i in ids if rows[i][column] not in held], [i for i in ids if rows[i][column] in held])
        held = []
        for group in buckets(ids):
            if len(group) < 2:
                raise ValueError("Each class needs enough rows for independent test, training, and validation partitions.")
            rng.shuffle(group)
            held.extend(group[:max(1, round(len(group) * fraction))])
        selected = set(held)
        return [i for i in ids if i not in selected], sorted(held)

    def check(train, validation):
        if manifest["taskType"] in {"clustering", "reduction"} and (len(train) < 3 or len(validation) < 3):
            raise ValueError("Unsupervised evaluation needs at least 3 training and 3 evaluation rows in every split. Add rows, use fewer folds, or adjust the split.")
        if manifest["taskType"] == "reduction" and manifest["reductionMode"] == "dimensions" and len(train) <= manifest["dimensions"]:
            raise ValueError("Each training split needs more rows than the requested output dimensions.")
        if not train or not validation or set(train) & set(validation):
            raise ValueError("The requested evaluation produces an empty or overlapping partition.")
        if classes and (set(rows[i][target] for i in train) != set(classes)
                        or set(rows[i][target] for i in validation) != set(classes)):
            raise ValueError("Each training and evaluation partition must contain every class. Add samples, use fewer folds, or adjust groups/time.")

    development, test = partition(list(range(len(rows))), seeds[0], test_fraction) if test_fraction else (list(range(len(rows))), [])
    if test:
        check(development, test)
    evaluations = []
    for seed in seeds:
        if method == "holdout":
            pairs = [partition(development, seed, holdout_fraction)]
        elif strategy == "time":
            times = sorted({time_value(rows[i][column]) for i in development})
            if len(times) < folds + 1:
                raise ValueError("Chronological CV needs at least folds + 1 distinct development timestamps.")
            chunks = [set(times[i * len(times) // (folds + 1):(i + 1) * len(times) // (folds + 1)]) for i in range(folds + 1)]
            pairs = []
            for f in range(1, folds + 1):
                earlier = set.union(*chunks[:f])
                pairs.append(([i for i in development if time_value(rows[i][column]) in earlier],
                              [i for i in development if time_value(rows[i][column]) in chunks[f]]))
        else:
            rng = random.Random(seed)
            held = [[] for _ in range(folds)]
            if strategy == "group":
                groups = sorted({rows[i][column] for i in development})
                if len(groups) < folds:
                    raise ValueError("Development data needs at least one distinct group per fold.")
                rng.shuffle(groups)
                members = {group: [] for group in groups}
                for i in development:
                    members[rows[i][column]].append(i)
                for group in sorted(groups, key=lambda g: -len(members[g])):
                    min(held, key=len).extend(members[group])
            else:
                offset = 0
                for group in buckets(development):
                    if len(group) < folds:
                        raise ValueError("Each development class needs at least one row per fold. Add samples or use fewer folds.")
                    rng.shuffle(group)
                    for j, row in enumerate(group):
                        held[(offset + j) % folds].append(row)
                    offset = (offset + len(group)) % folds
            pairs = []
            for h in held:
                excluded = set(h)
                pairs.append(([i for i in development if i not in excluded], sorted(h)))
        for fold, (train, validation) in enumerate(pairs):
            check(train, validation)
            evaluations.append({"id": f"seed-{seed}-fold-{fold + 1}", "seed": seed, "fold": fold + 1,
                                "train": sorted(train), "validation": sorted(validation)})
    protocol = {"version": 2, "fingerprint": manifest["fingerprint"], "validation": method,
                "folds": folds if method == "cv" else 1, "seeds": seeds, "splitStrategy": strategy,
                "splitColumn": column, "development": development, "test": test, "evaluations": evaluations,
                "seedRule": "First seed fixes the test partition; each seed fixes development folds and model RNG. Time partitions stay chronological.",
                "aggregation": "Unweighted mean of all fold scores; population standard deviation. Incomplete candidates are rejected.",
                "testFraction": test_fraction, "holdoutFractionOfDevelopment": holdout_fraction}
    protocol["sha256"] = hash_json(protocol)
    return protocol


def prepare(task, destination):
    source, rows, manifest = inspect_dataset(task)
    protocol = make_protocol(task, rows, manifest)
    destination = Path(destination)
    if destination.exists():
        previous = json.loads((destination / "protocol.json").read_text())
        if previous != protocol or json.loads((destination / "task.json").read_text()) != task:
            raise ValueError("This run's dataset or configuration changed. Start a new run; saved partitions cannot be replaced.")
        return json.loads((destination / "manifest.json").read_text())
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".preparation-", dir=destination.parent))
    try:
        os.chmod(temporary, 0o700)
        shutil.copyfile(source, temporary / "dataset.csv")
        manifest["preparedCsvSha256"] = file_hash(temporary / "dataset.csv")
        if manifest["preparedCsvSha256"] != manifest["sourceSha256"]:
            raise ValueError("The dataset changed during preparation. Retry with a stable CSV.")
        for name, expected in manifest["assets"].items():
            output = temporary / "assets" / name
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(asset_path(source.parent, name), output)
            with output.open("rb") as handle:
                if hashlib.file_digest(handle, "sha256").hexdigest() != expected["sha256"]:
                    raise ValueError("An asset changed during preparation. Retry with a stable dataset.")
        labels, asset_types = Counter(), set()
        def training_rows():
            selected = iter(sorted(protocol["development"]))
            current = next(selected, None)
            with (temporary / "dataset.csv").open(newline="", encoding="utf-8-sig") as handle:
                for index, row in enumerate(csv.DictReader(handle)):
                    if index == current:
                        if manifest["classes"]:
                            labels[row[task["target"]]] += 1
                        asset_types.update(Path(row[c]).suffix.lower() for c in manifest["assetColumns"])
                        yield row
                        current = next(selected, None)
                    if current is None:
                        break
        schema = column_schema(training_rows(), manifest["features"], manifest["assetColumns"])
        manifest.update({"developmentRows": len(protocol["development"]), "testRows": len(protocol["test"]),
                         "fitsPerTrial": len(protocol["evaluations"]), "protocolHash": protocol["sha256"],
                         "developmentSchema": schema, "assetTypes": sorted(asset_types), "classCounts": dict(labels)})
        dump(temporary / "manifest.json", manifest)
        dump(temporary / "protocol.json", protocol)
        dump(temporary / "task.json", task)
        temporary.rename(destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return manifest


def reference_prediction(metric, training_targets, classes):
    """Fit the trivial comparator on this fold's training labels only."""
    if metric in {"rmse", "mae"}:
        values = (float(value) for value in training_targets)
        return {"name": "Training mean" if metric == "rmse" else "Training median",
                "prediction": fmean(values) if metric == "rmse" else median(values)}
    counts = Counter(training_targets)
    if metric == "accuracy":
        return {"name": "Most frequent class", "prediction": max(classes, key=lambda c: counts[c])}
    probabilities = [counts[c] / len(training_targets) for c in classes]
    return {"name": "Training class frequencies",
            "prediction": probabilities[1] if metric == "auroc" else probabilities}


def materialize(evaluation, destination, fold_id):
    evaluation, destination = Path(evaluation), Path(destination)
    manifest = json.loads((evaluation / "manifest.json").read_text())
    protocol = json.loads((evaluation / "protocol.json").read_text())
    recorded_hash = protocol["sha256"]
    if hash_json({k: v for k, v in protocol.items() if k != "sha256"}) != recorded_hash:
        raise ValueError("The recorded evaluation protocol was modified.")
    if file_hash(evaluation / "dataset.csv") != manifest["preparedCsvSha256"]:
        raise ValueError("The prepared dataset was modified. Start a new run.")
    fold = {"train": protocol["development"], "validation": (protocol["test"] if fold_id == "final" else []) or protocol["evaluations"][0]["validation"]} if fold_id in {"final", "refit"} else next(f for f in protocol["evaluations"] if f["id"] == fold_id)
    unsupervised = manifest["taskType"] in {"clustering", "reduction"}
    if unsupervised and fold_id in {"final", "refit"} and (fold_id == "refit" or not protocol["test"]):
        fold["validation"] = protocol["development"]
    destination.mkdir(parents=True, exist_ok=False)
    # Both lists are sorted CSV indices. Merge the selections while reading one row at a time.
    train_ids, valid_ids = iter(fold["train"]), iter(fold["validation"])
    train_index, valid_index = next(train_ids, None), next(valid_ids, None)
    classes = {value: value for value in manifest["classes"]}
    targets, names = [], set()
    training_targets = [] if classes else array("d")
    with (evaluation / "dataset.csv").open(newline="", encoding="utf-8-sig") as handle, (destination / "train.csv").open("w", newline="", encoding="utf-8") as train_file, (destination / "validation.csv").open("w", newline="", encoding="utf-8") as valid_file:
        train = csv.DictWriter(train_file, fieldnames=manifest["features"] + ([] if unsupervised else [manifest["target"]]), extrasaction="ignore")
        valid = csv.DictWriter(valid_file, fieldnames=manifest["features"], extrasaction="ignore")
        train.writeheader(); valid.writeheader()
        for index, row in enumerate(csv.DictReader(handle)):
            if index == train_index or index == valid_index:
                names.update(row[c] for c in manifest["assetColumns"])
            if index == train_index:
                train.writerow(row)
                if not unsupervised:
                    value = row[manifest["target"]]
                    training_targets.append(classes[value] if classes else float(value))
                train_index = next(train_ids, None)
            if index == valid_index:
                valid.writerow(row)
                if not unsupervised:
                    value = row[manifest["target"]]
                    targets.append(classes[value] if classes else value)
                valid_index = next(valid_ids, None)
            if train_index is None and valid_index is None:
                break
        if train_index is not None or valid_index is not None:
            raise ValueError("The prepared dataset is missing recorded rows.")

    # No whole-dataset statistics, paths, labels, or other folds enter this mount.
    public = {k: manifest[k] for k in ("target", "metric", "features", "classes", "assetColumns", "taskType")}
    if unsupervised:
        public.update({k: manifest[k] for k in ("dimensions", "reductionMode", "varianceTarget")})
        if "featureEncoding" in manifest:
            public["featureEncoding"] = manifest["featureEncoding"]
    dump(destination / "manifest.json", public)
    dump(destination / "classes.json", manifest["classes"])
    for name in names:
        output = destination / name
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(evaluation / "assets" / name, output)
        with output.open("rb") as handle:
            if hashlib.file_digest(handle, "sha256").hexdigest() != manifest["assets"][name]["sha256"]:
                raise ValueError("A prepared asset was modified. Start a new run.")
    if unsupervised:
        return {"targets": [None] * len(fold["validation"]), "baseline": None}
    return {"targets": targets, "baseline": reference_prediction(manifest["metric"], training_targets, manifest["classes"])}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task")
    parser.add_argument("--output")
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--columns", action="store_true")
    parser.add_argument("--evaluation")
    parser.add_argument("--fold")
    parser.add_argument("--truth")
    parser.add_argument("--resources", help="Estimate candidate memory without executing its source")
    args = parser.parse_args()
    try:
        if args.resources:
            result = check_fit_memory(Path(args.evaluation), args.resources)
        elif args.evaluation:
            result = materialize(args.evaluation, args.output, args.fold)
            if args.truth:
                dump(Path(args.truth), result)
                result = {"rows": len(result["targets"])}
        else:
            task = json.loads(Path(args.task).read_text())
            if args.columns:
                result = inspect_dataset(task, columns_only=True)
            elif args.inspect:
                _, rows, result = inspect_dataset(task)
                protocol = make_protocol(task, rows, result)
                result.update({"developmentRows": len(protocol["development"]), "testRows": len(protocol["test"]),
                               "fitsPerTrial": len(protocol["evaluations"])})
            else:
                result = prepare(task, args.output)
        print(json.dumps(result, allow_nan=False))
    except (ValueError, OSError, KeyError, StopIteration, csv.Error, sqlite3.Error, SyntaxError) as error:
        parser.exit(1, str(error) + "\n")
