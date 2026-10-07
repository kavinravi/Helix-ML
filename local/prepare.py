"""Inspect CSV/asset datasets and persist an immutable evaluation protocol.

Everything written here is private evaluator input. Mount only a materialized
fold in training; never expose this directory or the original CSV to an agent.
"""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import random
import shutil
from statistics import fmean, median
import tempfile

METRICS = {"accuracy", "auroc", "log_loss", "rmse", "mae"}
MAX_CSV = 128 * 1024 * 1024
MAX_ASSETS = 2 * 1024 ** 3


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


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
    schema = []
    for column in columns:
        present = [row[column] for row in rows if row[column].strip()]
        numeric = bool(present)
        try:
            numeric = numeric and all(math.isfinite(float(v)) for v in present)
        except ValueError:
            numeric = False
        schema.append({"name": column, "type": "asset" if column in assets else "number" if numeric else "text",
                       "missing": len(rows) - len(present)})
    return schema


def inspect_dataset(task):
    source = Path(task["dataset"]).expanduser().resolve()
    if source.is_dir():
        source = next((source / name for name in ("train.csv", "labels.csv", "metadata.csv")
                       if (source / name).is_file()), source)
    if not source.is_file() or source.suffix.lower() != ".csv":
        raise ValueError("Choose a CSV, or a folder containing train.csv, labels.csv, or metadata.csv.")
    if source.stat().st_size > MAX_CSV:
        raise ValueError("CSV files are limited to 128 MiB in the CPU MVP.")
    # ponytail: CSV rows are held in memory up to 128 MiB; use streaming storage for larger datasets.
    raw = source.read_bytes()
    if len(raw) > MAX_CSV:
        raise ValueError("CSV files are limited to 128 MiB in the CPU MVP.")
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig"), newline=""))
    columns = reader.fieldnames
    target, metric = task["target"], task["metric"]
    if not columns or any(not c.strip() for c in columns) or len(set(columns)) != len(columns):
        raise ValueError("CSV column names must be nonempty and unique.")
    if len(columns) > 200:
        raise ValueError("The CPU MVP supports up to 200 CSV columns.")
    if target not in columns:
        raise ValueError(f"Target column {target!r} is absent. Columns: {', '.join(columns)}")
    if metric not in METRICS:
        raise ValueError("Choose a supported classification or regression metric.")
    rows = []
    for row in reader:
        rows.append(row)
        if len(rows) > 100_000:
            raise ValueError("The CPU MVP supports up to 100,000 labeled rows.")
    if len(rows) < 10:
        raise ValueError("At least 10 labeled rows are needed; classification and CV may need more.")
    if any(None in row or any(v is None for v in row.values()) for row in rows):
        raise ValueError("Every CSV row must have the same number of fields as the header.")
    if any(not row[target].strip() for row in rows):
        raise ValueError("Every row must have a target. Remove or label the missing targets first.")
    classification = metric in {"accuracy", "auroc", "log_loss"}
    classes = sorted({r[target] for r in rows}) if classification else []
    if classification and len(classes) < 2:
        raise ValueError("Classification needs at least two classes.")
    if classification and len(classes) > 1000:
        raise ValueError("The CPU MVP supports up to 1,000 classes.")
    if metric == "auroc" and len(classes) != 2:
        raise ValueError("AUROC supports binary classification; choose accuracy or log loss for multiple classes.")
    if not classification:
        try:
            if any(not math.isfinite(float(r[target])) for r in rows):
                raise ValueError()
        except ValueError:
            raise ValueError("Regression targets must be finite numbers.") from None
    strategy = task.get("splitStrategy", "independent")
    if strategy not in {"independent", "group", "time"}:
        raise ValueError("Choose independent rows, grouped rows, or chronological evaluation.")
    split_column = task.get("groupColumn" if strategy == "group" else "timeColumn", "") if strategy != "independent" else ""
    if strategy != "independent" and (split_column not in columns or split_column == target
                                      or any(not r[split_column].strip() for r in rows)):
        raise ValueError("Select a non-target group/time column with a value in every row.")
    features = [c for c in columns if c not in {target, split_column}]
    if not features:
        raise ValueError("Include at least one input column in addition to target and group/time columns.")
    assets = task.get("assetColumns", [])
    if (not isinstance(assets, list) or len(set(assets)) != len(assets)
            or any(c not in features for c in assets)):
        raise ValueError("Asset columns must be unique input column names.")
    files, total = {}, 0
    for column in assets:
        for name in sorted({row[column] for row in rows}):
            if name in files:
                continue
            path = asset_path(source.parent, name)
            if path == source:
                raise ValueError("The labeled CSV cannot also be an input asset.")
            size = path.stat().st_size
            total += size
            if total > MAX_ASSETS:
                raise ValueError("Referenced assets exceed the CPU MVP's 2 GiB dataset limit.")
            with path.open("rb") as handle:
                digest = hashlib.file_digest(handle, "sha256").hexdigest()
            files[name] = {"bytes": size, "sha256": digest}
    duplicate_rows = len(rows) - len({tuple(row[c] for c in columns) for row in rows})
    # Identical predictors, even under different labels, cannot cross a random split.
    keys = [tuple(files[row[c]]["sha256"] if c in assets else row[c] for c in features) for row in rows]
    duplicates = len(rows) - len(set(keys))
    if strategy == "independent" and duplicates:
        raise ValueError(f"Found {duplicates} repeated input rows. Deduplicate them or choose a group column to avoid leakage.")
    if strategy == "group":
        seen = {}
        for key, row in zip(keys, rows):
            if key in seen and seen[key] != row[split_column]:
                raise ValueError("Identical inputs occur in different groups. Put duplicates in the same group or deduplicate them.")
            seen[key] = row[split_column]
    schema = column_schema(rows, columns, assets)
    fingerprint = hashlib.sha256(raw + json.dumps(files, sort_keys=True).encode()).hexdigest()
    manifest = {"version": 2, "fingerprint": fingerprint, "rows": len(rows), "target": target,
                "metric": metric, "taskType": "classification" if classification else "regression",
                "features": features, "classes": classes, "schema": schema, "assetColumns": assets,
                "assets": files, "assetBytes": total, "duplicateRows": duplicate_rows,
                "splitStrategy": strategy, "splitColumn": split_column,
                "warnings": ["Confirm that rows are independent; hidden repeated entities cannot be detected automatically."]
                if strategy == "independent" else []}
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
    protocol["sha256"] = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
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
        write_csv(temporary / "dataset.csv", rows[0].keys(), rows)
        manifest["preparedCsvSha256"] = hashlib.sha256((temporary / "dataset.csv").read_bytes()).hexdigest()
        for name, expected in manifest["assets"].items():
            output = temporary / "assets" / name
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(asset_path(source.parent, name), output)
            with output.open("rb") as handle:
                if hashlib.file_digest(handle, "sha256").hexdigest() != expected["sha256"]:
                    raise ValueError("An asset changed during preparation. Retry with a stable dataset.")
        manifest.update({"developmentRows": len(protocol["development"]), "testRows": len(protocol["test"]),
                         "fitsPerTrial": len(protocol["evaluations"]), "protocolHash": protocol["sha256"],
                         "developmentSchema": column_schema([rows[i] for i in protocol["development"]], manifest["features"], manifest["assetColumns"]),
                         "assetTypes": sorted({Path(rows[i][c]).suffix.lower() for i in protocol["development"] for c in manifest["assetColumns"]}),
                         "classCounts": dict(Counter(rows[i][task["target"]] for i in protocol["development"])) if manifest["classes"] else {}})
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
        values = [float(value) for value in training_targets]
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
    if hashlib.sha256(json.dumps({k: v for k, v in protocol.items() if k != "sha256"}, sort_keys=True).encode()).hexdigest() != recorded_hash:
        raise ValueError("The recorded evaluation protocol was modified.")
    if hashlib.sha256((evaluation / "dataset.csv").read_bytes()).hexdigest() != manifest["preparedCsvSha256"]:
        raise ValueError("The prepared dataset was modified. Start a new run.")
    with (evaluation / "dataset.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    # Without a test split, final inputs only check export/reload and receive no score.
    fold = {"train": protocol["development"], "validation": (protocol["test"] if fold_id == "final" else []) or protocol["evaluations"][0]["validation"]} if fold_id in {"final", "refit"} else next(f for f in protocol["evaluations"] if f["id"] == fold_id)
    destination.mkdir(parents=True, exist_ok=False)
    write_csv(destination / "train.csv", manifest["features"] + [manifest["target"]], (rows[i] for i in fold["train"]))
    write_csv(destination / "validation.csv", manifest["features"], (rows[i] for i in fold["validation"]))
    # No whole-dataset statistics, paths, labels, or other folds enter this mount.
    public = {k: manifest[k] for k in ("target", "metric", "features", "classes", "assetColumns", "taskType")}
    dump(destination / "manifest.json", public)
    dump(destination / "classes.json", manifest["classes"])
    names = {rows[i][c] for i in fold["train"] + fold["validation"] for c in manifest["assetColumns"]}
    for name in names:
        output = destination / name
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(evaluation / "assets" / name, output)
        with output.open("rb") as handle:
            if hashlib.file_digest(handle, "sha256").hexdigest() != manifest["assets"][name]["sha256"]:
                raise ValueError("A prepared asset was modified. Start a new run.")
    return {"targets": [rows[i][manifest["target"]] for i in fold["validation"]],
            "baseline": reference_prediction(manifest["metric"],
                [rows[i][manifest["target"]] for i in fold["train"]], manifest["classes"])}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task")
    parser.add_argument("--output")
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--evaluation")
    parser.add_argument("--fold")
    parser.add_argument("--truth")
    args = parser.parse_args()
    try:
        if args.evaluation:
            result = materialize(args.evaluation, args.output, args.fold)
            if args.truth:
                dump(Path(args.truth), result)
                result = {"rows": len(result["targets"])}
        else:
            task = json.loads(Path(args.task).read_text())
            if args.inspect:
                _, rows, result = inspect_dataset(task)
                protocol = make_protocol(task, rows, result)
                result.update({"developmentRows": len(protocol["development"]), "testRows": len(protocol["test"]),
                               "fitsPerTrial": len(protocol["evaluations"])})
            else:
                result = prepare(task, args.output)
        print(json.dumps(result, allow_nan=False))
    except (ValueError, OSError, KeyError, StopIteration, csv.Error) as error:
        parser.exit(1, str(error) + "\n")
