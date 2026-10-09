"""One stdlib regression check for partition integrity and asset containment."""
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from prepare import inspect_dataset, make_protocol, materialize, prepare, reference_prediction, write_csv
from audit import check_source


def rejects(call, fragment):
    try:
        call()
    except ValueError as error:
        assert fragment in str(error), str(error)
    else:
        raise AssertionError(f"Expected rejection: {fragment}")

policy_task = {"model": "tree-based models only", "policy": {key: False for key in ("pretrained", "ensemble", "features", "tuning", "augmentation", "regularization")}}
check_source("from sklearn.tree import DecisionTreeClassifier\nmodel = DecisionTreeClassifier(random_state=42)", policy_task)
rejects(lambda: check_source("from sklearn.linear_model import LogisticRegression as LR\nLR()", policy_task), "tree models only")
rejects(lambda: check_source("from transformers import AutoModel\nAutoModel.from_pretrained('model')", policy_task), "Pretrained")
rejects(lambda: check_source("from sklearn.model_selection import GridSearchCV\nGridSearchCV(model, {})", policy_task), "Hyperparameter")


with TemporaryDirectory(prefix="helix-protocol-") as folder:
    root = Path(folder)
    source = root / "source"
    source.mkdir()
    rows = [{"x": str(i), "subject": str(i // 4), "time": str(i // 4), "label": str(i % 2)} for i in range(120)]
    write_csv(source / "train.csv", rows[0].keys(), rows)
    task = {"dataset": str(source), "target": "label", "metric": "accuracy", "validation": "cv", "folds": 5,
            "seeds": [42, 17], "splitStrategy": "independent", "assetColumns": []}
    assert inspect_dataset({**task, "target": ""}, columns_only=True) == list(rows[0])
    unsupervised = {**task, "learning": "reduction", "target": "", "metric": "trustworthiness", "dimensions": 2, "reductionMode": "dimensions", "varianceTarget": .95, "excludedColumns": ["label"], "testFraction": 0}
    unsigned = root / "unsupervised"
    info = prepare(unsupervised, unsigned)
    assert info["taskType"] == "reduction" and info["target"] == "" and info["classes"] == []
    truth = materialize(unsigned, root / "unlabeled", "final")
    assert truth["baseline"] is None and truth["targets"] == [None] * len(rows)
    with (root / "unlabeled" / "train.csv").open() as handle:
        assert csv.DictReader(handle).fieldnames == ["x", "subject", "time"]
    benchmark = {**task, "learning": "clustering", "metric": "adjusted_rand"}
    reference = root / "external-clustering"
    info = prepare(benchmark, reference)
    assert info["target"] == "label" and "label" not in info["features"]
    truth = materialize(reference, root / "external-fold", "seed-42-fold-1")
    assert set(truth["targets"]) == {"0", "1"} and truth["baseline"] is None
    for name in ("train.csv", "validation.csv"):
        with (root / "external-fold" / name).open() as handle:
            assert "label" not in csv.DictReader(handle).fieldnames
    rejects(lambda: inspect_dataset({**unsupervised, "dimensions": 3}), "fewer")
    rejects(lambda: inspect_dataset({**unsupervised, "excludedColumns": ["missing"]}), "Excluded")
    reduction_policy = {**policy_task, "learning": "reduction", "model": "sklearn.decomposition.PCA"}
    check_source("from sklearn.decomposition import PCA\nmodel=PCA(n_components=2, svd_solver='full')", reduction_policy)
    rejects(lambda: check_source("from sklearn.cluster import KMeans\nmodel=KMeans()", reduction_policy), "PCA")

    # Recorded before streaming preparation: existing runs must keep identical partitions and hashes.
    previous_hashes = {
        "independent-cv": "81726eb0b84b0fdf002d6ef2a99143ffdc4a2b19b756f4fbfebcc22df4a2c103",
        "independent-holdout": "bfa8c3410aa5b6240d47fb9169d980217ef775bc737a8233b64cb3f540f580a9",
        "group-cv": "82aca09ecc19ec36b1b38a72d96842338ad12203fb325bb31f95c93b49fa477e",
        "group-holdout": "0f55aa0017cb7088d281ab026508bc85c3ac98efa6183938716b4ba28ec9efda",
        "time-cv": "a1667d8e3efcee6dcea2113ede1ae7c04bd6303b85dc1b6635a53c3421cf3f13",
        "time-holdout": "df379c0ea4ff9105dad0211cb9ad37b3aa64e1ca466e92fe5d52afa47507d9e0",
    }
    for strategy in ("independent", "group", "time"):
        for method in ("cv", "holdout"):
            config = {**task, "validation": method, "splitStrategy": strategy, "groupColumn": "subject", "timeColumn": "time"}
            destination = root / f"{strategy}-{method}"
            manifest = prepare(config, destination)
            protocol = json.loads((destination / "protocol.json").read_text())
            assert protocol["sha256"] == previous_hashes[destination.name]
            assert prepare(config, destination) == manifest
            refit_data = root / (destination.name + "-refit")
            materialize(destination, refit_data, "refit")
            refit_train = list(csv.DictReader((refit_data / "train.csv").open()))
            refit_validation = list(csv.DictReader((refit_data / "validation.csv").open()))
            assert {int(row["x"]) for row in refit_train} == set(protocol["development"])
            assert {int(row["x"]) for row in refit_validation} == set(protocol["evaluations"][0]["validation"])
            assert not {int(row["x"]) for row in refit_train + refit_validation} & set(protocol["test"])

            assert not set(protocol["test"]) & set(protocol["development"])
            assert set(protocol["test"]) | set(protocol["development"]) == set(range(len(rows)))
            assert len(protocol["evaluations"]) == len(task["seeds"]) * (5 if method == "cv" else 1)
            for fold in protocol["evaluations"]:
                train, val, test = (set(fold["train"]), set(fold["validation"]), set(protocol["test"]))
                assert train and val and not train & val and not (train | val) & test
                if strategy == "group":
                    groups = [{rows[i]["subject"] for i in ids} for ids in (train, val, test)]
                    assert not groups[0] & groups[1] and not (groups[0] | groups[1]) & groups[2]
                if strategy == "time":
                    times = [[int(rows[i]["time"]) for i in ids] for ids in (train, val, test)]
                    assert max(times[0]) < min(times[1]) and max(times[1]) < min(times[2])
                data = root / (destination.name + "-" + fold["id"])
                measured = materialize(destination, data, fold["id"])
                truth = measured["targets"]
                with (data / "validation.csv").open() as handle:
                    validation = list(csv.DictReader(handle))
                assert "label" not in validation[0]
                assert len(truth) == len(validation)
                assert truth == [rows[i]["label"] for i in fold["validation"]]
                training_labels = [rows[i]["label"] for i in fold["train"]]
                assert measured["baseline"] == reference_prediction("accuracy", training_labels, manifest["classes"])
                assert set(p.name for p in data.iterdir()) == {"train.csv", "validation.csv", "manifest.json", "classes.json"}
            for seed in task["seeds"]:
                if method == "cv" and strategy != "time":
                    validation_ids = [i for f in protocol["evaluations"] if f["seed"] == seed for i in f["validation"]]
                    assert len(validation_ids) == len(set(validation_ids)) == len(protocol["development"])
            if strategy == "independent" and method == "cv":
                assert protocol["evaluations"][0]["validation"] != protocol["evaluations"][5]["validation"]
            rejects(lambda: prepare({**config, "seeds": [1]}, destination), "changed")
    for strategy in ("independent", "group", "time"):
        config = {**task, "validation": "holdout", "testFraction": .3, "holdoutFraction": .5,
                  "splitStrategy": strategy, "groupColumn": "subject", "timeColumn": "time"}
        _, records, manifest = inspect_dataset(config)
        protocol = make_protocol(config, records, manifest)
        assert len(protocol["test"]) == 36
        assert len(protocol["evaluations"][0]["validation"]) in (42, 44)
        no_test = make_protocol({**config, "testFraction": 0}, records, manifest)
        assert no_test["test"] == [] and len(no_test["development"]) == 120
        assert no_test["testFraction"] == 0
        rejects(lambda: make_protocol({**config, "holdoutFraction": 0}, records, manifest), "Validation split")
    _, records, manifest = inspect_dataset({**task, "metric": "rmse"})
    assert make_protocol({**task, "metric": "rmse"}, records, manifest)["test"]
    for metric in ("rmse", "mae"):
        destination = root / metric
        prepare({**task, "metric": metric}, destination)
        fold = json.loads((destination / "protocol.json").read_text())["evaluations"][0]
        measured = materialize(destination, root / (metric + "-fold"), fold["id"])
        assert measured["targets"] == [rows[i]["label"] for i in fold["validation"]]
        assert measured["baseline"] == reference_prediction(metric, [rows[i]["label"] for i in fold["train"]], [])
    joint={**task,"metric":"auroc","metrics":["auroc","f1"],"positiveClass":"0"}
    info=prepare(joint,root/"joint")
    assert info["taskType"]=="classification" and info["positiveClass"]=="0" and info["metrics"]==["auroc","f1"]
    rejects(lambda: inspect_dataset({**joint,"positiveClass":"absent"}), "positive class")
    joint_truth=materialize(root/"joint",root/"joint-fold","seed-42-fold-1")
    assert len(joint_truth["baseline"]["prediction"])==2
    assert inspect_dataset({**task,"metric":"f1"})[2]["taskType"]=="classification"
    rejects(lambda: inspect_dataset({**task, "target": "absent"}), "absent")
    write_csv(source / "train.csv", rows[0].keys(), rows + [rows[0]])
    rejects(lambda: inspect_dataset(task), "repeated")
    grouped = {**task, "splitStrategy": "group", "groupColumn": "subject"}
    assert inspect_dataset(grouped)[2]["duplicateRows"] == 1
    write_csv(source / "train.csv", rows[0].keys(), rows + [{**rows[0], "subject": "different"}])
    rejects(lambda: inspect_dataset(grouped), "different groups")
    (source / "train.csv").write_text("x,label\n1\n")
    rejects(lambda: inspect_dataset(task), "same number of fields")
    rows[0]["label"] = ""
    write_csv(source / "train.csv", rows[0].keys(), rows)
    rejects(lambda: inspect_dataset(task), "target")
    rows[0]["label"] = "0"
    (source / "assets").mkdir()
    for i, row in enumerate(rows):
        row["asset"] = f"assets/{i}.txt"
        (source / row["asset"]).write_text(f"asset {i}")
    (source / "neighbor-secret.txt").write_text("DO NOT COPY")
    write_csv(source / "train.csv", rows[0].keys(), rows)
    config = {**task, "assetColumns": ["asset"]}
    destination = root / "media"
    manifest = prepare(config, destination)
    assert len(manifest["assets"]) == 120
    assert not (destination / "assets" / "neighbor-secret.txt").exists()
    fold = json.loads((destination / "protocol.json").read_text())["evaluations"][0]
    materialize(destination, root / "media-fold", fold["id"])
    assert {p.name for p in (root / "media-fold" / "assets").iterdir()} == {f"{i}.txt" for i in fold["train"] + fold["validation"]}
    (source / "assets" / "0.txt").unlink()
    (source / "assets" / "0.txt").symlink_to(source / "neighbor-secret.txt")
    rejects(lambda: inspect_dataset(config), "symlink")
    rows[0]["asset"] = "../escape.txt"
    write_csv(source / "train.csv", rows[0].keys(), rows)
    rejects(lambda: inspect_dataset(config), "traversal")
assert reference_prediction("rmse", ["0", "0", "12"], []) == {"name": "Training mean", "prediction": 4}
assert reference_prediction("mae", ["0", "0", "12"], []) == {"name": "Training median", "prediction": 0}
assert reference_prediction("accuracy", ["b", "b", "a"], ["a", "b"])["prediction"] == "b"
assert reference_prediction("auroc", ["b", "b", "a"], ["a", "b"])["prediction"] == 2 / 3
assert reference_prediction("log_loss", ["b", "b", "a"], ["a", "b"])["prediction"] == [1 / 3, 2 / 3]
print("Protocol checks passed: holdout/CV, seeds, groups, time, targets, training-only baselines, immutable splits, and assets.")
