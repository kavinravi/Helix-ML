"""Catalog coverage and scoring boundaries, run inside the installed CPU runtime."""
import csv
import json
import math
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from supervised import CATALOG, evaluate
from unsupervised import evaluate as evaluate_unsupervised


def rejects(call, fragment):
    try:
        call()
    except ValueError as error:
        assert fragment in str(error), str(error)
    else:
        raise AssertionError(f"Expected rejection: {fragment}")


truth, classes = ["n", "p", "n", "p"], ["n", "p"]
probabilities = [[.9, .1], [.5, .5], [.5, .5], [.1, .9]]
task = {"metric": "auroc", "metrics": ["auroc", "f1", "precision", "recall", "accuracy"]}
result = evaluate(task, truth, probabilities, classes)
assert result["metricScores"] == {"auroc": .875, "f1": .8, "precision": 2/3, "recall": 1, "accuracy": .75}
assert math.isclose(result["score"], np.mean(list(result["metricScores"].values())))
assert evaluate({"metric": "f1"}, ["n", "n"], ["n", "n"], classes)["score"] == 0
assert evaluate({"metric": "f1", "positiveClass": "n"}, classes, ["n", "n"], classes)["score"] == 2/3
assert math.isclose(evaluate({"metric": "f1"}, ["a", "b", "c"], ["a", "a", "c"], ["a", "b", "c"])["score"], (2/3+1)/3)
assert evaluate({"metric": "auroc"}, truth, [.1, .5, .5, .9], classes)["score"] == .875
assert evaluate({"metric": "auroc", "positiveClass": "n"}, truth, [.9, .5, .5, .1], classes)["score"] == .875
assert evaluate({"metric": "mse", "metrics": ["mse", "rmse", "mae"]}, [1, 3], [1, 5], [])["metricScores"] == {"mse": 2, "rmse": math.sqrt(2), "mae": 1}
for bad in ([[.9, .9]]*4, [[".5", ".5"]]*4, [[True, False]]*4, [[float("nan"), .5]]*4, [.5]*4):
    rejects(lambda: evaluate(task, truth, bad, classes), "Probabilities" if isinstance(bad[0], list) and isinstance(bad[0][0], (str, bool)) else "probability")
rejects(lambda: evaluate(task, truth, [], classes), "Expected")
rejects(lambda: evaluate({"metric":"f1"}, truth, ["unknown"]*4, classes), "class labels")
rejects(lambda: evaluate({"metric":"f1"}, truth, probabilities, classes), "class labels")
rejects(lambda: evaluate({"metric":"mae"}, [1, 2], [True, 2], []), "finite numbers")
rejects(lambda: evaluate({"metric":"mae"}, [1, float("nan")], [1, 2], []), "finite numbers")
rejects(lambda: evaluate({"metric":"mean_gamma_deviance"}, [1, 2], [-1, 2], []), "mean_gamma_deviance")
rejects(lambda: evaluate({"metric":"auroc"}, ["n", "n"], [.1, .2], classes), "undefined")

checked = set()
for name, spec in CATALOG.items():
    if spec["kind"] == "regression":
        values = [1., 2., 4., 8.]
        result = evaluate({"metric": name}, values, values, [])
        assert math.isclose(result["score"], 1 if spec["maximize"] else 0, abs_tol=1e-12), (name, result)
    elif spec["kind"] == "classification":
        for labels in (["a", "b"], ["a", "b", "c"]):
            if len(labels) == 3 and name in {"auroc", "average_precision"} or len(labels) == 2 and name == "top_k_accuracy":
                continue
            actual = labels * 3
            output = actual
            if spec["response"] == "probabilities":
                output = [[float(value == label) for label in labels] for value in actual]
            result = evaluate({"metric": name}, actual, output, labels)
            assert math.isclose(result["score"], 1 if spec["maximize"] else 0, abs_tol=1e-12), (name, result)
    else:
        continue
    checked.add(name)

with TemporaryDirectory() as folder:
    data = Path(folder)
    request = {"task": {"metric": "mean_poisson_deviance"}, "truth": [0, 0], "predictions": [.1, .1], "classes": [], "baseline": {"prediction": 0}}
    (data / "input.json").write_text(json.dumps(request))
    subprocess.run([sys.executable, str(Path(__file__).with_name("supervised.py")), folder], check=True)
    result = json.loads((data / "score.json").read_text())
    assert math.isclose(result["score"], .2) and "baselineError" in result and "baseline" not in result
    rows = [[i % 3 * 8 + math.sin(i), i % 3 * 6 + math.cos(i), math.sin(i)] for i in range(90)]
    for filename in ("train.csv", "validation.csv"):
        with (data / filename).open("w") as handle:
            writer = csv.writer(handle); writer.writerow(["a", "b", "c"]); writer.writerows(rows)
    for name, spec in CATALOG.items():
        if spec["kind"] not in {"clustering", "reduction"}:
            continue
        metadata = {"features": ["a", "b", "c"], "taskType": spec["kind"], "metric": name, "dimensions": 2}
        (data / "manifest.json").write_text(json.dumps(metadata))
        predictions = [i % 3 for i in range(90)] if spec["kind"] == "clustering" else [row[:2] for row in rows]
        labels = [str(i % 3) for i in range(90)]
        result = evaluate_unsupervised(data, predictions, 42, truth=labels)
        assert math.isfinite(result["score"]), (name, result)
        if spec.get("targetRequired"):
            assert math.isclose(result["score"], math.log(3) if name == "mutual_info" else 1, abs_tol=1e-12), (name, result)
            rejects(lambda: evaluate_unsupervised(data, predictions, 42), "ground-truth")
        checked.add(name)

assert checked == set(CATALOG)
print(f"All {len(checked)} metrics passed: binary/multiclass, regression, clustering, reduction, joint scores, and invalid outputs.")
