"""Independent metrics from canonical prediction JSON, never candidate code."""
import json
from pathlib import Path
import sys
import warnings

import numpy as np
from sklearn import metrics as sklearn_metrics

CATALOG = json.loads(Path(__file__).with_name("metric_catalog.json").read_text())


def evaluate(task, truth, predictions, classes):
    names = task.get("metrics", [task["metric"]])
    kind = CATALOG[task["metric"]]["kind"]
    n = len(truth)
    if not n or not isinstance(predictions, list) or len(predictions) != n:
        raise ValueError(f"Expected {n} predictions, one for each evaluation row.")
    if kind == "classification":
        if len(classes) < 2 or len(set(classes)) != len(classes) or any(value not in classes for value in truth):
            raise ValueError("Unknown or duplicate evaluation class.")
        positive = task.get("positiveClass", classes[1])
        if len(classes) == 2 and positive not in classes:
            raise ValueError("Unknown positive class.")
        probability_output = len(names) > 1 or CATALOG[names[0]]["response"] == "probabilities"
        probabilities = None
        if probability_output:
            if any(type(value) not in (int, float) for row in predictions for value in (row if isinstance(row, list) else [row])):
                raise ValueError("Probabilities must be numbers.")
            probabilities = np.asarray(predictions, dtype=float)
            if probabilities.ndim == 1 and len(names) == 1 and names[0] == "auroc" and len(classes) == 2:
                p = probabilities
                probabilities = np.column_stack((1-p, p) if positive == classes[1] else (p, 1-p))
            if probabilities.shape != (n, len(classes)) or not np.isfinite(probabilities).all() or (probabilities < 0).any() or (probabilities > 1).any() or not np.allclose(probabilities.sum(axis=1), 1, rtol=0, atol=.001):
                raise ValueError("Expected class-probability arrays in metadata.classes order, summing to 1.")
            labels = [classes[index] for index in probabilities.argmax(axis=1)]
            if len(classes) == 2:
                labels = [positive if row[classes.index(positive)] >= .5 else classes[1-classes.index(positive)] for row in probabilities]
        else:
            if any(not isinstance(value, (str, int, float, bool)) or str(value) not in classes for value in predictions):
                raise ValueError("This metric requires class labels, not probabilities.")
            labels = list(map(str, predictions))
    else:
        if any(type(value) not in (int, float) for value in predictions):
            raise ValueError("Regression predictions must be finite numbers.")
        truth, labels = np.asarray(truth, dtype=float), np.asarray(predictions, dtype=float)
        if truth.ndim != 1 or labels.ndim != 1 or not np.isfinite(truth).all() or not np.isfinite(labels).all():
            raise ValueError("Regression targets and predictions must be finite numbers.")
    scores = {}
    for name in names:
        spec = CATALOG[name]
        if spec["kind"] != kind:
            raise ValueError("All metrics must describe the same learning task.")
        kwargs, actual, output = dict(spec["kwargs"]), truth, labels
        if kind == "classification":
            if kwargs.get("average") == "binary":
                kwargs.update({"pos_label": positive} if len(classes) == 2 else {"average": "macro", "labels": classes})
            elif "average" in kwargs and spec["response"] == "labels":
                kwargs["labels"] = classes
            if spec["response"] == "probabilities":
                output = probabilities
                if name in {"auroc", "average_precision"} or (name.startswith("roc_auc_") and len(classes) == 2):
                    if len(classes) != 2:
                        raise ValueError(f"{name} requires two classes; use a multiclass ROC-AUC variant for multiclass targets.")
                    actual = [int(label == positive) for label in truth]
                    output = probabilities[:, classes.index(positive)]
                else:
                    kwargs["labels"] = classes
                    if name == "top_k_accuracy" and len(classes) < 3:
                        raise ValueError("Top-2 accuracy requires at least three classes; use accuracy for a binary task.")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                value = float(getattr(sklearn_metrics, spec["function"])(actual, output, **kwargs))
            if not np.isfinite(value):
                raise ValueError("undefined for these targets or predictions")
        except (ValueError, TypeError, AttributeError) as error:
            raise ValueError(f"{name}: {error}") from None
        scores[name] = value
    return {"score": sum(scores.values()) / len(scores), "metricScores": scores}


if __name__ == "__main__":
    work = Path(sys.argv[1])
    try:
        request = json.loads((work / "input.json").read_text())
        result = evaluate(request["task"], request["truth"], request["predictions"], request["classes"])
        baseline = request.get("baseline")
        if baseline:
            try:
                result["baseline"] = evaluate(request["task"], request["truth"], [baseline["prediction"]] * len(request["truth"]), request["classes"])
            except ValueError as error:
                # A valid model can beat a constant baseline outside the metric's domain.
                result["baselineError"] = str(error)
        (work / "score.json").write_text(json.dumps(result, allow_nan=False))
    except (ValueError, TypeError, KeyError) as error:
        sys.exit(str(error))
