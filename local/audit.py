"""Reject known policy violations without executing candidate code.

This is a conservative syntax check, not a proof of arbitrary Python behavior.
Runtime isolation and a recorded agent review are separate required checks.
"""
import ast
import json
from pathlib import Path
import re
import sys


def check_source(source, task):
    tree = ast.parse(source)
    aliases = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for name in node.names:
                aliases[name.asname or name.name.split(".")[0]] = name.name if name.asname else name.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom):
            for name in node.names:
                aliases[name.asname or name.name] = (node.module or "") + "." + name.name

    def symbol(node):
        if isinstance(node, ast.Name):
            return aliases.get(node.id, node.id)
        if isinstance(node, ast.Attribute):
            return symbol(node.value) + "." + node.attr
        return ""

    policy = task["policy"]
    requested = task.get("model", "").strip()
    exact = re.fullmatch(r"(?:sklearn\.[A-Za-z_]+\.)?([A-Z][A-Za-z0-9]+(?:Classifier|Regressor|Regression))", requested)
    reduction = task.get("learning") == "reduction"
    if task.get("learning") in {"clustering", "reduction"}:
        exact = re.fullmatch(r"(?:sklearn\.[A-Za-z_]+\.|umap\.)?([A-Z][A-Za-z0-9]+)", requested)
    tree_only = bool(re.fullmatch(r"tree(?:[- ]based)? models?(?: only)?", requested, re.I))
    estimators = ("sklearn.tree.", "sklearn.ensemble.", "sklearn.linear_model.", "sklearn.svm.", "sklearn.neighbors.", "sklearn.naive_bayes.", "sklearn.neural_network.", "xgboost.", "lightgbm.", "catboost.")
    if task.get("learning") in {"clustering", "reduction"}:
        estimators += ("sklearn.cluster.", "sklearn.mixture.", "sklearn.decomposition.", "umap.")
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = symbol(node.func)
        leaf = name.split(".")[-1]
        keywords = {arg.arg: arg.value for arg in node.keywords}
        def literal(key, expected):
            try:
                return ast.literal_eval(keywords[key]) == expected
            except (KeyError, ValueError, TypeError):
                return False
        reason = None
        if exact and name.startswith(estimators) and leaf != exact.group(1):
            reason = f"Only the requested estimator {exact.group(1)} is permitted"
        if leaf in {"eval", "exec", "__import__"} or name.startswith("importlib."):
            reason = "Dynamic code loading cannot be reviewed by the policy checker"
        if not policy["pretrained"] and leaf in {"from_pretrained", "load_state_dict_from_url", "hub_load"}:
            reason = "Pretrained weights are forbidden"
        if not policy["pretrained"] and (("pretrained" in keywords and not literal("pretrained", False)) or (name.startswith("torchvision.models.") and "weights" in keywords and not literal("weights", None))):
            reason = "Pretrained weights are forbidden"
        if not policy["ensemble"] and leaf.startswith(("Voting", "Stacking")):
            reason = "Combining models is forbidden"
        if not policy["features"] and (leaf in {"SelectKBest", "SelectPercentile", "SelectFromModel", "PolynomialFeatures", "RFE", "RFECV"} or (leaf in {"PCA", "KernelPCA"} and not reduction)):
            reason = "Derived features and feature selection are forbidden"
        if not policy["tuning"] and (leaf.endswith("SearchCV") or leaf in {"create_study", "fmin"}):
            reason = "Hyperparameter search is forbidden"
        if not policy["augmentation"] and (leaf in {"SMOTE", "ADASYN", "RandomOverSampler", "RandomUnderSampler", "RandomHorizontalFlip", "RandomVerticalFlip", "RandomRotation", "ColorJitter", "RandAugment", "MixUp", "CutMix"}):
            reason = "Synthetic samples and randomized augmentation are forbidden"
        if not policy["regularization"]:
            if leaf.startswith("Dropout") or leaf in {"dropout", "dropout2d", "dropout3d", "Ridge", "Lasso", "ElasticNet", "AdamW"}:
                reason = "Added regularization is forbidden"
            if leaf == "LogisticRegression" and not literal("penalty", None):
                reason = "LogisticRegression requires penalty=None when regularization is disabled"
            if "weight_decay" in keywords and not literal("weight_decay", 0):
                reason = "Weight decay is forbidden"
        if tree_only:
            if name.startswith(("sklearn.linear_model.", "sklearn.svm.", "sklearn.neighbors.", "torch.nn.", "tensorflow.keras.", "keras.layers.")):
                reason = "The requested family permits tree models only"
        if reason:
            raise ValueError(f"Line {node.lineno}: {reason}: {name}")


if __name__ == "__main__":
    folder, task_file = map(Path, sys.argv[1:])
    task = json.loads(task_file.read_text())
    try:
        for path in sorted(folder.rglob("*.py")):
            if path.is_symlink():
                raise ValueError("Source symlinks are forbidden")
            check_source(path.read_text(), task)
        print(json.dumps({"knownPolicyChecks": "passed", "provesArbitraryCodeCompliance": False}))
    except (SyntaxError, ValueError) as error:
        sys.exit(str(error))
