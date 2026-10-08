"""Trusted scoring of held-out cluster assignments and embeddings.

Runs without candidate code or candidate-installed packages on its import path.
Reference preprocessing is fitted on training rows and fixed across candidates.
"""
import csv
import json
from pathlib import Path
import sys

import numpy as np
from scipy import sparse
from helix_features import dense_features, preprocessor, read_features
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.manifold import trustworthiness
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def evaluate(data, predictions, seed, unscored=False):
    metadata = json.loads((data / "manifest.json").read_text())
    features, kind = metadata["features"], metadata["taskType"]

    def matrix(name):
        with (data / name).open() as handle:
            values = (float(row[key]) if row[key].strip() else np.nan for row in csv.DictReader(handle) for key in features)
            return np.fromiter(values, dtype=float).reshape(-1, len(features))

    if metadata.get("featureEncoding") == "mixed-v1":
        rows = read_features(data / "train.csv", features)
        pipeline = preprocessor(rows)
        train = pipeline.fit_transform(rows)
        del rows
        reference = pipeline.transform(read_features(data / "validation.csv", features))
        axes = [str(name).split("__", 1)[-1] for name in pipeline.get_feature_names_out()[:2]]
    else:
        pipeline = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler())
        train = pipeline.fit_transform(matrix("train.csv"))
        reference = pipeline.transform(matrix("validation.csv"))
        axes = [name + " (scaled)" for name in features[:2]]
    if any(not np.isfinite(value.data if sparse.issparse(value) else value).all() for value in (train, reference)):
        raise ValueError("Reference features overflowed during scaling; rescale the input values.")
    n = reference.shape[0]
    if not isinstance(predictions, list) or len(predictions) != n:
        raise ValueError(f"Expected {n} outputs, one per evaluation row.")
    # ponytail: quadratic metrics use the same seeded 1,000-row sample for every candidate.
    # Increase this bound or use a blockwise estimator when large-sample precision is needed.
    sample = np.sort(np.random.default_rng(seed).choice(n, min(n, 1000), replace=False))
    display = sample[np.linspace(0, len(sample) - 1, min(len(sample), 300), dtype=int)]
    projection = {"kind": kind, "rows": n, "points": []}
    extra = {}
    if kind == "clustering":
        if any(type(label) is not int or not 0 <= label <= 1_000_000 for label in predictions):
            raise ValueError("Clustering requires nonnegative integer cluster IDs for every row; noise or unassigned rows are unsupported.")
        labels = np.asarray(predictions)
        unique, counts = np.unique(labels, return_counts=True)
        if len(unique) > 1000:
            raise ValueError("Clustering supports at most 1,000 distinct clusters per evaluation.")
        projection["clusters"] = [{"label": int(label), "count": int(count)} for label, count in zip(unique, counts)]
        projection["axes"] = axes + ([""] if len(axes) == 1 else [])
        projection["points"] = [{"row": int(i), "x": float(reference[i, 0]), "y": float(reference[i, 1]) if reference.shape[1] > 1 else 0, "cluster": int(labels[i])} for i in display]
        if unscored:
            score = None
        else:
            if not 2 <= len(np.unique(labels[sample])) < len(sample):
                raise ValueError("Clustering metrics need between 2 and n-1 clusters in the fixed evaluation sample.")
            score = float(silhouette_score(reference[sample], labels[sample]) if metadata["metric"] == "silhouette" else davies_bouldin_score(dense_features(reference[sample]), labels[sample]))
    elif kind == "reduction":
        variance_mode = metadata.get("reductionMode") == "variance"
        expected = None
        if variance_mode:
            target = metadata["varianceTarget"]
            pca = PCA(n_components=target if target < 1 else None, svd_solver="full").fit(dense_features(train))
            dimensions = pca.n_components_
            retained = float(pca.explained_variance_ratio_.sum())
            if not np.isfinite(retained):
                raise ValueError("PCA needs nonzero variance in the training inputs.")
            expected = pca.transform(dense_features(reference))
            extra = {"cumulativeVariance": retained}
        else:
            dimensions = metadata["dimensions"]
            if not 1 <= dimensions < train.shape[1]:
                raise ValueError(f"Output dimensions must be fewer than the {train.shape[1]} encoded training features.")
        if any(not isinstance(row, list) or len(row) != dimensions or any(type(v) not in (int, float) for v in row) for row in predictions):
            raise ValueError(f"Expected exactly {dimensions} numeric dimensions for every evaluation row.")
        embedding = np.asarray(predictions, dtype=float)
        if not np.isfinite(embedding).all():
            raise ValueError("Embedding coordinates must be finite numbers.")
        if expected is not None:
            # Verify the PCA geometry independently, allowing equivalent signs/rotations.
            checked = sample
            left, _, right = np.linalg.svd(expected[checked].T @ embedding[checked], full_matrices=False)
            if not np.allclose(embedding, expected @ (left @ right), rtol=1e-5, atol=1e-6):
                raise ValueError("Variance-target outputs must come from PCA fitted on the training-fitted reference encoding, with whitening disabled.")
        projection.update({"dimensions": int(dimensions), **extra, "axes": ["Dimension 1", "Dimension 2" if dimensions > 1 else ""]})
        projection["points"] = [{"row": int(i), "x": float(embedding[i, 0]), "y": float(embedding[i, 1]) if dimensions > 1 else 0} for i in display]
        score = None if unscored else float(trustworthiness(reference[sample], embedding[sample], n_neighbors=min(5, (len(sample) - 1) // 2)))
        extra["dimensions"] = int(dimensions)
    else:
        raise ValueError("Unsupported unsupervised learning mode.")
    if score is not None and not np.isfinite(score):
        raise ValueError("The unsupervised metric is undefined for these outputs.")
    return {"score": score, "scoringRows": len(sample), "projection": projection, **extra}


if __name__ == "__main__":
    try:
        data, work = Path(sys.argv[1]), Path(sys.argv[2])
        result = evaluate(data, json.loads((work / "predictions.json").read_text()), int(sys.argv[3]), "--unscored" in sys.argv)
        (work / "score.json").write_text(json.dumps(result, allow_nan=False))
    except (ValueError, TypeError) as error:
        sys.exit(str(error))
