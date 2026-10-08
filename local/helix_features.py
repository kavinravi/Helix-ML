"""Training-fitted CSV encoding, shared by candidates and the trusted scorer."""
import os

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


def read_features(path, features):
    return pd.read_csv(path, dtype=str, keep_default_na=False, usecols=features)[features].apply(lambda col: col.str.strip())


def numeric_values(frame):
    values = frame.replace("", np.nan).apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    if np.isinf(values).any():
        raise ValueError("Numeric features must be finite; replace infinity before training.")
    return values


def empty_values(frame):
    return np.zeros(frame.shape)


def preprocessor(train):
    transforms = []
    for index, column in enumerate(train.columns):
        present = train[column][train[column] != ""]
        if present.empty:
            transform = FunctionTransformer(empty_values, feature_names_out="one-to-one")
        else:
            try:
                pd.to_numeric(present, errors="raise")
                transform = make_pipeline(FunctionTransformer(numeric_values, feature_names_out="one-to-one"),
                                          SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler())
            except ValueError:
                # shortcut: repeated short strings are categories; add explicit column roles if this inference is insufficient.
                prose = present.str.split().str.len().max() >= 3 or (present.nunique() > 20 and present.str.contains(r"\s").any())
                if prose and present.str.contains(r"\w", regex=True).any():
                    transforms.append((f"f{index}", TfidfVectorizer(token_pattern=r"(?u)\b\w+\b"), column))
                    continue
                transform = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
        transforms.append((f"f{index}", transform, [column]))
    return ColumnTransformer(transforms, sparse_threshold=1.0)


def dense_features(values):
    # Full PCA and Davies–Bouldin require dense input. Check the expanded width before allocating it.
    limit = int(os.environ.get("HELIX_TRAIN_MEMORY_MB") or (1200 if os.environ.get("HELIX_HOSTED_WORKER") else 3072)) * 1024 ** 2
    estimate = 192 * 1024 ** 2 + values.shape[0] * values.shape[1] * 8 * 6
    if estimate > limit:
        raise ValueError(f"Dense encoding needs about {estimate // 1024 ** 2:,} MiB for {values.shape[0]:,} rows × {values.shape[1]:,} encoded features; available memory is {limit // 1024 ** 2:,} MiB. Exclude IDs or unused columns, use a sparse-compatible model with fixed dimensions, or increase worker memory.")
    return values.toarray() if sparse.issparse(values) else values
