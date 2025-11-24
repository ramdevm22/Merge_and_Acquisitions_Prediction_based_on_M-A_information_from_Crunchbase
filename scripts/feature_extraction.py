"""
Feature extraction and preprocessing pipeline for M&A acquisition prediction.

This module:
- Loads raw train/test data from the data/ directory.
- Builds a preprocessing pipeline using:
    * SimpleImputer for missing-value handling
    * StandardScaler for numeric feature scaling
    * OneHotEncoder for categorical encoding
- Persists the fitted preprocessor for re-use at inference time.
- Outputs train/test design matrices as CSV files in data/processed/.
"""

import os
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# Paths
ROOT_DIR = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR = os.path.join(ROOT_DIR, "data")
PROC_DIR = os.path.join(DATA_DIR, "processed")
MODEL_DIR = os.path.join(ROOT_DIR, "models")

os.makedirs(PROC_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

TARGET_COL = "is_acquired"


def _build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Build a scikit-learn ColumnTransformer for the input features."""
    numeric_features = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [
    c for c in X.columns
    if c not in numeric_features and c != "company_id"
]


    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor


def do_feature_extraction() -> None:
    """Fit preprocessing pipeline and export processed train/test sets."""
    train_path = os.path.join(DATA_DIR, "train_data.csv")
    test_path = os.path.join(DATA_DIR, "test_data.csv")

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    if TARGET_COL not in train.columns:
        raise ValueError(f"Expected target column '{TARGET_COL}' in train_data.csv")

    # Split into features/target
    y_train = train[TARGET_COL]
    X_train = train.drop(columns=[TARGET_COL])

    if TARGET_COL not in test.columns:
        raise ValueError(f"Expected target column '{TARGET_COL}' in test_data.csv")

    y_test = test[TARGET_COL]
    X_test = test.drop(columns=[TARGET_COL])

    # Build & fit preprocessor
    preprocessor = _build_preprocessor(X_train)
    preprocessor.fit(X_train)

    # Persist preprocessor for reuse in the Streamlit UI / batch inference
    preprocessor_path = os.path.join(MODEL_DIR, "preprocessor.joblib")
    joblib.dump(preprocessor, preprocessor_path)

    # Transform train/test
    Xt_train = preprocessor.transform(X_train)
    Xt_test = preprocessor.transform(X_test)

    # Convert to dense DataFrames (RandomForest can work with dense matrices)
    import numpy as np

    Xt_train = Xt_train.toarray() if hasattr(Xt_train, "toarray") else Xt_train
    Xt_test = Xt_test.toarray() if hasattr(Xt_test, "toarray") else Xt_test

    train_out = pd.DataFrame(Xt_train)
    train_out["y"] = y_train.values
    train_out.to_csv(os.path.join(PROC_DIR, "train_processed.csv"), index=False)

    test_out = pd.DataFrame(Xt_test)
    test_out["y"] = y_test.values
    test_out.to_csv(os.path.join(PROC_DIR, "test_processed.csv"), index=False)

    print("Feature extraction complete. Processed files written to data/processed/.")
    print(f"Preprocessor saved to {preprocessor_path}")


if __name__ == "__main__":  # pragma: no cover
    do_feature_extraction()
