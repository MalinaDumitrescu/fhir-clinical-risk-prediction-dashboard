import json

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from backend.app.config import FEATURES_CSV, MODEL_PATH, METRICS_PATH
from backend.app.features import build_feature_dataframe


TARGET_COL = "target_long_icu_stay"


def load_or_create_features():
    if FEATURES_CSV.exists():
        return pd.read_csv(FEATURES_CSV)

    df = build_feature_dataframe()
    df.to_csv(FEATURES_CSV, index=False)
    return df


def train():
    df = load_or_create_features()

    if TARGET_COL not in df.columns:
        raise ValueError(f"Missing target column: {TARGET_COL}")

    if df[TARGET_COL].nunique() < 2:
        raise ValueError(
            "The target has only one class. "
            "This can happen with a very small demo dataset."
        )

    drop_cols = [
        "patient_id",
        "birth_date",
        "deceased",
        "icu_los_days",
        "hospital_los_days",
        TARGET_COL,
    ]

    feature_cols = [col for col in df.columns if col not in drop_cols]

    X = df[feature_cols]
    y = df[TARGET_COL]

    # Robustly determine numeric vs categorical columns using pandas dtypes
    numeric_cols = X.select_dtypes(include="number").columns.tolist()
    categorical_cols = X.select_dtypes(exclude="number").columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    transformers = []
    if numeric_cols:
        transformers.append(("numeric", numeric_pipeline, numeric_cols))
    if categorical_cols:
        transformers.append(("categorical", categorical_pipeline, categorical_cols))

    if not transformers:
        raise ValueError("No feature columns detected after dropping non-feature columns")

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=5,
        random_state=42,
        class_weight="balanced",
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    stratify = y if y.value_counts().min() >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=stratify,
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    # Compute probability scores only when available and binary classification
    y_proba = None
    try:
        proba = pipeline.predict_proba(X_test)
        if proba is not None and proba.shape[1] == 2:
            y_proba = proba[:, 1]
    except Exception:
        y_proba = None

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(
            y_test,
            y_pred,
            output_dict=True,
            zero_division=0,
        ),
        "average_precision": None,
        "roc_auc": None,
    }

    if y_proba is not None and y_test.nunique() == 2:
        metrics["average_precision"] = float(average_precision_score(y_test, y_proba))
        metrics["roc_auc"] = float(roc_auc_score(y_test, y_proba))
    else:
        print("Skipping average_precision/roc_auc because probability estimates are unavailable or not binary")

    artifact = {
        "pipeline": pipeline,
        "feature_columns": feature_cols,
        "target_column": TARGET_COL,
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
    }

    joblib.dump(artifact, MODEL_PATH)

    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    print(f"Saved model to: {MODEL_PATH}")
    print(f"Saved metrics to: {METRICS_PATH}")
    print()
    print("Metrics:")
    print(json.dumps(metrics, indent=4))


if __name__ == "__main__":
    train()