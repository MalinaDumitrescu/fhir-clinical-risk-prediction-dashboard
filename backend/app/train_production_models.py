import argparse
import json
from datetime import datetime, timezone

import joblib
import mlflow
import mlflow.sklearn
import optuna
import pandas as pd

from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, train_test_split

from backend.app.config import (
    FEATURES_CSV,
    MODEL_PATH,
    METRICS_PATH,
    MODEL_COMPARISON_PATH,
    MODEL_DETAILS_PATH,
    OPTUNA_TRIALS_CSV,
    MODELS_DIR,
    MLFLOW_DIR,
)
from backend.app.evaluation_utils import evaluate_binary_classifier
from backend.app.features import build_feature_dataframe, PREDICTION_WINDOW_HOURS
from backend.app.train_advanced_models import (
    TARGET_COL,
    RANDOM_STATE,
    prepare_xy,
    objective,
    get_best_trial_per_model,
    build_model_from_params,
    build_ensembles,
    make_pipeline,
    score_for_selection,
    get_available_model_names,
)


def load_or_rebuild_features():
    df = build_feature_dataframe()
    df.to_csv(FEATURES_CSV, index=False)
    return df


def get_calibration_cv(y):
    min_class_count = int(y.value_counts().min())
    n_splits = min(3, min_class_count)

    if n_splits < 2:
        return "prefit"

    return StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RANDOM_STATE,
    )


def make_calibrated_classifier(estimator, cv):
    try:
        return CalibratedClassifierCV(
            estimator=estimator,
            method="sigmoid",
            cv=cv,
        )
    except TypeError:
        return CalibratedClassifierCV(
            base_estimator=estimator,
            method="sigmoid",
            cv=cv,
        )


def flatten_numeric_metrics(metrics):
    flat = {}

    for key, value in metrics.items():
        if isinstance(value, (int, float)) and value is not None:
            flat[key] = float(value)

    return flat


def log_candidate_to_mlflow(model_name, params, metrics, model_file):
    with mlflow.start_run(run_name=model_name):
        mlflow.set_tag("project", "clinical-risk-fhir-dashboard")
        mlflow.set_tag("model_name", model_name)
        mlflow.set_tag("stage", "candidate")

        if params:
            mlflow.log_params(params)

        mlflow.log_metrics(flatten_numeric_metrics(metrics))

        if model_file:
            mlflow.log_artifact(str(model_file))


def train_production(n_trials):
    mlflow.set_tracking_uri("sqlite:///" + str(MLFLOW_DIR / "mlflow.db"))
    mlflow.set_experiment("clinical-risk-fhir-dashboard")

    df = load_or_rebuild_features()

    X, y, feature_cols, numeric_cols, categorical_cols = prepare_xy(df)

    print("Dataset shape:", df.shape)
    print("Target distribution:")
    print(y.value_counts())
    print("Available model families:", get_available_model_names())

    stratify = y if y.value_counts().min() >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=stratify,
    )

    study = optuna.create_study(
        direction="maximize",
        study_name="clinical_risk_production_tuning",
    )

    study.optimize(
        lambda trial: objective(
            trial,
            X_train,
            y_train,
            numeric_cols,
            categorical_cols,
        ),
        n_trials=n_trials,
    )

    trials_df = study.trials_dataframe()
    trials_df.to_csv(OPTUNA_TRIALS_CSV, index=False)

    best_trials_by_model = get_best_trial_per_model(study)

    comparison = []
    trained_artifacts = {}

    for model_name, trial in best_trials_by_model.items():
        print(f"Training candidate: {model_name}")

        model = build_model_from_params(trial.params, y_train)
        pipeline = make_pipeline(model, numeric_cols, categorical_cols)
        pipeline.fit(X_train, y_train)

        metrics = evaluate_binary_classifier(pipeline, X_test, y_test)

        artifact = {
            "pipeline": pipeline,
            "prediction_pipeline": pipeline,
            "model_name": model_name,
            "feature_columns": feature_cols,
            "target_column": TARGET_COL,
            "numeric_columns": numeric_cols,
            "categorical_columns": categorical_cols,
            "optuna_cv_score": float(trial.value),
            "optuna_params": trial.params,
            "calibrated": False,
        }

        model_file = MODELS_DIR / f"{model_name}.joblib"
        joblib.dump(artifact, model_file)

        trained_artifacts[model_name] = artifact

        row = {
            "model_name": model_name,
            "model_file": str(model_file),
            "optuna_cv_roc_auc": float(trial.value),
            **metrics,
        }

        comparison.append(row)

        log_candidate_to_mlflow(
            model_name=model_name,
            params=trial.params,
            metrics=metrics,
            model_file=model_file,
        )

    ensemble_models = build_ensembles(best_trials_by_model, y_train)

    for ensemble_name, ensemble_model in ensemble_models.items():
        print(f"Training ensemble: {ensemble_name}")

        pipeline = make_pipeline(ensemble_model, numeric_cols, categorical_cols)
        pipeline.fit(X_train, y_train)

        metrics = evaluate_binary_classifier(pipeline, X_test, y_test)

        artifact = {
            "pipeline": pipeline,
            "prediction_pipeline": pipeline,
            "model_name": ensemble_name,
            "feature_columns": feature_cols,
            "target_column": TARGET_COL,
            "numeric_columns": numeric_cols,
            "categorical_columns": categorical_cols,
            "optuna_cv_score": None,
            "optuna_params": None,
            "calibrated": False,
        }

        model_file = MODELS_DIR / f"{ensemble_name}.joblib"
        joblib.dump(artifact, model_file)

        trained_artifacts[ensemble_name] = artifact

        row = {
            "model_name": ensemble_name,
            "model_file": str(model_file),
            "optuna_cv_roc_auc": None,
            **metrics,
        }

        comparison.append(row)

        log_candidate_to_mlflow(
            model_name=ensemble_name,
            params={},
            metrics=metrics,
            model_file=model_file,
        )

    comparison = sorted(
        comparison,
        key=lambda item: score_for_selection(item),
        reverse=True,
    )

    best_model_name = comparison[0]["model_name"]
    best_artifact = trained_artifacts[best_model_name]
    best_base_pipeline = best_artifact["pipeline"]

    print("Best uncalibrated model:", best_model_name)

    calibration_cv = get_calibration_cv(y_train)

    if calibration_cv == "prefit":
        calibrated_pipeline = best_base_pipeline
        calibrated = False
    else:
        fresh_model = build_model_from_params(
            best_artifact.get("optuna_params") or {},
            y_train,
        ) if best_artifact.get("optuna_params") else best_base_pipeline.named_steps["model"]

        fresh_pipeline = make_pipeline(fresh_model, numeric_cols, categorical_cols)

        calibrated_pipeline = make_calibrated_classifier(
            fresh_pipeline,
            cv=calibration_cv,
        )

        calibrated_pipeline.fit(X_train, y_train)
        calibrated = True

    calibrated_metrics = evaluate_binary_classifier(
        calibrated_pipeline,
        X_test,
        y_test,
    )

    training_date = datetime.now(timezone.utc).isoformat()

    best_trial_details = {
        "number": int(study.best_trial.number),
        "value": float(study.best_trial.value),
        "params": study.best_trial.params,
        "datetime": study.best_trial.datetime_start.isoformat() if study.best_trial.datetime_start else None,
    }

    model_details = {
        "model_name": best_model_name,
        "active_prediction_pipeline": "calibrated_pipeline" if calibrated else "base_pipeline",
        "calibrated": calibrated,
        "target": TARGET_COL,
        "target_definition": "Long ICU stay, defined as ICU length of stay >= 3 days; fallback median split only if demo data has one class.",
        "prediction_window_hours": PREDICTION_WINDOW_HOURS,
        "leakage_policy": "Only features with timestamps inside the first 24 hours after first encounter start are used.",
        "n_patients": int(len(df)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_features": int(len(feature_cols)),
        "feature_columns": feature_cols,
        "training_date_utc": training_date,
        "selection_metric": "ROC-AUC if available, else average precision, else balanced accuracy",
        "best_optuna_trial": best_trial_details,
    }

    active_artifact = {
        **best_artifact,
        "prediction_pipeline": calibrated_pipeline,
        "model_name": best_model_name,
        "calibrated": calibrated,
        "model_details": model_details,
        "active_metrics": calibrated_metrics,
    }

    active_metrics = {
        "model_name": best_model_name,
        "calibrated": calibrated,
        "model_details": model_details,
        **calibrated_metrics,
    }

    joblib.dump(active_artifact, MODEL_PATH)

    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(active_metrics, file, indent=4)

    with open(MODEL_DETAILS_PATH, "w", encoding="utf-8") as file:
        json.dump(model_details, file, indent=4)

    with open(MODEL_COMPARISON_PATH, "w", encoding="utf-8") as file:
        json.dump(
            {
                "best_model_name": best_model_name,
                "selection_metric": model_details["selection_metric"],
                "models": comparison,
            },
            file,
            indent=4,
        )

    with mlflow.start_run(run_name=f"active_{best_model_name}_calibrated"):
        mlflow.set_tag("project", "clinical-risk-fhir-dashboard")
        mlflow.set_tag("stage", "active")
        mlflow.set_tag("model_name", best_model_name)
        mlflow.set_tag("calibrated", str(calibrated))

        mlflow.log_params(
            {
                "model_name": best_model_name,
                "target": TARGET_COL,
                "prediction_window_hours": PREDICTION_WINDOW_HOURS,
                "n_features": len(feature_cols),
                "n_train": len(X_train),
                "n_test": len(X_test),
            }
        )

        mlflow.log_metrics(flatten_numeric_metrics(active_metrics))
        mlflow.sklearn.log_model(
            calibrated_pipeline,
            "calibrated_prediction_pipeline",
            skops_trusted_types=[
                "numpy.dtype",
                "sklearn.calibration._CalibratedClassifier",
                "sklearn.calibration._SigmoidCalibration",
                "sklearn.model_selection._split.StratifiedKFold",
                "xgboost.core.Booster",
                "xgboost.sklearn.XGBClassifier",
            ],
        )
        mlflow.log_artifact(str(METRICS_PATH))
        mlflow.log_artifact(str(MODEL_COMPARISON_PATH))
        mlflow.log_artifact(str(MODEL_DETAILS_PATH))

    print("Saved active model:", MODEL_PATH)
    print("Saved metrics:", METRICS_PATH)
    print("Saved model comparison:", MODEL_COMPARISON_PATH)
    print("Saved model details:", MODEL_DETAILS_PATH)
    print("Saved MLflow runs in:", MLFLOW_DIR)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--n-trials",
        type=int,
        default=40,
        help="Number of Optuna trials.",
    )

    args = parser.parse_args()

    train_production(n_trials=args.n_trials)


if __name__ == "__main__":
    main()
