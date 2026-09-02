import argparse
import json
from datetime import datetime, timezone

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import optuna
import pandas as pd

from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
)

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
from backend.app.features import (
    build_feature_dataframe,
    PREDICTION_WINDOW_HOURS,
)
from backend.app.train_advanced_models import (
    TARGET_COL,
    RANDOM_STATE,
    prepare_xy,
    objective,
    get_best_trial_per_model,
    build_model_from_params,
    build_ensembles,
    build_selected_model,
    make_pipeline,
    score_for_selection,
    get_available_model_names,
    split_train_validation_test,
)


def load_or_rebuild_features():
    """
    Always rebuild features from the raw FHIR resources.

    This prevents stale features.csv files from silently being reused after
    changes to cohort construction, target definition, temporal filtering,
    code mappings, or unit handling.
    """
    df = build_feature_dataframe()
    df.to_csv(FEATURES_CSV, index=False)
    return df


def get_calibration_cv(y):
    """
    Build a small stratified CV object suitable for probability calibration.
    """
    min_class_count = int(y.value_counts().min())

    if min_class_count < 2:
        raise ValueError(
            "Not enough samples per class for probability calibration."
        )

    return StratifiedKFold(
        n_splits=min(3, min_class_count),
        shuffle=True,
        random_state=RANDOM_STATE,
    )


def make_calibrated_classifier(estimator, cv):
    """
    Compatibility helper for different scikit-learn versions.
    """
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
    """
    Keep only scalar numeric values that MLflow can log directly.
    """
    return {
        key: float(value)
        for key, value in metrics.items()
        if isinstance(value, (int, float))
        and value is not None
    }


def select_classification_threshold(
    y_true,
    probabilities,
):
    """
    Select a binary decision threshold using balanced accuracy.

    Candidate thresholds are placed midway between observed predicted
    probabilities rather than scanning an arbitrary fixed grid.

    When several thresholds have identical balanced accuracy, choose the
    threshold closest to 0.5 as a deterministic and non-extreme tie-breaker.
    """
    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    y_true = np.asarray(y_true)

    unique_probabilities = np.unique(
        probabilities
    )

    if len(unique_probabilities) < 2:
        return 0.5, 0.5

    candidate_thresholds = (
        unique_probabilities[:-1]
        + unique_probabilities[1:]
    ) / 2.0

    results = []

    for threshold in candidate_thresholds:
        predictions = (
            probabilities >= threshold
        ).astype(int)

        score = balanced_accuracy_score(
            y_true,
            predictions,
        )

        results.append(
            (
                float(score),
                float(threshold),
            )
        )

    best_score = max(
        score
        for score, _ in results
    )

    best_thresholds = [
        threshold
        for score, threshold in results
        if np.isclose(
            score,
            best_score,
        )
    ]

    best_threshold = min(
        best_thresholds,
        key=lambda threshold: abs(
            threshold - 0.5
        ),
    )

    return (
        float(best_threshold),
        float(best_score),
    )


def log_candidate_to_mlflow(
    model_name,
    params,
    metrics,
    model_file,
):
    """
    Log validation-stage candidate information.

    Candidate model files themselves are stored locally with joblib. MLflow
    receives parameters, metrics and the local artifact file.
    """
    with mlflow.start_run(
        run_name=model_name
    ):
        mlflow.set_tag(
            "project",
            "clinical-risk-fhir-dashboard",
        )
        mlflow.set_tag(
            "model_name",
            model_name,
        )
        mlflow.set_tag(
            "stage",
            "candidate_validation",
        )

        if params:
            mlflow.log_params(params)

        mlflow.log_metrics(
            flatten_numeric_metrics(metrics)
        )

        if model_file:
            mlflow.log_artifact(
                str(model_file)
            )


def train_production(n_trials):
    # ---------------------------------------------------------
    # MLflow setup
    # ---------------------------------------------------------

    mlflow.set_tracking_uri(
        "sqlite:///"
        + str(MLFLOW_DIR / "mlflow.db")
    )

    mlflow.set_experiment(
        "clinical-risk-fhir-dashboard"
    )

    # ---------------------------------------------------------
    # Feature generation
    # ---------------------------------------------------------

    df = load_or_rebuild_features()

    (
        X,
        y,
        feature_cols,
        numeric_cols,
        categorical_cols,
    ) = prepare_xy(df)

    print(
        "Dataset shape:",
        df.shape,
    )

    print(
        "Target distribution:"
    )
    print(
        y.value_counts()
    )

    print(
        "Available model families:",
        get_available_model_names(),
    )

    # ---------------------------------------------------------
    # TRAIN / VALIDATION / TEST
    # ---------------------------------------------------------

    (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
    ) = split_train_validation_test(
        X,
        y,
    )

    print(
        "Train size:",
        len(X_train),
    )
    print(
        "Validation size:",
        len(X_validation),
    )
    print(
        "Test size:",
        len(X_test),
    )

    print(
        "Train target distribution:"
    )
    print(
        y_train.value_counts()
    )

    print(
        "Validation target distribution:"
    )
    print(
        y_validation.value_counts()
    )

    print(
        "Test target distribution:"
    )
    print(
        y_test.value_counts()
    )

    # ---------------------------------------------------------
    # Optuna
    #
    # Hyperparameter optimization uses TRAIN only.
    #
    # The seeded sampler makes repeated runs reproducible when
    # the dataset, code and environment are unchanged.
    # ---------------------------------------------------------

    study = optuna.create_study(
        direction="maximize",
        study_name=(
            "clinical_risk_production_tuning"
        ),
        sampler=optuna.samplers.TPESampler(
            seed=RANDOM_STATE,
        ),
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

    study.trials_dataframe().to_csv(
        OPTUNA_TRIALS_CSV,
        index=False,
    )

    best_trials_by_model = (
        get_best_trial_per_model(
            study
        )
    )

    # ---------------------------------------------------------
    # Candidate model evaluation
    #
    # TRAIN:
    #   fit candidate
    #
    # VALIDATION:
    #   compare candidate models
    #
    # TEST:
    #   remains untouched
    # ---------------------------------------------------------

    comparison = []
    trained_artifacts = {}

    for (
        model_name,
        trial,
    ) in best_trials_by_model.items():

        print(
            f"Training candidate: "
            f"{model_name}"
        )

        model = build_model_from_params(
            trial.params,
            y_train,
        )

        pipeline = make_pipeline(
            model,
            numeric_cols,
            categorical_cols,
        )

        pipeline.fit(
            X_train,
            y_train,
        )

        metrics = (
            evaluate_binary_classifier(
                pipeline,
                X_validation,
                y_validation,
            )
        )

        artifact = {
            "pipeline": pipeline,
            "prediction_pipeline": pipeline,
            "model_name": model_name,
            "feature_columns": feature_cols,
            "target_column": TARGET_COL,
            "numeric_columns": numeric_cols,
            "categorical_columns": (
                categorical_cols
            ),
            "optuna_cv_score": float(
                trial.value
            ),
            "optuna_params": (
                trial.params
            ),
            "calibrated": False,
        }

        model_file = (
            MODELS_DIR
            / f"{model_name}.joblib"
        )

        joblib.dump(
            artifact,
            model_file,
        )

        trained_artifacts[
            model_name
        ] = artifact

        row = {
            "model_name": model_name,
            "model_file": str(
                model_file
            ),
            "evaluation_split": (
                "validation"
            ),
            "optuna_cv_roc_auc": float(
                trial.value
            ),
            **metrics,
        }

        comparison.append(row)

        log_candidate_to_mlflow(
            model_name,
            trial.params,
            metrics,
            model_file,
        )

    # ---------------------------------------------------------
    # Ensembles
    # ---------------------------------------------------------

    ensembles = build_ensembles(
        best_trials_by_model,
        y_train,
    )

    for (
        ensemble_name,
        ensemble_model,
    ) in ensembles.items():

        print(
            f"Training ensemble: "
            f"{ensemble_name}"
        )

        pipeline = make_pipeline(
            ensemble_model,
            numeric_cols,
            categorical_cols,
        )

        pipeline.fit(
            X_train,
            y_train,
        )

        metrics = (
            evaluate_binary_classifier(
                pipeline,
                X_validation,
                y_validation,
            )
        )

        artifact = {
            "pipeline": pipeline,
            "prediction_pipeline": pipeline,
            "model_name": (
                ensemble_name
            ),
            "feature_columns": feature_cols,
            "target_column": TARGET_COL,
            "numeric_columns": numeric_cols,
            "categorical_columns": (
                categorical_cols
            ),
            "optuna_cv_score": None,
            "optuna_params": None,
            "calibrated": False,
        }

        model_file = (
            MODELS_DIR
            / f"{ensemble_name}.joblib"
        )

        joblib.dump(
            artifact,
            model_file,
        )

        trained_artifacts[
            ensemble_name
        ] = artifact

        row = {
            "model_name": (
                ensemble_name
            ),
            "model_file": str(
                model_file
            ),
            "evaluation_split": (
                "validation"
            ),
            "optuna_cv_roc_auc": None,
            **metrics,
        }

        comparison.append(row)

        log_candidate_to_mlflow(
            ensemble_name,
            {},
            metrics,
            model_file,
        )

    # ---------------------------------------------------------
    # Select best model using VALIDATION only
    # ---------------------------------------------------------

    comparison = sorted(
        comparison,
        key=score_for_selection,
        reverse=True,
    )

    best_model_name = (
        comparison[0]["model_name"]
    )

    best_artifact = (
        trained_artifacts[
            best_model_name
        ]
    )

    print(
        "Best model selected "
        "on validation:",
        best_model_name,
    )

    # ---------------------------------------------------------
    # Development cohort
    #
    # Once model family / hyperparameters are selected,
    # TRAIN + VALIDATION can be combined.
    #
    # TEST still remains untouched.
    # ---------------------------------------------------------

    X_train_validation = pd.concat(
        [
            X_train,
            X_validation,
        ],
        axis=0,
    )

    y_train_validation = pd.concat(
        [
            y_train,
            y_validation,
        ],
        axis=0,
    )

    # ---------------------------------------------------------
    # Threshold selection
    #
    # Generate OUT-OF-FOLD calibrated probabilities across the
    # entire 80-patient development cohort.
    #
    # This avoids:
    #   - tuning on TEST;
    #   - choosing the threshold from only ~20 validation cases;
    #   - the previous arbitrary 0.05 tie artifact.
    # ---------------------------------------------------------

    threshold_model = (
        build_selected_model(
            best_model_name,
            best_artifact,
            best_trials_by_model,
            y_train_validation,
        )
    )

    threshold_base_pipeline = (
        make_pipeline(
            threshold_model,
            numeric_cols,
            categorical_cols,
        )
    )

    threshold_calibrated_pipeline = (
        make_calibrated_classifier(
            threshold_base_pipeline,
            cv=get_calibration_cv(
                y_train_validation
            ),
        )
    )

    outer_cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    oof_probabilities = (
        cross_val_predict(
            threshold_calibrated_pipeline,
            X_train_validation,
            y_train_validation,
            cv=outer_cv,
            method="predict_proba",
            n_jobs=-1,
        )[:, 1]
    )

    (
        decision_threshold,
        oof_threshold_score,
    ) = select_classification_threshold(
        y_train_validation,
        oof_probabilities,
    )

    print(
        "Selected OOF "
        "decision threshold:",
        round(
            decision_threshold,
            3,
        ),
    )

    print(
        "OOF balanced accuracy "
        "at threshold:",
        round(
            oof_threshold_score,
            3,
        ),
    )

    # ---------------------------------------------------------
    # Final uncalibrated model
    #
    # Useful for SHAP / tree-based interpretation because
    # calibrated wrappers are less convenient to explain.
    # ---------------------------------------------------------

    final_base_model = (
        build_selected_model(
            best_model_name,
            best_artifact,
            best_trials_by_model,
            y_train_validation,
        )
    )

    final_base_pipeline = (
        make_pipeline(
            final_base_model,
            numeric_cols,
            categorical_cols,
        )
    )

    final_base_pipeline.fit(
        X_train_validation,
        y_train_validation,
    )

    # ---------------------------------------------------------
    # Final calibrated prediction pipeline
    # ---------------------------------------------------------

    calibration_model = (
        build_selected_model(
            best_model_name,
            best_artifact,
            best_trials_by_model,
            y_train_validation,
        )
    )

    calibration_pipeline = (
        make_pipeline(
            calibration_model,
            numeric_cols,
            categorical_cols,
        )
    )

    calibrated_pipeline = (
        make_calibrated_classifier(
            calibration_pipeline,
            cv=get_calibration_cv(
                y_train_validation
            ),
        )
    )

    calibrated_pipeline.fit(
        X_train_validation,
        y_train_validation,
    )

    # ---------------------------------------------------------
    # First and only final evaluation on TEST
    #
    # Neither model family, hyperparameters nor threshold are
    # selected using the test labels.
    # ---------------------------------------------------------

    test_metrics = (
        evaluate_binary_classifier(
            calibrated_pipeline,
            X_test,
            y_test,
            decision_threshold=(
                decision_threshold
            ),
        )
    )

    # ---------------------------------------------------------
    # Metadata
    # ---------------------------------------------------------

    training_date = (
        datetime.now(
            timezone.utc
        ).isoformat()
    )

    model_details = {
        "model_name": (
            best_model_name
        ),
        "active_prediction_pipeline": (
            "calibrated_pipeline"
        ),
        "calibrated": True,

        "target": TARGET_COL,

        "target_definition": (
            "Long ICU stay: total ICU "
            "duration >= 3 days among ICU "
            "encounters whose partOf reference "
            "is the patient's index hospital "
            "admission."
        ),

        "index_admission_definition": (
            "Earliest hospital Encounter "
            "per patient."
        ),

        "prediction_window_hours": (
            PREDICTION_WINDOW_HOURS
        ),

        "leakage_policy": (
            "Predictors must occur during the "
            "first 24 hours of the index hospital "
            "admission and, when an encounter/"
            "context reference is present, must "
            "belong to that admission or one of "
            "its linked ICU encounters."
        ),

        "observation_mapping": (
            "Explicit MIMIC item-code mapping "
            "with unit validation; temperature "
            "is normalized to Celsius; broad "
            "display-name keyword matching is "
            "not used."
        ),

        "n_patients": int(
            len(df)
        ),
        "n_train": int(
            len(X_train)
        ),
        "n_validation": int(
            len(X_validation)
        ),
        "n_test": int(
            len(X_test)
        ),
        "n_development": int(
            len(
                X_train_validation
            )
        ),

        "n_features": int(
            len(feature_cols)
        ),

        "feature_columns": (
            feature_cols
        ),

        "training_date_utc": (
            training_date
        ),

        "selection_metric": (
            "ROC-AUC if available, else "
            "average precision, else "
            "balanced accuracy"
        ),

        "selection_split": (
            "validation"
        ),

        "final_evaluation_split": (
            "untouched test"
        ),

        "decision_threshold": float(
            decision_threshold
        ),

        "threshold_selection_method": (
            "5-fold out-of-fold calibrated "
            "predictions on train+validation"
        ),

        "threshold_selection_metric": (
            "balanced_accuracy"
        ),

        "oof_balanced_accuracy_at_threshold": (
            float(
                oof_threshold_score
            )
        ),

        "random_state": (
            RANDOM_STATE
        ),

        "optuna_sampler": (
            "TPESampler"
        ),

        "optuna_trials": int(
            n_trials
        ),
    }

    # ---------------------------------------------------------
    # Save active deployable artifact
    # ---------------------------------------------------------

    active_artifact = {
        **best_artifact,

        "pipeline": (
            final_base_pipeline
        ),

        "prediction_pipeline": (
            calibrated_pipeline
        ),

        "model_name": (
            best_model_name
        ),

        "calibrated": True,

        "decision_threshold": float(
            decision_threshold
        ),

        "model_details": (
            model_details
        ),

        "active_metrics": (
            test_metrics
        ),
    }

    joblib.dump(
        active_artifact,
        MODEL_PATH,
    )

    # ---------------------------------------------------------
    # Save test metrics
    # ---------------------------------------------------------

    active_metrics = {
        "model_name": (
            best_model_name
        ),

        "calibrated": True,

        "evaluation_split": (
            "test"
        ),

        "decision_threshold": float(
            decision_threshold
        ),

        "model_details": (
            model_details
        ),

        **test_metrics,
    }

    with open(
        METRICS_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            active_metrics,
            file,
            indent=4,
        )

    with open(
        MODEL_DETAILS_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            model_details,
            file,
            indent=4,
        )

    with open(
        MODEL_COMPARISON_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                "best_model_name": (
                    best_model_name
                ),
                "selection_split": (
                    "validation"
                ),
                "selection_metric": (
                    model_details[
                        "selection_metric"
                    ]
                ),
                "models": comparison,
            },
            file,
            indent=4,
        )

    # ---------------------------------------------------------
    # MLflow active model logging
    # ---------------------------------------------------------

    with mlflow.start_run(
        run_name=(
            f"active_"
            f"{best_model_name}"
            f"_calibrated"
        )
    ):
        mlflow.set_tag(
            "project",
            "clinical-risk-fhir-dashboard",
        )

        mlflow.set_tag(
            "stage",
            "active_test",
        )

        mlflow.set_tag(
            "model_name",
            best_model_name,
        )

        mlflow.set_tag(
            "calibrated",
            "True",
        )

        mlflow.log_params(
            {
                "model_name": (
                    best_model_name
                ),

                "target": (
                    TARGET_COL
                ),

                "prediction_window_hours": (
                    PREDICTION_WINDOW_HOURS
                ),

                "n_features": (
                    len(feature_cols)
                ),

                "n_train": (
                    len(X_train)
                ),

                "n_validation": (
                    len(
                        X_validation
                    )
                ),

                "n_test": (
                    len(X_test)
                ),

                "n_development": (
                    len(
                        X_train_validation
                    )
                ),

                "decision_threshold": float(
                    decision_threshold
                ),

                "threshold_selection_metric": (
                    "balanced_accuracy"
                ),

                "threshold_selection_method": (
                    "5-fold OOF "
                    "train+validation"
                ),

                "random_state": (
                    RANDOM_STATE
                ),

                "optuna_trials": (
                    n_trials
                ),
            }
        )

        mlflow.log_metrics(
            flatten_numeric_metrics(
                test_metrics
            )
        )

        mlflow.log_metric(
            "oof_balanced_accuracy_at_threshold",
            float(
                oof_threshold_score
            ),
        )

        # Cloudpickle is used here because the calibrated sklearn
        # object contains internal calibration/CV classes rejected
        # by strict skops trust checking in some MLflow versions.
        mlflow.sklearn.log_model(
            calibrated_pipeline,
            name=(
                "calibrated_prediction_pipeline"
            ),
            serialization_format=(
                mlflow.sklearn
                .SERIALIZATION_FORMAT_CLOUDPICKLE
            ),
        )

        mlflow.log_artifact(
            str(METRICS_PATH)
        )

        mlflow.log_artifact(
            str(
                MODEL_COMPARISON_PATH
            )
        )

        mlflow.log_artifact(
            str(
                MODEL_DETAILS_PATH
            )
        )

    # ---------------------------------------------------------
    # Final console output
    # ---------------------------------------------------------

    print(
        "Saved active model:",
        MODEL_PATH,
    )

    print(
        "Frozen OOF decision threshold:",
        round(
            decision_threshold,
            3,
        ),
    )

    print(
        "OOF balanced accuracy "
        "at threshold:",
        round(
            oof_threshold_score,
            3,
        ),
    )

    print(
        "Untouched-test metrics:",
        test_metrics,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--n-trials",
        type=int,
        default=40,
    )

    args = parser.parse_args()

    train_production(
        n_trials=args.n_trials
    )


if __name__ == "__main__":
    main()