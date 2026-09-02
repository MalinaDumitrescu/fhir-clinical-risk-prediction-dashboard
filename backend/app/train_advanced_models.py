import argparse
import json
import warnings

import joblib
import numpy as np
import optuna
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from backend.app.config import (
    FEATURES_CSV,
    MODEL_PATH,
    METRICS_PATH,
    MODEL_COMPARISON_PATH,
    OPTUNA_TRIALS_CSV,
    MODELS_DIR,
)
from backend.app.features import build_feature_dataframe
from backend.app.evaluation_utils import evaluate_binary_classifier


warnings.filterwarnings("ignore")

TARGET_COL = "target_long_icu_stay"
RANDOM_STATE = 42


try:
    from xgboost import XGBClassifier

    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False


try:
    from lightgbm import LGBMClassifier

    HAS_LIGHTGBM = True
except Exception:
    HAS_LIGHTGBM = False


def load_or_create_features():
    # Always rebuild from source FHIR so stale artifacts cannot silently
    # survive feature-definition changes.
    df = build_feature_dataframe()
    df.to_csv(FEATURES_CSV, index=False)
    return df


def prepare_xy(df):
    if TARGET_COL not in df.columns:
        raise ValueError(f"Missing target column: {TARGET_COL}")

    if df[TARGET_COL].nunique() < 2:
        raise ValueError(
            "The target has only one class. "
            "The demo dataset may be too small or the target threshold may need adjustment."
        )

    drop_cols = [
        "patient_id",
        "birth_date",
        "deceased",
        "index_hospital_encounter_id",
        "icu_los_days",
        "hospital_los_days",
        TARGET_COL,
    ]

    feature_cols = [col for col in df.columns if col not in drop_cols]

    X = df[feature_cols]
    y = df[TARGET_COL]

    numeric_cols = X.select_dtypes(include="number").columns.tolist()
    categorical_cols = X.select_dtypes(exclude="number").columns.tolist()

    return X, y, feature_cols, numeric_cols, categorical_cols


def make_onehot_encoder():
    """
    Handles different scikit-learn versions.
    Newer versions use sparse_output.
    Older versions use sparse.
    """
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(numeric_cols, categorical_cols):
    transformers = []

    if numeric_cols:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
            ]
        )

        transformers.append(("numeric", numeric_pipeline, numeric_cols))

    if categorical_cols:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", make_onehot_encoder()),
            ]
        )

        transformers.append(("categorical", categorical_pipeline, categorical_cols))

    if not transformers:
        raise ValueError("No usable feature columns found.")

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.0,
    )

    return preprocessor


def get_available_model_names():
    names = [
        "random_forest",
        "extra_trees",
        "gradient_boosting",
        "hist_gradient_boosting",
    ]

    if HAS_XGBOOST:
        names.append("xgboost")

    if HAS_LIGHTGBM:
        names.append("lightgbm")

    return names


def get_scale_pos_weight(y):
    positives = int((y == 1).sum())
    negatives = int((y == 0).sum())

    if positives == 0:
        return 1.0

    return negatives / positives


def build_model_from_trial(trial, y):
    model_name = trial.suggest_categorical("model_name", get_available_model_names())

    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=trial.suggest_int("rf_n_estimators", 100, 800, step=100),
            max_depth=trial.suggest_int("rf_max_depth", 2, 14),
            min_samples_split=trial.suggest_int("rf_min_samples_split", 2, 12),
            min_samples_leaf=trial.suggest_int("rf_min_samples_leaf", 1, 8),
            max_features=trial.suggest_categorical(
                "rf_max_features",
                ["sqrt", "log2", None],
            ),
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    if model_name == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=trial.suggest_int("et_n_estimators", 100, 800, step=100),
            max_depth=trial.suggest_int("et_max_depth", 2, 14),
            min_samples_split=trial.suggest_int("et_min_samples_split", 2, 12),
            min_samples_leaf=trial.suggest_int("et_min_samples_leaf", 1, 8),
            max_features=trial.suggest_categorical(
                "et_max_features",
                ["sqrt", "log2", None],
            ),
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    if model_name == "gradient_boosting":
        return GradientBoostingClassifier(
            n_estimators=trial.suggest_int("gb_n_estimators", 50, 500, step=50),
            learning_rate=trial.suggest_float("gb_learning_rate", 0.01, 0.25, log=True),
            max_depth=trial.suggest_int("gb_max_depth", 1, 5),
            min_samples_leaf=trial.suggest_int("gb_min_samples_leaf", 1, 10),
            subsample=trial.suggest_float("gb_subsample", 0.6, 1.0),
            random_state=RANDOM_STATE,
        )

    if model_name == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(
            max_iter=trial.suggest_int("hgb_max_iter", 50, 500, step=50),
            learning_rate=trial.suggest_float("hgb_learning_rate", 0.01, 0.25, log=True),
            max_leaf_nodes=trial.suggest_int("hgb_max_leaf_nodes", 8, 64),
            min_samples_leaf=trial.suggest_int("hgb_min_samples_leaf", 5, 30),
            l2_regularization=trial.suggest_float(
                "hgb_l2_regularization",
                0.0001,
                10.0,
                log=True,
            ),
            random_state=RANDOM_STATE,
        )

    if model_name == "xgboost":
        return XGBClassifier(
            n_estimators=trial.suggest_int("xgb_n_estimators", 50, 500, step=50),
            max_depth=trial.suggest_int("xgb_max_depth", 2, 8),
            learning_rate=trial.suggest_float("xgb_learning_rate", 0.01, 0.25, log=True),
            subsample=trial.suggest_float("xgb_subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("xgb_colsample_bytree", 0.6, 1.0),
            min_child_weight=trial.suggest_float("xgb_min_child_weight", 1.0, 10.0),
            reg_lambda=trial.suggest_float("xgb_reg_lambda", 0.001, 10.0, log=True),
            reg_alpha=trial.suggest_float("xgb_reg_alpha", 0.001, 10.0, log=True),
            scale_pos_weight=get_scale_pos_weight(y),
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    if model_name == "lightgbm":
        return LGBMClassifier(
            n_estimators=trial.suggest_int("lgbm_n_estimators", 50, 500, step=50),
            learning_rate=trial.suggest_float("lgbm_learning_rate", 0.01, 0.25, log=True),
            num_leaves=trial.suggest_int("lgbm_num_leaves", 8, 64),
            max_depth=trial.suggest_int("lgbm_max_depth", 2, 10),
            min_child_samples=trial.suggest_int("lgbm_min_child_samples", 5, 30),
            subsample=trial.suggest_float("lgbm_subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("lgbm_colsample_bytree", 0.6, 1.0),
            reg_lambda=trial.suggest_float("lgbm_reg_lambda", 0.001, 10.0, log=True),
            reg_alpha=trial.suggest_float("lgbm_reg_alpha", 0.001, 10.0, log=True),
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        )

    raise ValueError(f"Unknown model name: {model_name}")


def make_pipeline(model, numeric_cols, categorical_cols):
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(numeric_cols, categorical_cols)),
            ("model", model),
        ]
    )


def get_cv(y):
    min_class_count = int(y.value_counts().min())

    if min_class_count < 2:
        raise ValueError("Not enough samples per class for cross-validation.")

    n_splits = min(5, min_class_count)

    return StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RANDOM_STATE,
    )


def objective(trial, X, y, numeric_cols, categorical_cols):
    model = build_model_from_trial(trial, y)
    pipeline = make_pipeline(model, numeric_cols, categorical_cols)

    cv = get_cv(y)

    scores = cross_val_score(
        pipeline,
        X,
        y,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        error_score="raise",
    )

    return float(np.mean(scores))


def evaluate_pipeline(pipeline, X_train, X_test, y_train, y_test):
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    y_proba = None
    if hasattr(pipeline, "predict_proba"):
        try:
            proba = pipeline.predict_proba(X_test)

            if proba is not None and proba.shape[1] == 2:
                y_proba = proba[:, 1]
        except Exception:
            y_proba = None

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(
            y_test,
            y_pred,
            output_dict=True,
            zero_division=0,
        ),
        "roc_auc": None,
        "average_precision": None,
    }

    if y_proba is not None and y_test.nunique() == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_test, y_proba))
        metrics["average_precision"] = float(average_precision_score(y_test, y_proba))

    return metrics


def score_for_selection(metrics):
    if metrics.get("roc_auc") is not None:
        return metrics["roc_auc"]

    if metrics.get("average_precision") is not None:
        return metrics["average_precision"]

    return metrics["balanced_accuracy"]


def get_best_trial_per_model(study):
    best_by_model = {}

    for trial in study.trials:
        if trial.value is None:
            continue

        model_name = trial.params.get("model_name")

        if not model_name:
            continue

        if model_name not in best_by_model:
            best_by_model[model_name] = trial
            continue

        if trial.value > best_by_model[model_name].value:
            best_by_model[model_name] = trial

    return best_by_model


def build_model_from_params(params, y):
    fixed_trial = optuna.trial.FixedTrial(params)
    return build_model_from_trial(fixed_trial, y)


def build_ensembles(best_trials_by_model, y):
    """
    Creates ensemble models from the best tuned base models.
    The base models are bare classifiers.
    The preprocessing is added later in make_pipeline().
    """
    estimators = []

    for model_name, trial in best_trials_by_model.items():
        model = build_model_from_params(trial.params, y)
        estimators.append((model_name, model))

    ensembles = {}

    if len(estimators) >= 2:
        ensembles["soft_voting_ensemble"] = VotingClassifier(
            estimators=estimators,
            voting="soft",
            n_jobs=-1,
        )

    if len(estimators) >= 2:
        ensembles["stacking_ensemble"] = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
            stack_method="predict_proba",
            n_jobs=-1,
        )

    return ensembles


def split_train_validation_test(X, y):
    """Create a stratified 60/20/20 train/validation/test split."""
    stratify = y if y.value_counts().min() >= 2 else None

    X_dev, X_test, y_dev, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=stratify,
    )

    stratify_dev = y_dev if y_dev.value_counts().min() >= 2 else None
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_dev,
        y_dev,
        test_size=0.25,  # 25% of 80% = 20% of full data
        random_state=RANDOM_STATE,
        stratify=stratify_dev,
    )

    return X_train, X_validation, X_test, y_train, y_validation, y_test


def build_selected_model(best_model_name, best_artifact, best_trials_by_model, y):
    """Rebuild the selected base or ensemble model for final fitting."""
    if best_artifact.get("optuna_params"):
        return build_model_from_params(best_artifact["optuna_params"], y)

    ensembles = build_ensembles(best_trials_by_model, y)
    if best_model_name not in ensembles:
        raise ValueError(f"Could not rebuild selected model: {best_model_name}")

    return ensembles[best_model_name]


def train_advanced(n_trials):
    df = load_or_create_features()
    X, y, feature_cols, numeric_cols, categorical_cols = prepare_xy(df)

    print("Dataset shape:", df.shape)
    print("Features:", len(feature_cols))
    print("Target distribution:")
    print(y.value_counts())
    print("Available model families:", get_available_model_names())

    (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
    ) = split_train_validation_test(X, y)

    study = optuna.create_study(
        direction="maximize",
        study_name="clinical_risk_model_tuning",
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

    study.trials_dataframe().to_csv(OPTUNA_TRIALS_CSV, index=False)
    best_trials_by_model = get_best_trial_per_model(study)

    comparison = []
    trained_artifacts = {}

    for model_name, trial in best_trials_by_model.items():
        model = build_model_from_params(trial.params, y_train)
        pipeline = make_pipeline(model, numeric_cols, categorical_cols)
        pipeline.fit(X_train, y_train)
        validation_metrics = evaluate_binary_classifier(
            pipeline, X_validation, y_validation
        )

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

        comparison.append({
            "model_name": model_name,
            "model_file": str(model_file),
            "evaluation_split": "validation",
            "optuna_cv_roc_auc": float(trial.value),
            **validation_metrics,
        })

    for ensemble_name, ensemble_model in build_ensembles(
        best_trials_by_model, y_train
    ).items():
        pipeline = make_pipeline(ensemble_model, numeric_cols, categorical_cols)
        pipeline.fit(X_train, y_train)
        validation_metrics = evaluate_binary_classifier(
            pipeline, X_validation, y_validation
        )

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

        comparison.append({
            "model_name": ensemble_name,
            "model_file": str(model_file),
            "evaluation_split": "validation",
            "optuna_cv_roc_auc": None,
            **validation_metrics,
        })

    comparison = sorted(comparison, key=score_for_selection, reverse=True)
    best_model_name = comparison[0]["model_name"]
    best_artifact = trained_artifacts[best_model_name]

    X_train_validation = pd.concat([X_train, X_validation], axis=0)
    y_train_validation = pd.concat([y_train, y_validation], axis=0)

    final_model = build_selected_model(
        best_model_name,
        best_artifact,
        best_trials_by_model,
        y_train_validation,
    )
    final_pipeline = make_pipeline(final_model, numeric_cols, categorical_cols)
    final_pipeline.fit(X_train_validation, y_train_validation)
    test_metrics = evaluate_binary_classifier(final_pipeline, X_test, y_test)

    final_artifact = {
        **best_artifact,
        "pipeline": final_pipeline,
        "prediction_pipeline": final_pipeline,
        "model_name": best_model_name,
        "calibrated": False,
        "active_metrics": test_metrics,
    }
    joblib.dump(final_artifact, MODEL_PATH)

    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump({
            "model_name": best_model_name,
            "evaluation_split": "test",
            **test_metrics,
        }, file, indent=4)

    with open(MODEL_COMPARISON_PATH, "w", encoding="utf-8") as file:
        json.dump({
            "best_model_name": best_model_name,
            "selection_split": "validation",
            "selection_metric": "ROC-AUC if available, else average precision, else balanced accuracy",
            "models": comparison,
        }, file, indent=4)

    print("Selected on validation:", best_model_name)
    print("Final untouched-test metrics:")
    print(test_metrics)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--n-trials",
        type=int,
        default=40,
        help="Number of Optuna trials.",
    )

    args = parser.parse_args()

    train_advanced(n_trials=args.n_trials)


if __name__ == "__main__":
    main()