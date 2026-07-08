import numpy as np

from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def safe_float(value):
    if value is None:
        return None

    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None

    return float(value)


def get_probability_scores(model, X):
    try:
        proba = model.predict_proba(X)

        if proba is not None and proba.shape[1] == 2:
            return proba[:, 1]
    except Exception:
        return None

    return None


def compute_curves(y_true, y_proba, n_bins=10):
    curves = {
        "roc_curve": [],
        "pr_curve": [],
        "calibration_curve": [],
    }

    if y_proba is None:
        return curves

    if len(set(y_true)) < 2:
        return curves

    fpr, tpr, roc_thresholds = roc_curve(y_true, y_proba)

    curves["roc_curve"] = [
        {
            "fpr": safe_float(fpr[i]),
            "tpr": safe_float(tpr[i]),
            "threshold": safe_float(roc_thresholds[i]),
        }
        for i in range(len(fpr))
    ]

    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_proba)

    curves["pr_curve"] = [
        {
            "recall": safe_float(recall[i]),
            "precision": safe_float(precision[i]),
            "threshold": safe_float(pr_thresholds[i]) if i < len(pr_thresholds) else None,
        }
        for i in range(len(precision))
    ]

    frac_pos, mean_pred = calibration_curve(
        y_true,
        y_proba,
        n_bins=n_bins,
        strategy="uniform",
    )

    curves["calibration_curve"] = [
        {
            "mean_predicted_probability": safe_float(mean_pred[i]),
            "fraction_of_positives": safe_float(frac_pos[i]),
        }
        for i in range(len(frac_pos))
    ]

    return curves


def evaluate_binary_classifier(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = get_probability_scores(model, X_test)

    metrics = {
        "accuracy": safe_float(accuracy_score(y_test, y_pred)),
        "balanced_accuracy": safe_float(balanced_accuracy_score(y_test, y_pred)),
        "precision": safe_float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": safe_float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": safe_float(f1_score(y_test, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(
            y_test,
            y_pred,
            output_dict=True,
            zero_division=0,
        ),
        "roc_auc": None,
        "average_precision": None,
        "brier_score": None,
        "roc_curve": [],
        "pr_curve": [],
        "calibration_curve": [],
    }

    if y_proba is not None and len(set(y_test)) == 2:
        metrics["roc_auc"] = safe_float(roc_auc_score(y_test, y_proba))
        metrics["average_precision"] = safe_float(average_precision_score(y_test, y_proba))
        metrics["brier_score"] = safe_float(brier_score_loss(y_test, y_proba))

        curves = compute_curves(y_test, y_proba)
        metrics.update(curves)

    return metrics