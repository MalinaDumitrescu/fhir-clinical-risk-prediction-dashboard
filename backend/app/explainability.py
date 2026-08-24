import numpy as np
import shap

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


def get_shap_explanation(model_artifact, patient_df, max_features=5):
    """
    Generate SHAP explanation for a single patient prediction.
    Falls back to permutation importance if SHAP/feature importance fails.
    """
    try:
        pipeline = model_artifact.get("prediction_pipeline", model_artifact.get("pipeline"))
        feature_columns = model_artifact.get("feature_columns", [])
        
        if not feature_columns:
            return "Feature columns not found in model artifact."
        
        # Extract the actual model from wrappers
        model = pipeline
        
        # Unwrap CalibratedClassifierCV first
        if hasattr(model, 'base_estimator'):
            model = model.base_estimator
        elif hasattr(model, 'estimator'):
            model = model.estimator
        
        # Extract model from Pipeline
        if hasattr(model, 'named_steps'):
            model = model.named_steps.get("model", model)
        
        # If model is still a CalibratedClassifierCV, unwrap again
        if hasattr(model, 'base_estimator'):
            model = model.base_estimator
        elif hasattr(model, 'estimator'):
            model = model.estimator
        
        # Prepare features as numpy arrays
        patient_features = patient_df[feature_columns].fillna(0)
        
        # Get preprocessor from pipeline if available
        if hasattr(pipeline, 'named_steps') and 'preprocessor' in pipeline.named_steps:
            preprocessor = pipeline.named_steps['preprocessor']
            # Transform the features using the preprocessor
            patient_features_transformed = preprocessor.transform(patient_features)
        else:
            patient_features_transformed = patient_features.values
        
        # Try TreeExplainer for tree-based models
        if hasattr(model, 'feature_importances_'):
            try:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(patient_features_transformed)
                
                # Handle binary classification (shap_values might be a list)
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]  # Get values for positive class
                
                # Ensure shap_values is 2D
                if len(shap_values.shape) == 1:
                    shap_values = shap_values.reshape(1, -1)
                
                # Extract top features
                shap_values_flat = np.abs(shap_values[0])
                n_features = min(len(feature_columns), len(shap_values_flat))
                top_indices = np.argsort(shap_values_flat[:n_features])[-max_features:][::-1]
                
                explanation = []
                for idx in top_indices:
                    if idx < len(feature_columns):
                        feature_name = feature_columns[idx]
                        feature_value = patient_features.iloc[0, idx]
                        shap_value = shap_values[0, idx]
                        impact = "positive" if shap_value > 0 else "negative"
                        
                        explanation.append({
                            "feature": feature_name,
                            "value": safe_float(feature_value),
                            "shap_value": safe_float(shap_value),
                            "impact": impact,
                        })
                
                return explanation if explanation else "No SHAP values computed."
            except Exception as tree_err:
                # TreeExplainer failed, try permutation importance
                pass
        
        # Fallback: compute permutation importance
        try:
            from sklearn.inspection import permutation_importance
            
            # Get a dummy y for permutation importance (use all zeros/ones)
            y_dummy = np.zeros(len(patient_features))
            
            result = permutation_importance(
                pipeline,
                patient_features,
                y_dummy,
                n_repeats=10,
                random_state=42,
                n_jobs=-1,
            )
            
            importances = result.importances_mean
            top_indices = np.argsort(importances)[-max_features:][::-1]
            
            explanation = []
            for idx in top_indices:
                if idx < len(feature_columns):
                    feature_name = feature_columns[idx]
                    feature_value = patient_features.iloc[0, idx]
                    importance = importances[idx]
                    
                    explanation.append({
                        "feature": feature_name,
                        "value": safe_float(feature_value),
                        "shap_value": safe_float(importance),
                        "impact": "positive",
                    })
            
            return explanation if explanation else "No importance values computed."
        except Exception as perm_err:
            pass
        
        # Last fallback: use coef_ for linear models
        if hasattr(model, 'coef_'):
            coef = model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_
            top_indices = np.argsort(np.abs(coef))[-max_features:][::-1]
            
            explanation = []
            for idx in top_indices:
                if idx < len(feature_columns):
                    feature_name = feature_columns[idx]
                    feature_value = patient_features.iloc[0, idx]
                    coef_value = coef[idx]
                    impact = "positive" if coef_value > 0 else "negative"
                    
                    explanation.append({
                        "feature": feature_name,
                        "value": safe_float(feature_value),
                        "shap_value": safe_float(coef_value),
                        "impact": impact,
                    })
            
            return explanation if explanation else "No coefficients computed."
        
        return "Model type not supported for explanations."
    
    except Exception as e:
        return f"SHAP explanation failed: {str(e)}"