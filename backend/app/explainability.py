import numpy as np
import shap

from backend.app.evaluation_utils import safe_float


def _clean_feature_name(name):
    if "__" in name:
        return name.split("__", 1)[1]
    return name


def get_shap_explanation(model_artifact, patient_df, max_features=5):
    """
    Generate a patient-level explanation from the fitted *base* pipeline.

    The calibrated prediction wrapper is intentionally not used for SHAP.
    Explanations describe the underlying fitted prediction model. If a valid
    SHAP explanation cannot be produced, return an empty list rather than a
    statistically meaningless permutation-importance fallback.
    """
    try:
        base_pipeline = model_artifact.get("pipeline")
        feature_columns = model_artifact.get("feature_columns", [])

        if base_pipeline is None or not feature_columns:
            return []

        if not hasattr(base_pipeline, "named_steps"):
            return []

        preprocessor = base_pipeline.named_steps.get("preprocessor")
        model = base_pipeline.named_steps.get("model")
        if preprocessor is None or model is None:
            return []

        patient_features = patient_df[feature_columns]
        transformed = preprocessor.transform(patient_features)
        transformed = np.asarray(transformed)

        try:
            transformed_names = preprocessor.get_feature_names_out()
            transformed_names = [_clean_feature_name(x) for x in transformed_names]
        except Exception:
            transformed_names = [f"feature_{i}" for i in range(transformed.shape[1])]

        if not hasattr(model, "feature_importances_"):
            return []

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(transformed)

        if isinstance(shap_values, list):
            shap_values = shap_values[-1]

        shap_values = np.asarray(shap_values)

        # Some explainers return (samples, features, classes).
        if shap_values.ndim == 3:
            shap_values = shap_values[:, :, -1]
        elif shap_values.ndim == 1:
            shap_values = shap_values.reshape(1, -1)

        if shap_values.ndim != 2 or shap_values.shape[0] == 0:
            return []

        row_values = transformed[0]
        row_shap = shap_values[0]
        n = min(len(row_values), len(row_shap), len(transformed_names))
        if n == 0:
            return []

        top_indices = np.argsort(np.abs(row_shap[:n]))[-max_features:][::-1]
        explanation = []

        for idx in top_indices:
            value = row_values[idx]
            shap_value = row_shap[idx]
            explanation.append(
                {
                    "feature": transformed_names[idx],
                    "value": safe_float(value),
                    "shap_value": safe_float(shap_value),
                    "impact": "positive" if shap_value > 0 else "negative",
                }
            )

        return explanation

    except Exception:
        return []
