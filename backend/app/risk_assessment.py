import json
from datetime import datetime, timezone

from backend.app.config import RISK_ASSESSMENTS_PATH


def create_risk_note(explanation):
    if not explanation:
        return "No detailed explanation available."

    if isinstance(explanation, str):
        return explanation

    if not isinstance(explanation, list):
        return "Explanation format is not supported."

    feature_lines = []
    for item in explanation:
        feature = item.get("feature", "unknown feature")
        impact = item.get("impact", "unknown impact")
        value = item.get("value", "N/A")
        shap_value = item.get("shap_value", "N/A")

        line = f"- {feature} (value: {value}, SHAP: {shap_value}): {impact}"
        feature_lines.append(line)

    return "Risk factors considered:\\n" + "\\n".join(feature_lines)


def create_risk_assessment(
    patient_id,
    probability,
    risk_level,
    model_name,
    explanation=None,
    basis_refs=None,
):
    now = datetime.now(timezone.utc).isoformat()

    if basis_refs is None:
        basis_refs = [
            {
                "reference": f"Patient/{patient_id}"
            }
        ]

    risk_assessment = {
        "resourceType": "RiskAssessment",
        "id": f"risk-{patient_id}-{int(datetime.now().timestamp())}",
        "status": "final",
        "subject": {
            "reference": f"Patient/{patient_id}"
        },
        "occurrenceDateTime": now,
        "method": {
            "text": f"{model_name} selected after Optuna tuning; calibrated probability pipeline used for risk score"
        },
        "basis": basis_refs,
        "prediction": [
            {
                "probabilityDecimal": probability,
                "qualitativeRisk": {
                    "text": risk_level
                },
                "rationale": create_risk_note(explanation),
            }
        ],
        "note": [
            {
                "text": "Educational prototype only. Not for clinical use."
            }
        ],
    }

    return risk_assessment


def save_risk_assessment(risk_assessment):
    RISK_ASSESSMENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RISK_ASSESSMENTS_PATH, "a", encoding="utf-8") as file:
        file.write(json.dumps(risk_assessment) + "\\n")
