import json
from datetime import datetime, timezone

from backend.app.config import RISK_ASSESSMENTS_PATH


def build_rationale(explanation):
    if not explanation:
        return "No explanation available."

    parts = []

    for item in explanation[:5]:
        feature = item.get("feature", "unknown feature")
        impact = item.get("impact", "unknown impact")
        shap_value = item.get("shap_value", 0.0)

        parts.append(f"{feature} {impact} SHAP={shap_value:.4f}")

    return "Top contributors: " + "; ".join(parts)


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
                "outcome": {
                    "text": "Risk of long ICU stay"
                },
                "probabilityDecimal": round(float(probability), 4),
                "qualitativeRisk": {
                    "text": risk_level
                },
                "rationale": build_rationale(explanation)
            }
        ],
        "note": [
            {
                "text": "Educational prototype only. Not for clinical use."
            }
        ]
    }

    return risk_assessment


def save_risk_assessment(risk_assessment):
    with open(RISK_ASSESSMENTS_PATH, "a", encoding="utf-8") as file:
        file.write(json.dumps(risk_assessment) + "\n")