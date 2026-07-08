import json
from datetime import datetime, timezone

from backend.app.config import RISK_ASSESSMENTS_PATH


def create_risk_assessment(patient_id, probability, risk_level):
    now = datetime.now(timezone.utc).isoformat()

    risk_assessment = {
        "resourceType": "RiskAssessment",
        "id": f"risk-{patient_id}-{int(datetime.now().timestamp())}",
        "status": "final",
        "subject": {
            "reference": f"Patient/{patient_id}"
        },
        "occurrenceDateTime": now,
        "method": {
            "text": "RandomForestClassifier trained on MIMIC-IV-on-FHIR demo features"
        },
        "prediction": [
            {
                "outcome": {
                    "text": "Risk of long ICU stay"
                },
                "probabilityDecimal": round(float(probability), 4),
                "qualitativeRisk": {
                    "text": risk_level
                }
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