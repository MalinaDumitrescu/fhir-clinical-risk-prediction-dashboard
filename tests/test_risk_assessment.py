from backend.app.risk_assessment import create_risk_assessment


def test_create_risk_assessment():
    resource = create_risk_assessment(
        patient_id="patient-1",
        probability=0.72,
        risk_level="high",
        model_name="extra_trees",
        explanation=[
            {
                "feature": "age",
                "impact": "increases risk",
                "shap_value": 0.12,
            }
        ],
    )

    assert resource["resourceType"] == "RiskAssessment"
    assert resource["status"] == "final"
    assert resource["subject"]["reference"] == "Patient/patient-1"
    assert resource["prediction"][0]["probabilityDecimal"] == 0.72
    assert resource["prediction"][0]["qualitativeRisk"]["text"] == "high"
    assert "extra_trees" in resource["method"]["text"]