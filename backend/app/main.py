import json

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from backend.app.config import (
    FEATURES_CSV,
    MODEL_PATH,
    METRICS_PATH,
    MODEL_COMPARISON_PATH,
    MODEL_DETAILS_PATH,
    RISK_ASSESSMENTS_PATH,
)
from backend.app.explainability import get_shap_explanation
from backend.app.features import build_feature_dataframe
from backend.app.risk_assessment import (
    create_risk_assessment,
    save_risk_assessment,
)


app = FastAPI(
    title="FHIR Clinical Risk Dashboard API",
    description="Educational medical informatics prototype using FHIR data and ML.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def load_features():
    if FEATURES_CSV.exists():
        return pd.read_csv(FEATURES_CSV)

    df = build_feature_dataframe()
    df.to_csv(FEATURES_CSV, index=False)
    return df


def load_model_artifact():
    if not MODEL_PATH.exists():
        raise HTTPException(
            status_code=500,
            detail="Model not found. Run: python -m backend.app.train_production_models",
        )

    return joblib.load(MODEL_PATH)


def get_risk_level(probability):
    if probability < 0.33:
        return "low"

    if probability < 0.66:
        return "medium"

    return "high"


@app.get("/")
def root():
    return {
        "message": "FHIR Clinical Risk Dashboard API is running.",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {
        "status": "ok"
    }


@app.get("/metrics")
def get_metrics():
    if not METRICS_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="Metrics file not found. Train the production model first.",
        )

    with open(METRICS_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


@app.get("/models/details")
def get_model_details():
    if MODEL_DETAILS_PATH.exists():
        with open(MODEL_DETAILS_PATH, "r", encoding="utf-8") as file:
            return json.load(file)

    model_artifact = load_model_artifact()

    return model_artifact.get("model_details", {})


@app.get("/models/comparison")
def get_model_comparison():
    if not MODEL_COMPARISON_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="Model comparison file not found. Run production training first.",
        )

    with open(MODEL_COMPARISON_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


@app.get("/models/curves")
def get_model_curves():
    metrics = get_metrics()

    return {
        "roc_curve": metrics.get("roc_curve", []),
        "pr_curve": metrics.get("pr_curve", []),
        "calibration_curve": metrics.get("calibration_curve", []),
        "brier_score": metrics.get("brier_score"),
    }


@app.get("/patients")
def get_patients():
    df = load_features()

    columns = [
        "patient_id",
        "gender",
        "age",
        "condition_count",
        "medication_event_count",
        "procedure_count",
        "encounter_count",
        "icu_los_days",
        "target_long_icu_stay",
    ]

    available_columns = [col for col in columns if col in df.columns]

    return df[available_columns].fillna("").to_dict(orient="records")


@app.get("/patients/{patient_id}")
def get_patient(patient_id: str):
    df = load_features()

    patient = df[df["patient_id"] == patient_id]

    if patient.empty:
        raise HTTPException(status_code=404, detail="Patient not found")

    return patient.fillna("").iloc[0].to_dict()


@app.get("/patients/{patient_id}/predict")
def predict_patient(patient_id: str):
    df = load_features()

    patient = df[df["patient_id"] == patient_id]

    if patient.empty:
        raise HTTPException(status_code=404, detail="Patient not found")

    model_artifact = load_model_artifact()

    prediction_pipeline = model_artifact.get(
        "prediction_pipeline",
        model_artifact["pipeline"],
    )

    feature_columns = model_artifact["feature_columns"]

    patient_features = patient[feature_columns]

    probability = prediction_pipeline.predict_proba(patient_features)[0][1]
    risk_level = get_risk_level(probability)

    explanation = get_shap_explanation(
        model_artifact=model_artifact,
        patient_df=patient,
        max_features=5,
    )

    return {
        "patient_id": patient_id,
        "model_name": model_artifact.get("model_name", "unknown"),
        "calibrated": bool(model_artifact.get("calibrated", False)),
        "risk_probability": round(float(probability), 4),
        "risk_percent": round(float(probability) * 100, 2),
        "risk_level": risk_level,
        "explanation": explanation,
        "warning": "Educational prototype only. Not for clinical use.",
    }


@app.post("/patients/{patient_id}/risk-assessment")
def export_risk_assessment(patient_id: str):
    prediction = predict_patient(patient_id)

    basis_refs = [
        {
            "reference": f"Patient/{patient_id}"
        }
    ]

    risk_assessment = create_risk_assessment(
        patient_id=patient_id,
        probability=prediction["risk_probability"],
        risk_level=prediction["risk_level"],
        model_name=prediction["model_name"],
        explanation=prediction["explanation"],
        basis_refs=basis_refs,
    )

    save_risk_assessment(risk_assessment)

    return risk_assessment


@app.get("/risk-assessments")
def get_risk_assessments():
    if not RISK_ASSESSMENTS_PATH.exists():
        return []

    assessments = []
    with open(RISK_ASSESSMENTS_PATH, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                assessments.append(json.loads(line))
    return assessments


@app.get("/risk-assessments/download")
def download_risk_assessments():
    if not RISK_ASSESSMENTS_PATH.exists():
        raise HTTPException(status_code=404, detail="File not found.")
    return FileResponse(
        path=RISK_ASSESSMENTS_PATH,
        filename="RiskAssessment.ndjson",
        media_type="application/x-ndjson",
    )
