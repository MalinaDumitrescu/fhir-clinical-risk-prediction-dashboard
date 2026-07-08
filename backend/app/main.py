import json

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.app.config import FEATURES_CSV, MODEL_PATH, METRICS_PATH
from backend.app.features import build_feature_dataframe
from backend.app.risk_assessment import (
    create_risk_assessment,
    save_risk_assessment,
)


app = FastAPI(
    title="FHIR Clinical Risk Dashboard API",
    description="Educational medical informatics prototype using FHIR data and ML.",
    version="1.0.0",
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
            detail="Model not found. Run: python -m backend.app.train_model",
        )

    return joblib.load(MODEL_PATH)


def get_risk_level(probability):
    if probability < 0.33:
        return "low"

    if probability < 0.66:
        return "medium"

    return "high"


def explain_prediction(model_artifact, patient_df):
    """
    Simple explanation for V1.

    This is not full SHAP yet.
    It uses the random forest global feature importance and combines it with
    the patient's available values.
    """
    pipeline = model_artifact["pipeline"]
    feature_columns = model_artifact["feature_columns"]

    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    transformed = preprocessor.transform(patient_df[feature_columns])

    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        feature_names = [f"feature_{i}" for i in range(transformed.shape[1])]

    values = transformed[0]

    if hasattr(values, "toarray"):
        values = values.toarray()[0]

    importances = model.feature_importances_

    explanations = []

    for name, value, importance in zip(feature_names, values, importances):
        score = abs(float(value)) * float(importance)

        if score > 0:
            clean_name = name.replace("numeric__", "").replace("categorical__", "")

            explanations.append(
                {
                    "feature": clean_name,
                    "value": float(value),
                    "importance": float(importance),
                    "score": score,
                }
            )

    explanations = sorted(
        explanations,
        key=lambda item: item["score"],
        reverse=True,
    )

    return explanations[:5]


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
            detail="Metrics file not found. Train the model first.",
        )

    with open(METRICS_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


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

    pipeline = model_artifact["pipeline"]
    feature_columns = model_artifact["feature_columns"]

    patient_features = patient[feature_columns]

    probability = pipeline.predict_proba(patient_features)[0][1]
    risk_level = get_risk_level(probability)

    explanation = explain_prediction(model_artifact, patient)

    return {
        "patient_id": patient_id,
        "risk_probability": round(float(probability), 4),
        "risk_percent": round(float(probability) * 100, 2),
        "risk_level": risk_level,
        "explanation": explanation,
        "warning": "Educational prototype only. Not for clinical use.",
    }


@app.post("/patients/{patient_id}/risk-assessment")
def export_risk_assessment(patient_id: str):
    prediction = predict_patient(patient_id)

    risk_assessment = create_risk_assessment(
        patient_id=patient_id,
        probability=prediction["risk_probability"],
        risk_level=prediction["risk_level"],
    )

    save_risk_assessment(risk_assessment)

    return risk_assessment