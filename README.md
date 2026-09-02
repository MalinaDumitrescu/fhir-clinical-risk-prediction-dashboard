# FHIR Clinical Risk Prediction Dashboard

An end-to-end medical informatics and machine learning prototype that predicts prolonged ICU utilization using structured HL7 FHIR data.

Built on the open-source **MIMIC-IV-on-FHIR** demo dataset, the system processes longitudinal patient encounters, builds leakage-safe clinical features from the first 24 hours of an index hospitalization, and trains calibrated risk models. Predictions, model metrics, and patient-level SHAP explanations are surfaced through a FastAPI backend and an interactive React dashboard, along with interoperable FHIR `RiskAssessment` resources.

 **Disclaimer:** This project is strictly an educational and research prototype. It has not undergone clinical validation, is not cleared as a medical device, and must never be used for real-world triage, bedside decisions, or medical advice.

---

## Overview

Applying machine learning to Electronic Health Records (EHRs) is notoriously tricky: raw clinical data is messy, irregularly sampled, and prone to subtle forms of target leakage.

This project was built to address these practical clinical informatics challenges directly, demonstrating a realistic workflow:

```
FHIR NDJSON Resources -> Encounter Linkage & Cohort Definition -> 24-Hour Feature Extraction (Code & Unit Aware) -> Model Training & Optuna Tuning (Train / Val Split) -> Probability Calibration & Held-Out Test Evaluation -> SHAP Explanations + FHIR RiskAssessment JSON -> FastAPI Service -> React / Vite Clinical UI

```
#### *Demo Video* - [Recording 2026-09-02 061627.zip](https://github.com/user-attachments/files/31719619/Recording.2026-09-02.061627.zip)

### The Clinical Prediction Task

* **Cohort Unit:** A single adult patient anchored to their **index hospital admission** (their earliest eligible inpatient `Encounter`).
* **Observation Window:** Exactly the **first 24 hours** from the admission timestamp. Any clinical event occurring after hour 24 is strictly masked to eliminate forward-looking data leakage.
* **Target (`target_long_icu_stay`):** A binary flag indicating whether the patient accumulates **≥ 3 days (72 hours)** of total ICU utilization during that index stay. If a patient is transferred into an ICU multiple times during the same hospital encounter, these stays are linked via `EncounterICU.partOf` and summed.

---

## Informatics & Feature Engineering

Feature engineering logic lives in `backend/app/features.py`. Rather than taking shortcuts with raw text or unbounded queries, feature extraction respects clinical workflow realities and standard data semantics.

### Robust Code Mapping vs. Loose String Matching

A common pitfall in healthcare ML is relying on broad regex or string matching on test names (e.g., searching for `"lactate"` or `"temp"`). This frequently picks up ventilator settings, alarm thresholds, device diagnostics, or incompatible specimen types.

Here, features are extracted via strict clinical concept maps:

1. **Explicit Codings:** Observations are matched against curated MIMIC item IDs and standard LOINC concepts.
2. **Metadata Separation:** ICU device alarms (e.g., heart-rate high/low alert levels) and ventilator configurations are explicitly filtered out so only true physiological measurements enter the feature set.
3. **Unit Normalization:** Numerical values without valid units are rejected. Measurements with differing clinical units are normalized prior to aggregation (for example, converting temperatures from Fahrenheit to Celsius using `°C = (°F - 32) × 5 / 9`).

### Extracted Feature Sets (First 24 Hours Only)

* **Demographics:** Age (calculated at admission) and administrative sex.
* **Vital Signs:** Heart rate, respiratory rate, SpO₂, systolic/diastolic blood pressure, and core temperature. Aggregated into summary statistics: `mean`, `min`, and `max`.
* **Laboratory Panels:** Routine admission labs including serum creatinine, potassium, hemoglobin, sodium, lactate, white blood cell count (WBC), and glucose (`mean`, `min`, `max`).
* **Medications & Procedures:** Distinct counts for medication orders (`MedicationRequest`) versus confirmed deliveries (`MedicationAdministration`), acknowledging that orders do not always translate into bedside dosing within the first 24 hours. Validated timestamps also tally completed bedside and surgical procedures.
* **Informed Missingness:** In clinical datasets, missing values are rarely missing completely at random (MCAR); the absence of a lab often implies stability. Missing measurements are preserved as informative signals rather than imputed as zeros (which would represent fatal physiology).

---

## Machine Learning Pipeline

### Leakage-Safe Validation Strategy

To avoid over-optimistic performance estimates, the pipeline enforces a clean partition:

* **Training Set:** Used for model fitting and hyperparameter optimization via **Optuna**.
* **Validation Set:** Used exclusively for model selection (comparing across algorithms like XGBoost, LightGBM, and HistGradientBoosting).
* **Held-Out Test Set:** Completely quarantined until the final candidate model is frozen and calibrated.

### Calibration & Interpretability

In a healthcare setting, well-calibrated probabilities matter far more than raw discrimination metrics. An uncalibrated model might yield a high ROC-AUC while outputting risk numbers that cannot be safely translated into operational thresholds.

* **Probability Calibration:** Models undergo isotonic or sigmoid post-hoc calibration via `scikit-learn`.
* **Metrics Tracked:** Evaluated using Brier score, Calibration Curves, and PR-AUC alongside standard ROC-AUC, Balanced Accuracy, and F1.
* **Explainability:** When supported by the model architecture, TreeSHAP decomposes individual risk scores into feature-level contributions for bedside review. Crucially, these values are presented as descriptive model associations rather than causal clinical claims.

---

## Tech Stack

* **Data Layer:** HL7 FHIR (NDJSON format), MIMIC-IV-on-FHIR
* **ML & Informatics:** Python, pandas, NumPy, scikit-learn, XGBoost, LightGBM, Optuna, SHAP, MLflow
* **Backend API:** FastAPI, Uvicorn, Pydantic
* **Frontend:** React, Vite, Tailwind CSS

---

## Key Limitations & Responsible Use

* **Small Demo Sample:** The public MIMIC-IV-on-FHIR demo cohort comprises a limited number of records. Performance curves and point metrics serve as an architectural demonstration rather than generalizable performance baselines.
* **No External Validation:** The models are trained on retrospective single-center data (Beth Israel Deaconess Medical Center). They have not been tested across differing institutional EHR builds, patient demographics, or documentation patterns.
* **Non-Causal Interpretability:** SHAP plots highlight variables that informed a statistical prediction within the bounds of this dataset; they do not suggest clinical interventions or demonstrate that correcting a specific lab value will improve patient outcomes.
* **Interoperability Notice:** The generated FHIR `RiskAssessment` resources adhere conceptually to standard FHIR schemas, but have not undergone production-grade conformance testing against an enterprise FHIR server.

---

## Author

**Malina Dumitrescu**

*Medical Informatics & Applied Machine Learning*
