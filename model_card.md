# Model Card: FHIR Clinical Risk Prediction Dashboard

## Model name

FHIR-based Clinical Risk Prediction Model for Long ICU Stay

## Intended use

This project is an educational medical informatics prototype. It predicts the risk of long ICU stay using structured FHIR resources from the MIMIC-IV-on-FHIR demo dataset.

This system is not intended for real clinical decision-making.

## Prediction target

The model predicts:

`target_long_icu_stay`

Definition:

Long ICU stay is defined as ICU length of stay greater than or equal to 3 days.

For very small demo datasets, a median split fallback may be used only to keep the machine-learning pipeline runnable.

## Prediction time

The prediction is made after the first 24 hours after the first encounter start.

## Leakage policy

Only features with timestamps inside the first 24 hours after first encounter start are used.

Events without timestamps are excluded from first-24h event-count features.

## Data source

MIMIC-IV-on-FHIR demo data.

FHIR resources used include:

- Patient
- Encounter
- EncounterICU
- Condition
- Procedure
- ProcedureICU
- MedicationRequest
- MedicationAdministration
- MedicationAdministrationICU
- MedicationDispense
- ObservationChartevents
- ObservationLabevents
- ObservationOutputevents
- ObservationDatetimeevents

## Features

Example feature groups:

- Age
- Gender
- First-24h condition count
- First-24h medication event count
- First-24h procedure count
- First-24h vital sign summaries
- First-24h lab summaries

## Models compared

The training pipeline compares:

- Random Forest
- Extra Trees
- Gradient Boosting
- HistGradientBoosting
- XGBoost
- LightGBM
- Soft Voting Ensemble
- Stacking Ensemble

Optuna is used for hyperparameter tuning.

## Evaluation metrics

The project reports:

- Accuracy
- Balanced accuracy
- Precision
- Recall
- F1
- ROC-AUC
- Average precision
- Brier score
- Confusion matrix
- ROC curve
- Precision-recall curve
- Calibration curve

## Explainability

The dashboard uses SHAP explanations for tree-based models.

For models that cannot be explained directly with SHAP, the system falls back to feature-importance based explanation.

## Calibration

The active prediction model uses sigmoid calibration when enough data is available.

Brier score and calibration curve are reported.

## Limitations

- The demo dataset is small.
- Results are not clinically validated.
- Performance may be unstable because of small sample size.
- The target is simplified.
- The model should not be used in real patient care.
- External validation is missing.
- Bias and subgroup performance are not fully evaluated.

## Ethical warning

This is an educational prototype only. It must not be used for diagnosis, treatment, triage, or real clinical decision support.