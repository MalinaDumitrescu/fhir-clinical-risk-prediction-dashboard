# Clinical Limitations

## Educational status

This system is an educational prototype for medical informatics.

It is not a medical device and is not intended for real clinical use.

## Dataset limitations

The project uses MIMIC-IV-on-FHIR demo data, which is small and not representative of all hospitals or patient populations.

The small sample size makes performance estimates unstable.

## Target limitations

The current target is long ICU stay.

This is useful for a technical prototype, but it is not a complete clinical deterioration endpoint.

## Leakage controls

The system restricts model features to the first 24 hours after the first encounter start.

Medication, procedure, condition, lab, and vital sign features are included only when timestamps show that they occurred inside the first 24 hours.

Events with missing timestamps are excluded from first-24h counts.

## Calibration limitations

Calibration is performed only when enough samples exist.

Because the dataset is small, calibration curves should be interpreted cautiously.

## Explainability limitations

SHAP explanations describe model behavior, not medical causality.

A feature that increases model risk is not necessarily a true clinical cause.

## Deployment limitations

A real deployment would require:

- clinical validation
- external validation
- privacy review
- security review
- bias evaluation
- monitoring
- clinician usability testing
- regulatory analysis