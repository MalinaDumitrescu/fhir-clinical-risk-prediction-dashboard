from collections import defaultdict
from datetime import timedelta

import pandas as pd

from backend.app.config import DATA_DIR, FEATURES_CSV
from backend.app.fhir_utils import (
    load_ndjson,
    get_reference_id,
    get_subject_patient_id,
    get_encounter_id,
    get_period_start,
    get_period_end,
    get_observation_time,
    get_numeric_value,
    get_observation_display,
    get_event_time,
    parse_datetime,
    days_between,
)


PREDICTION_WINDOW_HOURS = 24


OBSERVATION_KEYWORDS = {
    "heart_rate": ["heart rate"],
    "respiratory_rate": ["respiratory rate"],
    "spo2": ["oxygen saturation", "o2 saturation", "spo2"],
    "temperature": ["temperature"],
    "systolic_bp": ["systolic"],
    "diastolic_bp": ["diastolic"],
    "glucose": ["glucose"],
    "creatinine": ["creatinine"],
    "hemoglobin": ["hemoglobin"],
    "white_blood_cells": ["white blood", "wbc"],
    "sodium": ["sodium"],
    "potassium": ["potassium"],
    "lactate": ["lactate"],
}


def match_observation_feature(display_name):
    display_name = display_name.lower()

    for feature_name, keywords in OBSERVATION_KEYWORDS.items():
        for keyword in keywords:
            if keyword in display_name:
                return feature_name

    return None


def is_within_prediction_window(patient_id, event_time, patient_first_start):
    if event_time is None:
        return False

    first_start = patient_first_start.get(patient_id)

    if first_start is None:
        return False

    first_end = first_start + timedelta(hours=PREDICTION_WINDOW_HOURS)

    return first_start <= event_time <= first_end


def build_patient_base():
    patients = load_ndjson(DATA_DIR / "Patient.ndjson")

    rows = {}

    for patient in patients:
        patient_id = patient.get("id")

        if not patient_id:
            continue

        rows[patient_id] = {
            "patient_id": patient_id,
            "gender": patient.get("gender", "unknown"),
            "birth_date": patient.get("birthDate"),
            "deceased": 1 if patient.get("deceasedDateTime") else 0,

            # First-24h features only
            "condition_count": 0,
            "medication_event_count": 0,
            "procedure_count": 0,
            "encounter_count": 0,

            # Outcomes / descriptive variables
            "icu_los_days": 0.0,
            "hospital_los_days": 0.0,
        }

    return rows


def build_encounter_maps(patient_rows):
    encounter_to_patient = {}
    patient_first_start = {}

    encounter_files = [
        DATA_DIR / "Encounter.ndjson",
        DATA_DIR / "EncounterICU.ndjson",
    ]

    for file_path in encounter_files:
        encounters = load_ndjson(file_path)

        for encounter in encounters:
            encounter_id = encounter.get("id")
            subject = encounter.get("subject", {})
            patient_id = get_reference_id(subject.get("reference"))

            if not encounter_id or not patient_id:
                continue

            encounter_to_patient[encounter_id] = patient_id

            if patient_id not in patient_rows:
                continue

            patient_rows[patient_id]["encounter_count"] += 1

            start = get_period_start(encounter)
            end = get_period_end(encounter)

            if start:
                old_start = patient_first_start.get(patient_id)

                if old_start is None or start < old_start:
                    patient_first_start[patient_id] = start

            los_days = days_between(start, end)

            if file_path.name == "EncounterICU.ndjson":
                patient_rows[patient_id]["icu_los_days"] += los_days
            else:
                patient_rows[patient_id]["hospital_los_days"] += los_days

    return encounter_to_patient, patient_first_start


def add_age(patient_rows, patient_first_start):
    for patient_id, row in patient_rows.items():
        birth_date = row.get("birth_date")
        birth_dt = parse_datetime(birth_date)
        first_start = patient_first_start.get(patient_id)

        if birth_dt and first_start:
            age_days = (first_start.date() - birth_dt.date()).days
            row["age"] = int(age_days / 365.25)
        else:
            row["age"] = None


def count_conditions_24h(patient_rows, patient_first_start):
    conditions = load_ndjson(DATA_DIR / "Condition.ndjson")

    for condition in conditions:
        patient_id = get_subject_patient_id(condition)

        if patient_id not in patient_rows:
            continue

        event_time = get_event_time(condition)

        if is_within_prediction_window(patient_id, event_time, patient_first_start):
            patient_rows[patient_id]["condition_count"] += 1


def count_medications_24h(patient_rows, patient_first_start):
    medication_files = [
        DATA_DIR / "MedicationRequest.ndjson",
        DATA_DIR / "MedicationAdministration.ndjson",
        DATA_DIR / "MedicationAdministrationICU.ndjson",
        DATA_DIR / "MedicationDispense.ndjson",
    ]

    for file_path in medication_files:
        resources = load_ndjson(file_path)

        for resource in resources:
            patient_id = get_subject_patient_id(resource)

            if patient_id not in patient_rows:
                continue

            event_time = get_event_time(resource)

            if is_within_prediction_window(patient_id, event_time, patient_first_start):
                patient_rows[patient_id]["medication_event_count"] += 1


def count_procedures_24h(patient_rows, patient_first_start):
    procedure_files = [
        DATA_DIR / "Procedure.ndjson",
        DATA_DIR / "ProcedureICU.ndjson",
    ]

    for file_path in procedure_files:
        resources = load_ndjson(file_path)

        for resource in resources:
            patient_id = get_subject_patient_id(resource)

            if patient_id not in patient_rows:
                continue

            event_time = get_event_time(resource)

            if is_within_prediction_window(patient_id, event_time, patient_first_start):
                patient_rows[patient_id]["procedure_count"] += 1


def add_observation_features_24h(patient_rows, encounter_to_patient, patient_first_start):
    observation_values = defaultdict(lambda: defaultdict(list))

    observation_files = [
        DATA_DIR / "ObservationChartevents.ndjson",
        DATA_DIR / "ObservationLabevents.ndjson",
        DATA_DIR / "ObservationOutputevents.ndjson",
        DATA_DIR / "ObservationDatetimeevents.ndjson",
    ]

    for file_path in observation_files:
        observations = load_ndjson(file_path)

        for observation in observations:
            patient_id = get_subject_patient_id(observation)

            if not patient_id:
                encounter_id = get_encounter_id(observation)
                patient_id = encounter_to_patient.get(encounter_id)

            if patient_id not in patient_rows:
                continue

            value = get_numeric_value(observation)

            if value is None:
                continue

            display_name = get_observation_display(observation)
            feature_name = match_observation_feature(display_name)

            if not feature_name:
                continue

            obs_time = get_observation_time(observation)

            if not is_within_prediction_window(patient_id, obs_time, patient_first_start):
                continue

            observation_values[patient_id][feature_name].append(value)

    for patient_id, feature_dict in observation_values.items():
        for feature_name, values in feature_dict.items():
            if values:
                patient_rows[patient_id][f"{feature_name}_mean_24h"] = sum(values) / len(values)
                patient_rows[patient_id][f"{feature_name}_min_24h"] = min(values)
                patient_rows[patient_id][f"{feature_name}_max_24h"] = max(values)


def add_target(df):
    df["target_long_icu_stay"] = (df["icu_los_days"] >= 3.0).astype(int)

    if df["target_long_icu_stay"].nunique() < 2:
        positive_los = df[df["icu_los_days"] > 0]["icu_los_days"]

        if len(positive_los) > 0:
            median_los = positive_los.median()
            df["target_long_icu_stay"] = (
                df["icu_los_days"] >= median_los
            ).astype(int)

    return df


def build_feature_dataframe():
    patient_rows = build_patient_base()

    encounter_to_patient, patient_first_start = build_encounter_maps(patient_rows)

    add_age(patient_rows, patient_first_start)
    count_conditions_24h(patient_rows, patient_first_start)
    count_medications_24h(patient_rows, patient_first_start)
    count_procedures_24h(patient_rows, patient_first_start)
    add_observation_features_24h(patient_rows, encounter_to_patient, patient_first_start)

    df = pd.DataFrame(list(patient_rows.values()))

    df = add_target(df)

    return df


def main():
    df = build_feature_dataframe()

    df.to_csv(FEATURES_CSV, index=False)

    print(f"Saved leakage-safe features to: {FEATURES_CSV}")
    print()
    print("Dataset shape:")
    print(df.shape)
    print()
    print("Columns:")
    print(df.columns.tolist())
    print()
    print("Target distribution:")
    print(df["target_long_icu_stay"].value_counts())


if __name__ == "__main__":
    main()