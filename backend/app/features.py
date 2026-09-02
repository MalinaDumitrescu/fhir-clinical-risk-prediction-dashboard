from collections import defaultdict
from datetime import timedelta

import pandas as pd

from backend.app.config import DATA_DIR, FEATURES_CSV
from backend.app.fhir_utils import (
    days_between,
    get_encounter_id,
    get_event_time,
    get_observation_codes,
    get_observation_time,
    get_period_end,
    get_period_start,
    get_quantity,
    get_reference_id,
    get_subject_patient_id,
    load_ndjson,
    parse_datetime,
)


PREDICTION_WINDOW_HOURS = 24

MIMIC_LAB_SYSTEM = "http://fhir.mimic.mit.edu/CodeSystem/d-labitems"
MIMIC_CHART_SYSTEM = "http://fhir.mimic.mit.edu/CodeSystem/chartevents-d-items"

# Deliberately conservative mappings. These codes were verified against the
# supplied MIMIC-IV-on-FHIR resources. Broad display-name matching is avoided
# because it previously mixed measurements with alarm limits, ventilator
# settings, urine tests, LDH, pulmonary-artery pressures, etc.
LAB_CODE_MAP = {
    "50912": ("creatinine", {"mg/dl"}),
    "50971": ("potassium", {"meq/l"}),
    "51222": ("hemoglobin", {"g/dl"}),
    "50983": ("sodium", {"meq/l"}),
    "50813": ("lactate", {"mmol/l"}),
    "51301": ("white_blood_cells", {"k/ul"}),
    "50931": ("glucose", {"mg/dl"}),
}

CHART_CODE_MAP = {
    "220045": ("heart_rate", {"bpm"}),
    "220210": ("respiratory_rate", {"insp/min"}),
    "220277": ("spo2", {"%"}),
    "220179": ("systolic_bp", {"mmhg"}),
    "220050": ("systolic_bp", {"mmhg"}),
    "220180": ("diastolic_bp", {"mmhg"}),
    "220051": ("diastolic_bp", {"mmhg"}),
    "223761": ("temperature", {"°f", "degf", "f"}),
    "223762": ("temperature", {"°c", "degc", "c"}),
}

TEMPERATURE_F_CODE = "223761"
TEMPERATURE_C_CODE = "223762"


FEATURE_NAMES = [
    "creatinine",
    "potassium",
    "hemoglobin",
    "sodium",
    "lactate",
    "white_blood_cells",
    "glucose",
    "heart_rate",
    "respiratory_rate",
    "spo2",
    "systolic_bp",
    "diastolic_bp",
    "temperature",
]


def _normalize_unit(unit):
    if unit is None:
        return None

    return (
        str(unit)
        .strip()
        .lower()
        .replace("μ", "u")
        .replace("µ", "u")
        .replace(" ", "")
        .replace(".", "")
    )


def _resource_belongs_to_index_admission(resource, patient_id, eligible_encounter_ids):
    """
    If the resource carries an encounter/context reference, require it to
    belong to the patient's index admission. Untagged resources are allowed
    only because the subsequent timestamp window still has to match.
    """
    encounter_id = get_encounter_id(resource)
    if not encounter_id:
        return True

    return encounter_id in eligible_encounter_ids.get(patient_id, set())


def _is_within_prediction_window(patient_id, event_time, patient_index_start):
    if event_time is None:
        return False

    start = patient_index_start.get(patient_id)
    if start is None:
        return False

    end = start + timedelta(hours=PREDICTION_WINDOW_HOURS)
    return start <= event_time <= end


def build_patient_base():
    rows = {}

    for patient in load_ndjson(DATA_DIR / "Patient.ndjson"):
        patient_id = patient.get("id")
        if not patient_id:
            continue

        rows[patient_id] = {
            "patient_id": patient_id,
            "gender": patient.get("gender", "unknown"),
            "birth_date": patient.get("birthDate"),
            "deceased": 1 if patient.get("deceasedDateTime") else 0,
            "index_hospital_encounter_id": None,
            "medication_request_count": 0,
            "medication_administration_count": 0,
            "procedure_count": 0,
            "icu_los_days": 0.0,
            "hospital_los_days": 0.0,
        }

    return rows


def build_index_admission_maps(patient_rows):
    """
    Define one explicit index hospital admission per patient.

    The earliest hospital Encounter is the index admission. ICU Encounters are
    linked through ``EncounterICU.partOf`` and only ICU time belonging to this
    index admission contributes to the target.
    """
    encounter_to_patient = {}
    hospital_encounters_by_patient = defaultdict(list)

    hospital_encounters = load_ndjson(DATA_DIR / "Encounter.ndjson")
    icu_encounters = load_ndjson(DATA_DIR / "EncounterICU.ndjson")

    for encounter in hospital_encounters:
        encounter_id = encounter.get("id")
        patient_id = get_subject_patient_id(encounter)
        start = get_period_start(encounter)

        if encounter_id and patient_id:
            encounter_to_patient[encounter_id] = patient_id

        if patient_id in patient_rows and encounter_id and start:
            hospital_encounters_by_patient[patient_id].append(encounter)

    patient_index_start = {}
    patient_index_encounter = {}
    eligible_encounter_ids = defaultdict(set)

    for patient_id, encounters in hospital_encounters_by_patient.items():
        index_encounter = min(encounters, key=get_period_start)
        index_id = index_encounter.get("id")
        start = get_period_start(index_encounter)
        end = get_period_end(index_encounter)

        patient_index_encounter[patient_id] = index_id
        patient_index_start[patient_id] = start
        eligible_encounter_ids[patient_id].add(index_id)

        patient_rows[patient_id]["index_hospital_encounter_id"] = index_id
        patient_rows[patient_id]["hospital_los_days"] = days_between(start, end)

    for encounter in icu_encounters:
        encounter_id = encounter.get("id")
        patient_id = get_subject_patient_id(encounter)

        if encounter_id and patient_id:
            encounter_to_patient[encounter_id] = patient_id

        if patient_id not in patient_rows or not encounter_id:
            continue

        parent_id = get_reference_id(
            (encounter.get("partOf") or {}).get("reference")
        )

        if parent_id != patient_index_encounter.get(patient_id):
            continue

        eligible_encounter_ids[patient_id].add(encounter_id)
        patient_rows[patient_id]["icu_los_days"] += days_between(
            get_period_start(encounter),
            get_period_end(encounter),
        )

    return encounter_to_patient, patient_index_start, eligible_encounter_ids


def add_age(patient_rows, patient_index_start):
    for patient_id, row in patient_rows.items():
        birth_dt = parse_datetime(row.get("birth_date"))
        index_start = patient_index_start.get(patient_id)

        if birth_dt and index_start:
            age_days = (index_start.date() - birth_dt.date()).days
            row["age"] = int(age_days / 365.25)
        else:
            row["age"] = None


def count_medications_24h(patient_rows, patient_index_start, eligible_encounter_ids):
    request_seen = defaultdict(set)
    administration_seen = defaultdict(set)

    for resource in load_ndjson(DATA_DIR / "MedicationRequest.ndjson"):
        patient_id = get_subject_patient_id(resource)
        if patient_id not in patient_rows:
            continue
        if not _resource_belongs_to_index_admission(resource, patient_id, eligible_encounter_ids):
            continue
        if not _is_within_prediction_window(patient_id, get_event_time(resource), patient_index_start):
            continue

        resource_id = resource.get("id")
        if resource_id and resource_id not in request_seen[patient_id]:
            request_seen[patient_id].add(resource_id)
            patient_rows[patient_id]["medication_request_count"] += 1

    for filename in ("MedicationAdministration.ndjson", "MedicationAdministrationICU.ndjson"):
        for resource in load_ndjson(DATA_DIR / filename):
            patient_id = get_subject_patient_id(resource)
            if patient_id not in patient_rows:
                continue
            if not _resource_belongs_to_index_admission(resource, patient_id, eligible_encounter_ids):
                continue
            if not _is_within_prediction_window(patient_id, get_event_time(resource), patient_index_start):
                continue

            resource_id = resource.get("id")
            dedupe_key = f"{filename}:{resource_id}" if resource_id else None
            if dedupe_key and dedupe_key not in administration_seen[patient_id]:
                administration_seen[patient_id].add(dedupe_key)
                patient_rows[patient_id]["medication_administration_count"] += 1


def count_procedures_24h(patient_rows, patient_index_start, eligible_encounter_ids):
    seen = defaultdict(set)

    for filename in ("Procedure.ndjson", "ProcedureICU.ndjson"):
        for resource in load_ndjson(DATA_DIR / filename):
            patient_id = get_subject_patient_id(resource)
            if patient_id not in patient_rows:
                continue
            if not _resource_belongs_to_index_admission(resource, patient_id, eligible_encounter_ids):
                continue
            if not _is_within_prediction_window(patient_id, get_event_time(resource), patient_index_start):
                continue

            resource_id = resource.get("id")
            dedupe_key = f"{filename}:{resource_id}" if resource_id else None
            if dedupe_key and dedupe_key not in seen[patient_id]:
                seen[patient_id].add(dedupe_key)
                patient_rows[patient_id]["procedure_count"] += 1


def _map_observation(resource):
    quantity = get_quantity(resource)
    if quantity is None:
        return None

    unit = _normalize_unit(quantity.get("unit") or quantity.get("code"))
    value = quantity["value"]

    for system, code in get_observation_codes(resource):
        mapping = None

        if system == MIMIC_LAB_SYSTEM:
            mapping = LAB_CODE_MAP.get(code)
        elif system == MIMIC_CHART_SYSTEM:
            mapping = CHART_CODE_MAP.get(code)

        if mapping is None:
            continue

        feature_name, accepted_units = mapping
        if unit not in accepted_units:
            return None

        if code == TEMPERATURE_F_CODE:
            value = (value - 32.0) * 5.0 / 9.0
        elif code == TEMPERATURE_C_CODE:
            value = value

        return feature_name, float(value)

    return None


def add_observation_features_24h(
    patient_rows,
    encounter_to_patient,
    patient_index_start,
    eligible_encounter_ids,
):
    values = defaultdict(lambda: defaultdict(list))
    seen_ids = set()

    # Outputevents and Datetimeevents are deliberately excluded: they do not
    # represent the 13 numeric lab/vital concepts used by this model.
    for filename in ("ObservationChartevents.ndjson", "ObservationLabevents.ndjson"):
        for observation in load_ndjson(DATA_DIR / filename):
            observation_id = observation.get("id")
            dedupe_key = f"{filename}:{observation_id}" if observation_id else None
            if dedupe_key and dedupe_key in seen_ids:
                continue
            if dedupe_key:
                seen_ids.add(dedupe_key)

            patient_id = get_subject_patient_id(observation)
            if not patient_id:
                patient_id = encounter_to_patient.get(get_encounter_id(observation))

            if patient_id not in patient_rows:
                continue

            if not _resource_belongs_to_index_admission(
                observation,
                patient_id,
                eligible_encounter_ids,
            ):
                continue

            obs_time = get_observation_time(observation)
            if not _is_within_prediction_window(patient_id, obs_time, patient_index_start):
                continue

            mapped = _map_observation(observation)
            if mapped is None:
                continue

            feature_name, value = mapped
            values[patient_id][feature_name].append(value)

    for patient_id, feature_dict in values.items():
        for feature_name, feature_values in feature_dict.items():
            if not feature_values:
                continue

            patient_rows[patient_id][f"{feature_name}_mean_24h"] = (
                sum(feature_values) / len(feature_values)
            )
            patient_rows[patient_id][f"{feature_name}_min_24h"] = min(feature_values)
            patient_rows[patient_id][f"{feature_name}_max_24h"] = max(feature_values)


def add_target(df):
    """Target: >=3 ICU days during the explicit index hospital admission."""
    df["target_long_icu_stay"] = (df["icu_los_days"] >= 3.0).astype(int)
    return df


def build_feature_dataframe():
    patient_rows = build_patient_base()

    encounter_to_patient, patient_index_start, eligible_encounter_ids = (
        build_index_admission_maps(patient_rows)
    )

    add_age(patient_rows, patient_index_start)
    count_medications_24h(patient_rows, patient_index_start, eligible_encounter_ids)
    count_procedures_24h(patient_rows, patient_index_start, eligible_encounter_ids)
    add_observation_features_24h(
        patient_rows,
        encounter_to_patient,
        patient_index_start,
        eligible_encounter_ids,
    )

    df = pd.DataFrame(list(patient_rows.values()))
    df = add_target(df)
    return df


def main():
    df = build_feature_dataframe()
    df.to_csv(FEATURES_CSV, index=False)

    print(f"Saved corrected leakage-safe features to: {FEATURES_CSV}")
    print("Dataset shape:", df.shape)
    print("Target distribution:")
    print(df["target_long_icu_stay"].value_counts())


if __name__ == "__main__":
    main()
