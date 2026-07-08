import json
from datetime import datetime, timedelta
from pathlib import Path

def load_ndjson(file_path: Path):
    """
    Reads an NDJSON file.
    Each line is one JSON/FHIR resource.
    """
    resources = []

    if not file_path.exists():
        return resources

    with open(file_path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                resources.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"Could not parse line {line_number} in {file_path.name}")

    return resources


def get_reference_id(reference_value):
    """
    Converts:
        Patient/123 -> 123
        Encounter/abc -> abc
    """
    if not reference_value:
        return None

    if "/" in reference_value:
        return reference_value.split("/")[-1]

    return reference_value


def get_subject_patient_id(resource):
    """
    Gets patient id from resource['subject']['reference'].
    """
    subject = resource.get("subject", {})
    reference = subject.get("reference")
    return get_reference_id(reference)


def get_encounter_id(resource):
    """
    Gets encounter id from resource['encounter']['reference'].
    """
    encounter = resource.get("encounter", {})
    reference = encounter.get("reference")
    return get_reference_id(reference)


def parse_datetime(value):
    """
    Parses FHIR datetime safely.
    """
    if not value:
        return None

    try:
        value = value.replace("Z", "+00:00")
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def get_period_start(resource):
    period = resource.get("period", {})
    return parse_datetime(period.get("start"))


def get_period_end(resource):
    period = resource.get("period", {})
    return parse_datetime(period.get("end"))


def get_observation_time(resource):
    """
    Observation time can be stored in different FHIR fields.
    """
    if "effectiveDateTime" in resource:
        return parse_datetime(resource.get("effectiveDateTime"))

    if "issued" in resource:
        return parse_datetime(resource.get("issued"))

    effective_period = resource.get("effectivePeriod", {})
    if effective_period:
        return parse_datetime(effective_period.get("start"))

    return None


def get_numeric_value(resource):
    """
    Gets numeric value from FHIR Observation.
    """
    if "valueQuantity" in resource:
        value = resource["valueQuantity"].get("value")
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    if "valueInteger" in resource:
        try:
            return float(resource["valueInteger"])
        except (TypeError, ValueError):
            return None

    if "valueDecimal" in resource:
        try:
            return float(resource["valueDecimal"])
        except (TypeError, ValueError):
            return None

    return None


def get_observation_display(resource):
    """
    Gets human-readable name of an Observation.
    """
    code = resource.get("code", {})

    if code.get("text"):
        return code["text"]

    codings = code.get("coding", [])
    if codings:
        first_coding = codings[0]

        if first_coding.get("display"):
            return first_coding["display"]

        if first_coding.get("code"):
            return first_coding["code"]

    return "unknown_observation"


def days_between(start, end):
    if not start or not end:
        return 0.0

    seconds = (end - start).total_seconds()
    days = seconds / (60 * 60 * 24)

    if days < 0:
        return 0.0

    return days

def get_event_time(resource):
    """
    Gets a clinically relevant event timestamp from common FHIR fields.

    Used to prevent leakage:
    if an event has no timestamp, we do not use it as a first-24h feature.
    """
    direct_datetime_fields = [
        "effectiveDateTime",
        "authoredOn",
        "recordedDate",
        "issued",
        "onsetDateTime",
        "performedDateTime",
        "occurrenceDateTime",
        "whenHandedOver",
        "whenPrepared",
    ]

    for field in direct_datetime_fields:
        if field in resource:
            parsed = parse_datetime(resource.get(field))
            if parsed:
                return parsed

    period_fields = [
        "effectivePeriod",
        "performedPeriod",
        "period",
        "validityPeriod",
    ]

    for field in period_fields:
        period = resource.get(field)

        if isinstance(period, dict):
            parsed = parse_datetime(period.get("start"))

            if parsed:
                return parsed

    return None


def is_within_first_24h(patient_id, event_time, patient_first_start):
    """
    Returns True only if the event is known to be inside the first 24h.
    Missing timestamps are treated as unsafe and excluded.
    """
    if event_time is None:
        return False

    first_start = patient_first_start.get(patient_id)

    if first_start is None:
        return False

    first_end = first_start + timedelta(hours=24)

    return first_start <= event_time <= first_end