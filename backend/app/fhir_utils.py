import json
from datetime import datetime, timedelta
from pathlib import Path


def load_ndjson(file_path: Path):
    """Read an NDJSON file into a list of FHIR resources."""
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
    """Convert FHIR references such as ``Patient/123`` to ``123``."""
    if not reference_value:
        return None

    if "/" in reference_value:
        return reference_value.rstrip("/").split("/")[-1]

    return reference_value


def get_subject_patient_id(resource):
    subject = resource.get("subject", {})
    return get_reference_id(subject.get("reference"))


def get_encounter_id(resource):
    """
    Return the encounter referenced by a resource.

    Most resources use ``encounter``. MIMIC MedicationAdministration resources
    use ``context`` instead, so both are supported.
    """
    encounter = resource.get("encounter")
    if isinstance(encounter, dict):
        encounter_id = get_reference_id(encounter.get("reference"))
        if encounter_id:
            return encounter_id

    context = resource.get("context")
    if isinstance(context, dict):
        return get_reference_id(context.get("reference"))

    return None


def parse_datetime(value):
    """Parse a FHIR date/dateTime value safely."""
    if not value:
        return None

    try:
        value = str(value).replace("Z", "+00:00")
        return datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None


def get_period_start(resource):
    period = resource.get("period", {})
    return parse_datetime(period.get("start"))


def get_period_end(resource):
    period = resource.get("period", {})
    return parse_datetime(period.get("end"))


def get_observation_time(resource):
    """Return the clinically relevant time of a FHIR Observation."""
    if "effectiveDateTime" in resource:
        return parse_datetime(resource.get("effectiveDateTime"))

    effective_period = resource.get("effectivePeriod")
    if isinstance(effective_period, dict):
        parsed = parse_datetime(effective_period.get("start"))
        if parsed:
            return parsed

    if "issued" in resource:
        return parse_datetime(resource.get("issued"))

    return None


def get_numeric_value(resource):
    """
    Backwards-compatible numeric extractor.

    New feature code should prefer ``get_quantity`` when unit information
    matters. This helper intentionally keeps the original simple behavior for
    callers that only need a number.
    """
    quantity = get_quantity(resource)
    if quantity is not None:
        return quantity["value"]

    for field in ("valueInteger", "valueDecimal"):
        if field in resource:
            try:
                return float(resource[field])
            except (TypeError, ValueError):
                return None

    return None


def get_quantity(resource):
    """
    Return a FHIR ``valueQuantity`` without discarding its unit metadata.

    Returns ``None`` when no usable numeric ``valueQuantity`` is present.
    """
    value_quantity = resource.get("valueQuantity")
    if not isinstance(value_quantity, dict):
        return None

    try:
        value = float(value_quantity.get("value"))
    except (TypeError, ValueError):
        return None

    return {
        "value": value,
        "unit": value_quantity.get("unit"),
        "system": value_quantity.get("system"),
        "code": value_quantity.get("code"),
    }


def get_observation_codings(resource):
    """Return all complete coding entries from ``Observation.code``."""
    code = resource.get("code", {})
    codings = code.get("coding", []) if isinstance(code, dict) else []

    return [coding for coding in codings if isinstance(coding, dict)]


def get_observation_codes(resource):
    """Return ``(system, code)`` tuples for all Observation codings."""
    output = []
    for coding in get_observation_codings(resource):
        code = coding.get("code")
        if code:
            output.append((coding.get("system"), str(code)))
    return output


def get_observation_display(resource):
    code = resource.get("code", {})

    if isinstance(code, dict) and code.get("text"):
        return code["text"]

    for coding in get_observation_codings(resource):
        if coding.get("display"):
            return coding["display"]
        if coding.get("code"):
            return str(coding["code"])

    return "unknown_observation"


def days_between(start, end):
    if not start or not end:
        return 0.0

    days = (end - start).total_seconds() / (60 * 60 * 24)
    return max(days, 0.0)


def get_event_time(resource):
    """
    Return a clinically relevant event timestamp from common FHIR fields.

    Events without a usable timestamp are intentionally excluded from
    first-24-hour features rather than being assigned an invented time.
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
    if event_time is None:
        return False

    first_start = patient_first_start.get(patient_id)
    if first_start is None:
        return False

    first_end = first_start + timedelta(hours=24)
    return first_start <= event_time <= first_end
