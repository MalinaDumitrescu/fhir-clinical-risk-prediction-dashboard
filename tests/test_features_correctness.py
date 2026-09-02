from backend.app.features import _map_observation


CHART_SYSTEM = "http://fhir.mimic.mit.edu/CodeSystem/chartevents-d-items"
LAB_SYSTEM = "http://fhir.mimic.mit.edu/CodeSystem/d-labitems"


def observation(system, code, value, unit):
    return {
        "resourceType": "Observation",
        "code": {"coding": [{"system": system, "code": code}]},
        "valueQuantity": {"value": value, "unit": unit},
    }


def test_heart_rate_alarm_is_not_mapped():
    assert _map_observation(
        observation(CHART_SYSTEM, "220046", 150, "bpm")
    ) is None


def test_true_heart_rate_is_mapped():
    assert _map_observation(
        observation(CHART_SYSTEM, "220045", 88, "bpm")
    ) == ("heart_rate", 88.0)


def test_lactate_dehydrogenase_is_not_lactate():
    assert _map_observation(
        observation(LAB_SYSTEM, "50954", 300, "IU/L")
    ) is None


def test_lactate_is_mapped():
    assert _map_observation(
        observation(LAB_SYSTEM, "50813", 2.4, "mmol/L")
    ) == ("lactate", 2.4)


def test_temperature_fahrenheit_is_converted_to_celsius():
    feature, value = _map_observation(
        observation(CHART_SYSTEM, "223761", 98.6, "°F")
    )
    assert feature == "temperature"
    assert abs(value - 37.0) < 1e-6


def test_wrong_unit_is_rejected():
    assert _map_observation(
        observation(LAB_SYSTEM, "50912", 1.2, "mmol/L")
    ) is None
