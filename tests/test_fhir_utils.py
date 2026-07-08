from datetime import datetime, timezone

from backend.app.fhir_utils import get_reference_id, parse_datetime, days_between


def test_get_reference_id():
    assert get_reference_id("Patient/abc") == "abc"
    assert get_reference_id("Encounter/123") == "123"
    assert get_reference_id("abc") == "abc"
    assert get_reference_id(None) is None


def test_parse_datetime():
    parsed = parse_datetime("2020-01-01T10:00:00Z")

    assert parsed is not None
    assert parsed.year == 2020


def test_days_between():
    start = datetime(2020, 1, 1, tzinfo=timezone.utc)
    end = datetime(2020, 1, 3, tzinfo=timezone.utc)

    assert days_between(start, end) == 2.0