import pytest

from madar_fees import permit_fee

# Worked examples from POL-02


def test_building_minimum_applies():
    assert permit_fee("BLD", area_m2=180.4) == 500


def test_building_large():
    assert permit_fee("BLD", area_m2=640) == 1280


def test_area_rounds_up():
    # 250.2 m² is billed as 251 m²
    assert permit_fee("BLD", area_m2=250.2) == 502


def test_renovation_structural():
    assert permit_fee("REN", area_m2=35, structural=True) == 335


def test_renovation_non_structural_ignores_area():
    assert permit_fee("REN", area_m2=80) == 300


def test_illuminated_sign():
    assert permit_fee("SGN", sign_m2=3, illuminated=True) == 750


def test_small_sign_minimum():
    assert permit_fee("SGN", sign_m2=1.5) == 400


def test_small_illuminated_sign_surcharge_after_minimum():
    # 1 m² -> 200, minimum 400, then +25%
    assert permit_fee("SGN", sign_m2=1, illuminated=True) == 500


def test_event_days():
    assert permit_fee("EVT", event_days=3) == 3000


def test_event_too_long():
    with pytest.raises(ValueError):
        permit_fee("EVT", event_days=15)


def test_social_security_exempts_renovation():
    assert permit_fee("REN", area_m2=20, social_security=True) == 0


def test_social_security_does_not_exempt_other_types():
    assert permit_fee("BLD", area_m2=640, social_security=True) == 1280
