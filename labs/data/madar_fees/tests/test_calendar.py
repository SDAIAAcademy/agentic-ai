from datetime import date

from madar_fees import add_working_days, is_working_day, sla_due

# POL-03: Sunday to Thursday are working days; Friday, Saturday and public holidays are not.


def test_weekend_is_friday_and_saturday():
    assert not is_working_day(date(2026, 10, 30))  # Friday
    assert not is_working_day(date(2026, 10, 31))  # Saturday
    assert is_working_day(date(2026, 11, 1))  # Sunday


def test_national_day_is_not_a_working_day():
    assert not is_working_day(date(2026, 9, 23))


def test_add_working_days_skips_weekend():
    # Thursday + 1 working day = Sunday
    assert add_working_days(date(2026, 10, 29), 1) == date(2026, 11, 1)


def test_renovation_sla():
    # submitted Thursday 22 Oct; 7 working days -> Monday 2 Nov
    assert sla_due("REN", date(2026, 10, 22)) == date(2026, 11, 2)


def test_building_sla():
    assert sla_due("BLD", date(2026, 9, 28)) == date(2026, 10, 19)


def test_sla_skips_holiday():
    # submitted Monday 21 Sep 2026; National Day (Wed 23 Sep) does not count
    assert sla_due("SGN", date(2026, 9, 21)) == date(2026, 9, 29)


def test_committee_review_adds_ten_days():
    assert sla_due("SGN", date(2026, 10, 1), committee_review=True) == date(2026, 10, 22)
