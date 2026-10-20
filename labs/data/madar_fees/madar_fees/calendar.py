"""Working days and SLA due dates, implementing policy POL-03 (see docs/POL-03_service_levels.md)."""

from __future__ import annotations

from datetime import date, timedelta

WEEKEND = {4}  # Friday
HOLIDAYS = {date(2026, 9, 23), date(2027, 2, 22)}
SLA_DAYS = {"BLD": 15, "REN": 7, "SGN": 5, "FTR": 10, "EVT": 10}


def is_working_day(d: date) -> bool:
    return d.weekday() not in WEEKEND


def add_working_days(start: date, n: int) -> date:
    """The date n working days after start (start itself does not count)."""
    d = start
    while n > 0:
        d += timedelta(days=1)
        if is_working_day(d):
            n -= 1
    return d


def sla_due(permit_type: str, submitted: date, committee_review: bool = False) -> date:
    """Last working day for a decision. Signs reviewed by the Urban Design Committee get 10 extra working days."""
    days = SLA_DAYS[permit_type.upper()]
    if committee_review:
        days += 10
    return add_working_days(submitted, days)
