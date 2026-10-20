"""Permit fees in SAR, implementing policy POL-02 (see docs/POL-02_fees.md)."""

from __future__ import annotations

MINIMUM = {"BLD": 500, "REN": 300, "SGN": 400}
ILLUMINATED_SURCHARGE = 0.25


def area(m2: float) -> int:
    """Areas are rounded to whole square metres before a fee is computed."""
    return round(m2)


def permit_fee(permit_type: str, *, area_m2: float = 0, structural: bool = False, sign_m2: float = 0,
               illuminated: bool = False, event_days: int = 0, social_security: bool = False) -> int:
    """Fee for one application, in whole SAR."""
    t = permit_type.upper()
    if social_security:
        return 0
    if t == "BLD":
        return max(MINIMUM["BLD"], 2 * area(area_m2))
    if t == "REN":
        return 300 + (area(area_m2) if structural else 0)
    if t == "SGN":
        fee = 200 * area(sign_m2)
        if illuminated:
            fee = fee * (1 + ILLUMINATED_SURCHARGE)
        return int(max(MINIMUM["SGN"], fee))
    if t == "FTR":
        return 1500
    if t == "EVT":
        if event_days > 14:
            raise ValueError("an event permit covers at most 14 days")
        return 1000 * event_days
    raise ValueError(f"unknown permit type {permit_type!r}")
