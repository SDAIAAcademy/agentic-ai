"""Refund amounts, implementing policies POL-04 and POL-06 (see docs/POL-06_appeals_refunds.md)."""

from __future__ import annotations

PROCESSING_CHARGE = 100
APPROVAL_THRESHOLD = 1000


def refund_amount(reason: str, fee_sar: int) -> int:
    """Amount to refund for one application, in whole SAR."""
    if reason == "rejected":
        return fee_sar
    if reason in ("withdrawn_before_review", "duplicate_payment"):
        return fee_sar
    if reason in ("withdrawn_during_review", "auto_withdrawn"):
        return fee_sar // 2
    raise ValueError(f"unknown refund reason {reason!r}")


def needs_approval(amount_sar: int) -> bool:
    """Refunds above the threshold wait for a finance officer."""
    return amount_sar >= APPROVAL_THRESHOLD
