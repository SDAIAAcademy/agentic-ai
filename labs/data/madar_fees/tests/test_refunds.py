from madar_fees import needs_approval, refund_amount

# POL-04 and POL-06


def test_rejected_keeps_processing_charge():
    assert refund_amount("rejected", 2800) == 2700


def test_withdrawn_before_review_full():
    assert refund_amount("withdrawn_before_review", 335) == 335


def test_withdrawn_during_review_half():
    assert refund_amount("withdrawn_during_review", 3000) == 1500


def test_duplicate_payment_full():
    assert refund_amount("duplicate_payment", 500) == 500


def test_approval_only_above_threshold():
    assert needs_approval(1001)
    assert not needs_approval(1000)
    assert not needs_approval(500)
