---
name: refund-request
description: Create a refund request for a Madar permit (rejected, withdrawn, or paid twice). Use when someone asks for money back.
---

# Refund request

Follow these steps in order. Policy: POL-06 (refunds), POL-08 (identity).

1. **Verify identity.** Call `get_permit` with the permit ID, the last 4 digits of the national ID and the
   registered mobile. Do not continue if verification fails.
2. **Find the situation** from the permit status and the person's words:

   | Situation | `reason` | Refund |
   |---|---|---|
   | status `rejected` | `rejected` | fee − 100 SAR |
   | withdrawn before review started | `withdrawn_before_review` | full fee |
   | withdrawn during review, or after an expired `needs_info` | `withdrawn_during_review` | 50% of the fee |
   | the same fee was paid twice | `duplicate_payment` | the duplicate amount |

3. **Tell the person the amount before you create anything**, and ask them to confirm.
4. Call `create_refund(permit_id, reason, national_id_last4, mobile)`.
5. **Explain what happens next:**
   - up to 1,000 SAR: paid automatically to the original payment method within 10 working days;
   - above 1,000 SAR: status `pending_approval` until a finance officer approves it, then 10 working days.
6. A rejected applicant may prefer to **appeal** (POL-06, 30 calendar days). Mention it once; don't push.

Never promise a date earlier than policy allows, and never create two refunds for the same payment.
