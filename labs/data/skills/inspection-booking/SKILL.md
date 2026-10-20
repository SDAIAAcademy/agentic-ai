---
name: inspection-booking
description: Book or reschedule a Madar inspection (foundation, completion, food-truck health, event site visit). Use when someone wants an inspection slot or to move one.
---

# Inspection booking

Policy: POL-05 (inspections), POL-03 (working days), POL-08 (identity).

1. **Verify identity** with `get_permit`. Read the permit's `status`, `district` and booked `inspections`.
2. **Check eligibility:** `approved` permits (BLD foundation/completion, FTR health) or `under_review`
   (FTR and EVT pre-approval visits). Otherwise explain why no inspection can be booked yet.
3. **Find slots** with `list_slots(district, date_from, date_to)` for the permit's own district.
   - Slots are Sunday–Thursday, 08:00–13:00 starts, at least **2 working days** from today.
   - Offer at most three slots that match the person's preference (day, morning/afternoon).
4. **Book only after the person picks a slot:** `book_inspection(permit_id, slot, ...)`.
5. **Rescheduling:** `reschedule_inspection(inspection_id, new_slot, ...)` is free up to 24 hours before the
   current slot. Closer than that, explain the 150 SAR rebooking fee for a missed inspection instead.
6. Confirm the final slot, the district and what to prepare (approved drawings on site, site access).
