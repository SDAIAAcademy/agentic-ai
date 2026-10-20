---
name: escalation
description: Escalate an overdue Madar application to the duty supervisor. Use when a decision is late and the person asks what to do or asks to escalate.
---

# Escalation of an overdue application

Policy: POL-03 (service levels), POL-08 (identity).

1. **Verify identity** with `get_permit`. Read `sla_due` and `working_days_overdue`.
2. If `working_days_overdue` is 0, the application is **not** late: give the due date and stop. Do not escalate.
3. If it is overdue, tell the person by how many **working days** (Sunday–Thursday, excluding public holidays),
   and that a breached SLA does **not** approve the application automatically.
4. Offer to escalate. If they agree, call `escalate(permit_id, note, ...)` with a short factual note:
   the permit type, the submission date and the days overdue. No opinions about the outcome.
5. Tell them the duty supervisor replies within **2 working days**.
