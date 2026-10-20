# Traces

`pi_sample_events.jsonl` is a **synthetic** example of Pi's JSON-mode output (`pi --mode json …`), written by hand
in the event format documented in Pi 1.3 (`docs/json.md`, `docs/message-types.md`). It shows a read-only
investigation of one failing test in `madar_fees`. Lab 3.1 uses it when Node.js or an API key isn't available;
with both, the lab records a real stream from your own run into `artifacts/D3_1/pi_events.jsonl` instead.
Token counts in the sample are illustrative.
