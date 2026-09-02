# Invariants

## INV-ROBIN-001 — Astra mediates main face operations

**Rule:** Preserve private service authentication and explicit Astra user context.
**Evidence:** `README.md`, `src/dependencies.py`.

## INV-ROBIN-002 — Enrollment requires exactly ten valid images

**Rule:** Do not weaken the enrollment count and image-validation requirements.
**Evidence:** `src/api/routes/enrollment.py`, `tests/test_enrollment.py`.

## INV-ROBIN-003 — Robin does not own attendance state

**Rule:** A verification result remains technical; Astra makes domain attendance decisions.
**Evidence:** `README.md`.

## INV-ROBIN-004 — Readiness includes dependencies

**Rule:** Keep liveness and readiness distinct, including Qdrant/model availability.
**Evidence:** `src/api/routes/health.py`, `README.md`.
