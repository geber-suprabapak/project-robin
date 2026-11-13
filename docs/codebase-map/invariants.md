# Invariants

## INV-ROBIN-001 — Astra mediates main face operations

**Rule:** Preserve private service authentication, explicit Astra user context, and `X-Request-ID` + `X-Astra-Contract-Version: v1` headers on `/v1/*` routes.
**Evidence:** `README.md`, `src/dependencies.py`, `src/main.py`.

## INV-ROBIN-002 — Enrollment requires exactly ten valid images

**Rule:** Do not weaken the enrollment count and image-validation requirements; enrollment upserts deterministic UUIDv5 point IDs before deleting stale points so existing embeddings survive failed upserts.
**Evidence:** `src/api/routes/enrollment.py`, `src/services/qdrant_client.py`, `tests/test_enrollment.py`, `tests/test_qdrant_client.py`.

## INV-ROBIN-003 — Robin does not own attendance state

**Rule:** A verification result remains technical; Astra makes domain attendance decisions.
**Evidence:** `README.md`.

## INV-ROBIN-004 — Readiness includes dependencies

**Rule:** Keep liveness and readiness distinct, including Qdrant collection existence and model availability; `/ready` is strictly read-only and never auto-creates a missing Qdrant collection.
**Evidence:** `src/api/routes/health.py`, `src/services/qdrant_client.py`, `README.md`.
