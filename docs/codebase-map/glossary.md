# Glossary

## Astra service boundary

**Meaning:** Robin trusts Astra’s private credential and explicit user context for main operations.
**Evidence:** `README.md`, `src/dependencies.py`.

## Enrollment

**Meaning:** Replace a user’s stored Qdrant embeddings from exactly ten validated face images.
**Evidence:** `src/api/routes/enrollment.py`, `tests/test_enrollment.py`.

## Face verification result

**Meaning:** Technical match/quality/score output; not an attendance record.
**Evidence:** `README.md`, `src/api/routes/identification.py`.

## Qdrant

**Meaning:** Vector database that stores and retrieves face embeddings.
**Evidence:** `README.md`, `src/services/qdrant_client.py`.
