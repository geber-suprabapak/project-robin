# Face Verification

1. Astra sends an authenticated identification request with explicit user context and image payload.
2. Robin validates request/image limits, decodes the image, and requires an acceptable face-detection result.
3. ONNX inference produces an embedding.
4. Robin queries the user’s Qdrant embeddings and computes the technical match response.

Qdrant failure and invalid/unavailable enrollment are reported as technical failures; no attendance record is written by Robin.

**Evidence:** `src/api/routes/identification.py`, `src/services/qdrant_client.py`, `tests/test_identification.py`.
