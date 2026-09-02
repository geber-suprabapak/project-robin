# Robin Architecture

FastAPI routes validate service-authenticated requests and pass image data through decoding/face detection/inference before Qdrant persistence or similarity lookup. Health routes distinguish process liveness from operational readiness. Astra is the documented caller and sends explicit user context.

**Evidence:** `src/main.py`, `src/api/routes/identification.py`, `src/api/routes/enrollment.py`, `src/services/qdrant_client.py`, `README.md`.
