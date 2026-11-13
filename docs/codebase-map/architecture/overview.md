# Robin Architecture

FastAPI routes validate service-authenticated requests (requiring `X-Request-ID` and `X-Astra-Contract-Version: v1` headers on `/v1/*` routes) and pass image data through decoding/face detection/inference before Qdrant persistence or similarity lookup. Health routes distinguish process liveness from read-only operational readiness, while startup lifespan initializes the Qdrant collection. Astra is the documented caller and sends explicit user context.

**Evidence:** `src/main.py`, `src/api/routes/identification.py`, `src/api/routes/enrollment.py`, `src/services/qdrant_client.py`, `README.md`.
