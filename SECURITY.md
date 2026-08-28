# Security Policy

## Supported Versions

| Version | Supported |
| --- | --- |
| `1.x` (master) | ✅ Active development |

---

## Table of Contents

- [Security Posture](#security-posture)
- [Authentication & Authorization](#authentication--authorization)
- [Request Validation & Guardrails](#request-validation--guardrails)
- [Supply Chain Security](#supply-chain-security)
- [Container Security](#container-security)
- [Database Security](#database-security)
- [CI/CD Security](#cicd-security)
- [Error Handling & Information Leakage](#error-handling--information-leakage)
- [Threat Model](#threat-model)
- [Security Recommendations](#security-recommendations)
- [Reporting a Vulnerability](#reporting-a-vulnerability)

---

## Security Posture

Project Robin is a **face recognition attendance API** deployed on-prem or on internal school/agency networks. It handles **biometric data (face embeddings)** and receives a scoped user identifier from Astra; it does not access the domain database or identity provider directly. Below is the threat model and security controls currently in place.

### Design Principles

- **Biometric data minimization**: Only 512-dimensional embeddings (not raw images) are stored long-term in Qdrant. Uploaded images are processed in-memory and discarded after inference. No temporary image files are written to disk by Project Robin.
- **1:1 verification (not 1:N)**: The `/v1/identify` endpoint never searches the entire face database. It retrieves only embeddings belonging to the authenticated user and computes cosine similarity server-side inside the API. Stored embeddings are never returned to the client. This prevents user enumeration via the vector database.
- **Defense in depth**: Multi-layer input validation, checksum-verified model downloads, private network placement, and Qdrant API-key protection.

---

## Authentication & Authorization

### Astra Service Authentication

All protected endpoints (`/v1/enroll`, `/v1/enroll/status`, `/v1/identify`) require Astra's **Bearer service credential** and the `X-Astra-User-Id` header.

Validation (`src/dependencies.py`):

| Check | Implementation |
| --- | --- |
| Credential | `ROBIN_SERVICE_TOKEN` shared only between Astra and Robin |
| User context | `X-Astra-User-Id`, supplied after Astra authenticates the caller |
| Direct client access | Not supported; Robin is an internal service |
| Failure mode | Missing or mismatched credential/context returns 401 |

### Authorization Model

```
Astra-authenticated user context → Qdrant scope (`user_id`)
```

Every operation is scoped to the authenticated user's identity. The API never accesses data outside the caller's user boundary.

### Credential Requirements

| Credential | Purpose | Risk if leaked |
| --- | --- | --- |
| `ROBIN_SERVICE_TOKEN` | Authenticate Astra-to-Robin calls | Unauthorized callers can invoke protected face operations |
| `QDRANT_API_KEY` | Vector DB authentication | Access to stored embeddings |

---

## Request Validation & Guardrails

Multi-layer input validation protects against malformed or malicious requests:

```
HTTP Request
  │  Layer 1: Content-Length check (MAX_REQUEST_BYTES ≈ 60 MB)
  ▼
Middleware
  │  Layer 2: Base64 regex + padding validation (no decoding)
  ▼
Pydantic Schema
  │  Layer 3: Estimated decoded size, format constraints
  ▼
Image Decoder
  │  Layer 4: Actual byte size, decompression bomb prevention,
  │           dimension limits (4096×4096), pixel cap (16 MP)
  ▼
Face Detector
     Layer 5: Single-face enforcement, confidence threshold
```

### Guardrail Defaults

| Parameter | Value | Purpose |
| --- | --- | --- |
| `MAX_REQUEST_BYTES` | 62,914,560 (60 MB) | Request size limit |
| `MAX_IMAGE_BYTES` | 5,242,880 (5 MB) | Decoded image limit |
| `MAX_IMAGE_WIDTH` | 4096 px | Width limit |
| `MAX_IMAGE_HEIGHT` | 4096 px | Height limit |
| `MAX_IMAGE_PIXELS` | 16,777,216 (4096²) | Decompression bomb prevention |
| `Face confidence` | 0.5 (configurable) | Minimum face detection confidence |

---

## Supply Chain Security

### Dependency Management

- **Python dependency versions** are pinned in `pyproject.toml` and locked in `uv.lock`
- **Lockfile verification** runs in CI (`uv lock --check`)
- **Frozen installs** at deploy time (`uv sync --frozen`)
- **No `requirements.txt`** — uv.lock is the single source of truth

### Model Download Integrity

| Asset | Source | Integrity Check |
| --- | --- | --- |
| AuraFace ONNX (`glintr100.onnx`) | HuggingFace | SHA-256 checksum |
| Face detector prototxt (`deploy.prototxt`) | OpenCV GitHub | SHA-256 checksum |
| Face detector weights (`res10_300x300.caffemodel`) | OpenCV 3rdparty | SHA-256 checksum |

- Downloads are written to `.filename.download` temp files, then atomically renamed via `os.replace()`
- Temp files are cleaned up on any error
- Multiple hash algorithms supported: SHA-256, SHA-384, SHA-512 (auto-detected)

---

## Container Security

### Dockerfile (`docker/Dockerfile`)

| Aspect | Status |
| --- | --- |
| Base image | `python:3.12-slim-bookworm` (CPU), `nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04` (GPU) |
| Non-root user | **Not configured** — runs as root |
| Minimum packages | `ca-certificates`, `curl`, `libglib2.0-0`, `libgomp1` |
| Layer caching | Dependencies installed before source copy |
| Healthcheck | `/ready` endpoint every 30s, 40s start period |
| Python hardening | `PYTHONUNBUFFERED=1`, `PYTHONDONTWRITEBYTECODE=1` |

### Docker Compose

| Aspect | Status |
| --- | --- |
| Secrets | Passed via environment variables (not Docker secrets) |
| Restart policy | `unless-stopped` |
| Network exposure | Internal-only; published by the `infra` stack only behind Astra |
| Read-only volumes | Model directory mounted as `:ro` in dev |

---

## Database Security

### Qdrant (Vector Database)

- Embeddings are stored per-user with `user_id` as payload
- Queries are scoped to the authenticated user — no cross-user access
- Connection uses configurable timeout (`QDRANT_TIMEOUT_SECONDS`, default 3s)

---

## CI/CD Security

### GitHub Actions

| Workflow | Trigger | Security Controls |
| --- | --- | --- |
| CPU CI/CD | Push/PR to `master` | Lockfile check → lint → test (70% coverage) → build → push to GHCR |
| GPU CI/CD | Manual dispatch | Build → push to GHCR |

- **GHCR authentication** uses `secrets.GITHUB_TOKEN` (scoped, short-lived)
- **Image push** only occurs on `push` to `master`, never on PRs
- **Permissions**: `contents: read`, `packages: write` — least privilege
- **`.dockerignore`** excludes `.env`, `.git`, `tests/`, `sql/`, `models/`

---

## Error Handling & Information Leakage

- Internal error details (stack traces) are **suppressed in production** and only exposed in development mode
- Custom exception handler catches unhandled exceptions to prevent default FastAPI traceback exposure
- Dependency health status is exposed via `/ready` endpoint for operational monitoring

---

## Threat Model

### Assets Protected

| Asset | Location | Sensitivity |
| --- | --- | --- |
| Face embeddings (512-d vectors) | Qdrant | **High** — biometric data |
| User profiles (PII) | Astra/domain store | **High** |
| Face photos (uploaded) | Memory only | **High** — discarded post-inference |
| Service credentials | Astra and Robin host-side env files | **Critical** |
| API keys & secrets | Environment variables | **Critical** |
| ONNX model weights | Filesystem | **Low** — publicly available |

### Threat Scenarios

| Threat | Likelihood | Impact | Mitigation |
| --- | --- | --- | --- |
| Service credential misuse | Low | High | Private network placement, constant-time credential comparison, credential rotation |
| Embedding extraction via API | Low | Medium | 1:1 verification only, no batch/export endpoint |
| Decompression bomb | Low | High | Multi-layer image size validation |
| Model poisoning (MITM download) | Low | High | SHA-256 checksum verification |
| Container escape (root user) | Low | High | Mitigated by network isolation and non-root user (`appuser`) |
| Dependency vulnerability | Medium | Varies | Lockfile pinning, regular `uv sync` updates |
| Service role key leak | Low | Critical | Never committed, env var only |
| Brute-force enrollment | Medium | Low | Rate limiting recommended (not yet implemented) |

---

## Security Recommendations

### High Priority

1. **Run container as non-root user** — Add `USER appuser` to Dockerfile after package installation.
2. **Add per-user and per-IP rate limiting** — Implement rate limiting on `/v1/enroll` and `/v1/identify` endpoints to prevent brute-force, resource exhaustion on CPU/GPU, and abuse of biometric processing.
3. **Use secrets management** — Replace plain environment variables with Docker secrets, HashiCorp Vault, or cloud secrets manager in production.
4. **Rotate the Astra service credential** — Keep `ROBIN_SERVICE_TOKEN` out of images, Git, and client applications; rotate it through the shared host-side secret files.

### Medium Priority

5. **Structured audit logging** — Log identity verification events (who, when, confidence) for non-repudiation.
6. **Configure explicit CORS origins** — For browser-based deployments, set `CORS_ALLOWED_ORIGINS` to explicit values instead of `*`.
7. **TLS termination** — Deploy behind a reverse proxy (nginx, Caddy, Traefik) for HTTPS. The API does not handle TLS natively.

### Low Priority

8. **Zero-trust model download** — Pin model download URLs to specific content hashes for stronger supply chain integrity.
9. **Security headers** — Add `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY` via middleware or reverse proxy.
10. **Token revocation** — If user deactivation is required before JWT expiry, implement a token blacklist.

---

## Responsible Use

Project Robin is intended for **1:1 attendance verification in controlled institutional environments** (schools, universities, offices). It should not be used for:

- Mass surveillance or public-space identification
- Covert recognition without consent
- 1:N watchlist matching or unknown-subject search
- Any purpose that violates applicable privacy or biometric laws

Deploying organizations are responsible for ensuring their use complies with all relevant regulations.

---

## Reporting a Vulnerability

**This project is in active development and not yet deployed in production.** Security issues can be reported via:

1. **GitHub Issues** — For non-critical security bugs (with label `security`)
2. **Direct contact** — Reach out to the repository owner for critical vulnerabilities

Please **do not** publicly disclose security vulnerabilities until they have been addressed.

For vulnerabilities in dependencies (`fastapi`, `qdrant-client`, `onnxruntime`, etc.), refer to the respective project's security advisory process.
