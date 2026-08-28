# Privacy Policy

**Project Robin** — Face recognition attendance API, version `1.x`.

*Last updated: 2026-06-08*

---

## 1. Data Controller

Project Robin is deployed and operated by the educational institution or organization that runs the API. This document describes the privacy characteristics of the software itself; operational privacy policies should be defined by the deploying organization.

---

## 2. What Data Is Collected

### 2.1 Biometric Data (Face Embeddings)

When a user enrolls (`POST /v1/enroll`), **10 face photos** are uploaded and processed:

| Stage | Data Held | Duration |
| --- | --- | --- |
| Upload | Full-resolution face photo (JPEG/PNG) | **Temporary** — discarded after processing |
| Detection | Cropped face region | **Temporary** — discarded after inference |
| Embedding | 512-dimensional float vector | **Persistent** — stored in Qdrant |

The raw face photo is **never stored permanently by Project Robin**. Only the 512-dimensional embedding vector is retained in Qdrant.

Face embeddings are not raw images and are not designed to reconstruct the original face. However, they are still biometric data and must be treated as sensitive personal data. Deploying organizations should protect embeddings with strong access controls, encryption in transit, restricted database access, and appropriate retention/deletion policies.

### 2.2 User Identity Context

Project Robin does not store or resolve user profiles. Astra supplies an
already-authenticated user identifier in `X-Astra-User-Id` together with its
private service credential. Student names, classes, and other domain records
remain in the integrating application.

### 2.3 Attendance Records

Each successful identification (`POST /v1/identify`) produces an attendance record containing:

- User ID and student ID
- Verification confidence score
- Timestamp

Attendance records are managed by the integrating application, not by this API directly.

### 2.4 Technical Metadata

- Request timestamps and HTTP status codes (in application logs)
- Logs are written to `/app/logs` (configurable volume mount)
- Project Robin's application logger does not intentionally log IP addresses, User-Agent strings, session cookies, request bodies, face images, or embeddings by default. However, reverse proxies, container runtimes, hosting platforms, or custom access logs may collect additional network metadata depending on deployment configuration.

---

## 3. How Data Is Processed

### 3.1 Face Image Processing Pipeline

```
Upload
  │
  ▼
Content-Length check           ← no data stored
  │
  ▼
Base64 decode                  ← in memory only
  │
  ▼
Face detection (OpenCV DNN)    ← in memory only
  │
  ▼
ONNX inference (AuraFace)      ← produces embedding
  │
  ├──► Embedding → Qdrant    (enrollment)
  └──► Embedding → compare   (identification, then discarded)
```

- All image processing occurs in **memory** — no temporary files are written to disk
- Face detection uses OpenCV's DNN module (no cloud API)
- Face recognition runs locally via ONNX Runtime (no cloud API)

### 3.2 Where Face Images Are NOT Sent

| Service | What leaves the server |
| --- | --- |
| Astra | User identity context — **no face images or embeddings** |
| Qdrant | 512-d embedding vectors — **no face images** |
| HuggingFace / OpenCV assets (optional) | Public model file downloads — **no user images, embeddings, or identity data sent** |
| Other external APIs | **None** — Project Robin does not send user images, embeddings, or identity data to external ML APIs |

Face images **never leave the server** running Project Robin.

---

## 4. Data Storage

### 4.1 Qdrant (Vector Database)

| Aspect | Detail |
| --- | --- |
| Data stored | 512-d face embedding + user_id payload |
| Retention | Until explicitly deleted via enrollment replacement or Qdrant management |
| Encryption at rest | Depends on Qdrant deployment (self-hosted or Qdrant Cloud) |
| Encryption in transit | Via HTTPS (configured by `QDRANT_HTTPS` and `QDRANT_API_KEY`) |
| Isolation | Embeddings are scoped by `user_id` — no cross-user queries |

### 4.2 Logs (Local Volume)

- Written to configurable log directory (`./logs` by default)
- Contains timestamps and HTTP request/response summaries
- **Does not contain** face images, embeddings, or full request bodies
- Retention is managed by the deploying institution

---

## 5. Data Deletion

### 5.1 User-Initiated Deletion

- **Re-enrollment**: Calling `POST /v1/enroll` overwrites all existing embeddings for the user in Qdrant
- **Direct Qdrant access**: Institutions can delete specific points by `user_id` via the Qdrant REST API

### 5.2 Institutional Deletion

To fully remove a user's biometric data:

1. **Qdrant**: Delete points matching `user_id` via Qdrant API
2. **Domain system**: Apply the institution's user-data deletion policy in Astra or its backing store
3. **Logs**: Purge relevant log entries from the mounted log volume

There is no API endpoint for bulk or individual embedding deletion — this is by design to prevent unauthorized data removal. Institutions manage deletion through direct database/vector-store administration.

---

## 6. Data Sharing

Project Robin **does not**:

- Share face embeddings or face images with any third party
- Upload biometric data to cloud services
- Use external face recognition APIs
- Collect data for training or model improvement
- Include analytics, telemetry, or phone-home functionality

---

## 7. Data Retention Policy

| Data Type | Default Retention | Controlled By |
| --- | --- | --- |
| Face embeddings | Until explicitly deleted or overwritten | Institution (Qdrant admin) |
| User identity/profile data | Managed by the integrating domain system | Institution |
| Application logs | Until disk rotation or manual purge | Institution (log volume) |
| Uploaded face images | **Deleted immediately after processing** | Automatic (code-level) |

---

## 8. User Rights

The deploying institution should provide mechanisms for:

- **Access**: Confirm whether a user's embeddings exist via `GET /v1/enroll/status`
- **Rectification**: Re-enrollment overwrites old embeddings with new ones
- **Deletion**: Managed by institution via Qdrant and domain-system administration tools
- **Portability**: 512-d embeddings can be exported directly from Qdrant

The API itself is a tool; user rights administration is the responsibility of the deploying organization under applicable data protection regulations (GDPR, UU PDP, etc.).

---

## 9. Consent and Legal Basis

Project Robin does not collect consent by itself. Deploying organizations are responsible for obtaining valid consent or another lawful basis before enrolling users' biometric data. This includes providing notice, explaining retention/deletion procedures, and offering alternatives where required by applicable law.

---

## 10. Security Measures Protecting Privacy

| Measure | Detail |
| --- | --- |
| Service authentication | Every protected request requires Astra's service credential and user context |
| 1:1 verification | Users can only verify against their own embeddings |
| In-memory processing | Face images never written to persistent storage |
| No telemetry | Zero phone-home, analytics, or usage tracking |
| Local inference | All ML processing on-device/on-prem — no external API calls |
| Input sanitization | Multi-layer validation prevents malformed payloads |
| Dependency integrity | Lockfile-pinned dependencies, checksum-verified models |
| Minimal logging | No PII or biometric data written to logs |

---

## 11. Third-Party Services

| Service | Data Shared | Purpose |
| --- | --- | --- |
| Astra | User identity context | Authenticated domain integration |
| Qdrant | 512-d embedding vectors | Vector storage and similarity search |
| HuggingFace / OpenCV assets (optional) | No user data | Public model and detector asset downloads |

No third-party service receives face images or raw biometric data.

---

## 12. Changes to This Policy

This document is updated as the software evolves. Check the Git history for changes:

```bash
git log --oneline PRIVACY.md
```

Significant changes will be noted in release notes or commit messages.

---

## 13. Contact

For privacy-related inquiries about this software, open an issue on the repository. For operational privacy questions (how your institution uses this software), contact your institution's data protection officer or system administrator.

---

*This document describes the privacy characteristics of the software. It does not constitute legal advice. Deploying organizations should consult with legal counsel to ensure compliance with applicable privacy regulations.*
