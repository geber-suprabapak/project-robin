# Project Robin

Project Robin is an open-source, self-hosted face recognition API for 1:1 attendance verification in schools and institutions.

It is designed for controlled, consent-based deployments where biometric data stays on-prem or inside the deploying organization's infrastructure.

> **Face recognition attendance API** — 1:1 face verification untuk sistem presensi sekolah/lembaga berbasis Astra dan Qdrant.

Project Robin adalah REST API internal yang menerima foto wajah dari Astra, memproses embedding dengan model ONNX (AuraFace), lalu membandingkannya dengan embedding yang tersimpan di Qdrant. Astra memverifikasi identitas dan meneruskan `X-Astra-User-Id` bersama service credential. Target deployment utama adalah server on-prem atau server internal sekolah/lembaga.

---

## Daftar Isi

- [Fitur](#fitur)
- [Arsitektur](#arsitektur)
- [Cara Kerja](#cara-kerja)
- [Prasyarat](#prasyarat)
- [Quick Start](#quick-start)
- [Endpoint API](#endpoint-api)
- [Use Case & Contoh Request](#use-case--contoh-request)
- [Konfigurasi](#konfigurasi)
- [Model & Asset](#model--asset)
- [Health Check](#health-check)
- [Error Reference](#error-reference)
- [Development](#development)
- [GPU Runtime](#gpu-runtime)
- [CI/CD](#cicd)
- [Quality Check Lokal](#quality-check-lokal)

---

## Fitur

| Fitur | Keterangan |
| --- | --- |
| **Verifikasi presensi 1:1** | Cocokkan wajah user yang sedang login terhadap embedding miliknya di Qdrant |
| **Enrollment mandiri** | User mengirim 10 foto untuk menyimpan embedding ke Qdrant |
| **Status enrollment** | Cek apakah user sudah punya embedding tersimpan |
| **Astra service boundary** | Endpoint fitur utama hanya menerima credential service Astra dan user context eksplisit |
| **Health & readiness check** | Bedakan "proses hidup" vs "semua dependency siap" |
| **Guardrail request** | Batasan ukuran request, image, resolusi, dan pixel |
| **CPU & GPU image** | Image resmi tersedia untuk CPU (default) dan NVIDIA GPU |
| **Auto-download model** | Opsional; download AuraFace dari HuggingFace dengan verifikasi checksum |

---

## Arsitektur

```
Astra API (auth + domain)
        │  private service credential + X-Astra-User-Id
        ▼
┌───────────────────┐              ┌──────────────┐
│   FastAPI API     │─────────────▶│   Qdrant     │
│   Project Robin   │              │ vector store │
│  ┌─────────────┐  │              └──────────────┘
│  │ ONNX Runtime│  │
│  │ + OpenCV DNN│  │
│  └─────────────┘  │
└───────────────────┘
```

**Komponen runtime:**

- **FastAPI** — HTTP API, routing, middleware, auth dependency, response handling
- **ONNX Runtime** — menjalankan model face recognition dari file `.onnx`
- **OpenCV DNN** — face detector untuk memastikan foto berisi tepat satu wajah
- **Astra** — identity and domain boundary; Robin tidak mengakses database domain atau provider identity secara langsung
- **Qdrant** — vector database untuk penyimpanan dan pencarian embedding wajah

> **Catatan:** Qdrant tidak dijalankan oleh Compose repository ini. Gunakan Qdrant external—baik self-hosted maupun Qdrant Cloud.

---

## Cara Kerja

### Alur Presensi (`POST /v1/identify`)

```
Client                         Astra                    Robin        Qdrant
  │                             │                           │               │
  │── attendance request ──────▶│                           │               │
  │                             │── service call ──────────▶│               │
  │                             │   X-Astra-User-Id        │               │
  │   { "image_base64": "..." } │                           │               │
  │                             │                           │               │
  │                             │                           │               │
  │                             │  [decode & validate image]│               │
  │                             │  [detect face → crop]     │               │
  │                             │  [run ONNX → embedding]   │               │
  │                             │                           │               │
  │                             │── search embedding ───────────────────────▶│
  │                             │◀─ stored embedding ────────────────────────│
  │                             │                           │               │
  │                             │  [cosine similarity 1:1]  │               │
  │                             │◀─ verification result ────│               │
  │                             │                           │               │
  │◀── 200 OK ─────────────────│                           │               │
  │    { status, student_id,    │                           │               │
  │      confidence, ... }      │                           │               │
```

### Alur Enrollment (`POST /v1/enroll`)

1. Astra mengirim service credential, `X-Astra-User-Id`, dan tepat **10 foto wajah** via `multipart/form-data`
2. API memverifikasi service credential dan memakai user context dari Astra
3. Setiap foto dicek validitas dan dipastikan berisi tepat satu wajah
4. Setiap foto diproses menjadi embedding 512-dimensi
5. Embedding lama user di Qdrant diganti dengan embedding baru
6. API mengembalikan jumlah image berhasil diproses dan total embedding tersimpan

---

## Prasyarat

Sebelum menjalankan Project Robin, siapkan:

1. **Docker & Docker Compose** — untuk menjalankan container API
2. **Astra service credential** — credential internal dan user context dari Astra
3. **Qdrant instance** — self-hosted atau Qdrant Cloud (external; tidak dijalankan Compose ini)
4. **Model ONNX (AuraFace)** — file `glintr100.onnx` di direktori `./models/`

### Mendapatkan Model AuraFace

Model tidak dibake ke image Docker. Download manual sebelum start:

```bash
# Buat direktori models
mkdir -p models

# Download model AuraFace (glintr100.onnx)
curl -L "https://huggingface.co/fal/AuraFace-v1/resolve/686774cad65e40933022896195e07e01ee06bee9/glintr100.onnx" \
     -o models/glintr100.onnx

# Verifikasi checksum (Linux/macOS)
sha256sum models/glintr100.onnx
# Expected: a7933ea5330113b01c9b60351d8f4c33003f145d8470ac5f0e52ee2effe25c60
```

Atau set `AUTO_DOWNLOAD_MODELS=true` di `.env` agar API mengunduh otomatis saat startup (membutuhkan akses internet).

---

## Quick Start

### 1. Clone dan siapkan environment

```bash
git clone https://github.com/fizray/project-robin.git
cd project-robin

# Buat file .env dari template
cp .env.example .env
```

### 2. Isi konfigurasi di `.env`

Minimal yang wajib diisi:

```ini
# Astra service boundary
ROBIN_SERVICE_TOKEN=replace-with-the-same-secret-as-astra

# Qdrant (external)
QDRANT_HOST=https://your-qdrant-host.example.com
QDRANT_API_KEY=your-qdrant-api-key   # Jika pakai Qdrant Cloud

# Qdrant collection (buat manual di Qdrant sebelum start jika collection belum ada)
QDRANT_COLLECTION_NAME=face_embeddings
```

### 3. Letakkan model ONNX

```bash
# Taruh file glintr100.onnx di sini:
./models/glintr100.onnx
```

### 4. Pull image dan jalankan

```bash
docker compose pull
docker compose up -d
```

API tersedia di `http://localhost:8000`.

### 5. Verifikasi API berjalan

```bash
# Cek liveness (proses hidup)
curl http://localhost:8000/live

# Cek readiness (semua dependency siap)
curl http://localhost:8000/ready
```

Response readiness OK:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "face_detector_ready": true,
  "gpu_available": false,
  "qdrant_connected": true
}
```

---

## Endpoint API

| Method | Path | Auth | Fungsi |
| --- | --- | --- | --- |
| `GET` | `/` | ✗ | Metadata API (versi, nama) |
| `GET` | `/live` | ✗ | Liveness — proses API hidup |
| `GET` | `/ready` | ✗ | Readiness — semua dependency siap |
| `GET` | `/health` | ✗ | Alias `/ready` (backward compat) |
| `GET` | `/v1/enroll/status` | ✔ Bearer JWT | Cek apakah user sudah enrollment |
| `POST` | `/v1/enroll` | ✔ Bearer JWT | Enrollment wajah user |
| `POST` | `/v1/identify` | ✔ Bearer JWT | Verifikasi wajah untuk presensi |

---

## Use Case & Contoh Request

### Use Case 1: User pertama kali pakai aplikasi (enrollment)

Sebelum bisa melakukan presensi, user perlu mendaftarkan wajahnya. Kirim tepat **10 foto** wajah yang jelas.

**Request:**

```bash
curl -X POST http://localhost:8000/v1/enroll \
  -H "Authorization: Bearer <robin-service-token>" \
  -H "X-Astra-User-Id: <user-id>" \
  -F "files=@foto1.jpg" \
  -F "files=@foto2.jpg" \
  -F "files=@foto3.jpg" \
  -F "files=@foto4.jpg" \
  -F "files=@foto5.jpg" \
  -F "files=@foto6.jpg" \
  -F "files=@foto7.jpg" \
  -F "files=@foto8.jpg" \
  -F "files=@foto9.jpg" \
  -F "files=@foto10.jpg"
```

**Response berhasil (`200`):**

```json
{
  "status": "success",
  "student_id": "550e8400-e29b-41d4-a716-446655440000",
  "images_processed": 10,
  "images_failed": 0,
  "total_embeddings": 10,
  "message": "Face enrolled successfully with 10 images"
}
```

**Response jika ada foto yang gagal (wajah tidak terdeteksi, dll):**

```json
{
  "status": "success",
  "student_id": "550e8400-e29b-41d4-a716-446655440000",
  "images_processed": 9,
  "images_failed": 1,
  "total_embeddings": 9,
  "message": "Face enrolled successfully with 9 images"
}
```

---

### Use Case 2: Cek status enrollment

Aplikasi client dapat mengecek apakah user sudah enrollment sebelum memperbolehkan presensi.

**Request:**

```bash
curl http://localhost:8000/v1/enroll/status \
  -H "Authorization: Bearer <robin-service-token>" \
  -H "X-Astra-User-Id: <user-id>"
```

**Response sudah enrollment (`200`):**

```json
{
  "is_enrolled": true,
  "embedding_count": 10,
  "user_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

**Response belum enrollment (`200`):**

```json
{
  "is_enrolled": false,
  "embedding_count": 0,
  "user_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

---

### Use Case 3: Presensi harian (identifikasi wajah)

User melakukan presensi dengan mengirim satu foto wajah dalam format base64.

**Request:**

```bash
curl -X POST http://localhost:8000/v1/identify \
  -H "Authorization: Bearer <robin-service-token>" \
  -H "X-Astra-User-Id: <user-id>" \
  -H "Content-Type: application/json" \
  -d '{
    "image_base64": "<base64-encoded-image>"
  }'
```

**Response wajah cocok (`200`):**

```json
{
  "status": "ok",
  "student_id": "12345678",
  "student_name": "Ahmad Rizki",
  "confidence": 0.95,
  "process_time_ms": 45,
  "message": "Face verified successfully"
}
```

**Response user belum enrollment (`200`):**

```json
{
  "status": "not_found",
  "student_id": null,
  "student_name": null,
  "confidence": null,
  "process_time_ms": 45,
  "message": "No face embeddings enrolled for this user"
}
```

**Response wajah tidak cocok (`200`):**

```json
{
  "status": "not_found",
  "student_id": null,
  "student_name": null,
  "confidence": 0.42,
  "process_time_ms": 50,
  "message": "Face did not match enrolled embeddings"
}
```

> **Catatan:** Similarity di bawah `FACE_MATCH_THRESHOLD` (default `0.6`) dianggap tidak cocok. Field `status` berisi `"not_found"` hanya jika processing berhasil tapi wajah tidak cocok — bukan karena error dependency.

---

### Contoh encode image ke base64 (Python)

```python
import base64

with open("foto.jpg", "rb") as f:
    image_base64 = base64.b64encode(f.read()).decode("utf-8")

# Kirim ke API
import httpx

response = httpx.post(
    "http://localhost:8000/v1/identify",
    headers={
        "Authorization": f"Bearer {robin_service_token}",
        "X-Astra-User-Id": user_id,
    },
    json={"image_base64": image_base64},
)
print(response.json())
```

---

## Konfigurasi

Buat `.env` dari `.env.example`, lalu isi nilainya sesuai deployment.

### Konfigurasi Wajib

| Env | Keterangan |
| --- | --- |
| `ROBIN_SERVICE_TOKEN` | Credential internal yang dibagikan Astra saat memanggil Robin |
| `QDRANT_HOST` | Host atau URL Qdrant external |
| `QDRANT_COLLECTION_NAME` | Nama collection embedding (default: `face_embeddings`) |

### Model & Inference

| Env | Default | Keterangan |
| --- | --- | --- |
| `MODEL_PATH` | `/app/runtime-models/glintr100.onnx` | Path model ONNX di dalam container |
| `MODEL_INPUT_SIZE` | `112` | Ukuran input model (AuraFace: 112×112) |
| `EMBEDDING_DIM` | `512` | Dimensi vektor embedding |
| `AUTO_DOWNLOAD_MODELS` | `false` | Set `true` agar API unduh model otomatis saat startup |
| `MODEL_DOWNLOAD_URL` | HuggingFace URL | URL download model AuraFace |
| `MODEL_DOWNLOAD_CHECKSUM` | `sha256:...` | Checksum untuk verifikasi download |
| `SKIP_MODEL_LOAD` | `false` | `true` hanya untuk CI smoke test, **jangan** untuk runtime |

### Qdrant

| Env | Default | Keterangan |
| --- | --- | --- |
| `QDRANT_HOST` | `localhost` | Host/URL Qdrant |
| `QDRANT_PORT` | `6333` | Port REST Qdrant |
| `QDRANT_HTTPS` | `false` | Gunakan HTTPS untuk koneksi Qdrant |
| `QDRANT_API_KEY` | _(kosong)_ | API key Qdrant Cloud |
| `QDRANT_TIMEOUT_SECONDS` | `3` | Timeout request ke Qdrant |

### Face Recognition

| Env | Default | Keterangan |
| --- | --- | --- |
| `FACE_MATCH_THRESHOLD` | `0.6` | Cosine similarity minimum untuk dianggap cocok (0.0–1.0) |

### GPU

| Env | Default | Keterangan |
| --- | --- | --- |
| `GPU_DEVICE_ID` | `-1` | `-1` = CPU only; `0` = GPU pertama |
| `GPU_MEM_LIMIT` | `2147483648` | Batas memori GPU dalam bytes (default: 2 GB) |

### Guardrail Request & Image

| Env | Default | Keterangan |
| --- | --- | --- |
| `MAX_REQUEST_BYTES` | `62914560` | Maks ukuran HTTP request (60 MB) |
| `MAX_IMAGE_BYTES` | `5242880` | Maks ukuran image setelah decode (5 MB) |
| `MAX_IMAGE_WIDTH` | `4096` | Lebar maksimum image |
| `MAX_IMAGE_HEIGHT` | `4096` | Tinggi maksimum image |
| `MAX_IMAGE_PIXELS` | `16777216` | Total pixel maksimum (16 MP) |

> Validasi dan resize utama sebaiknya dilakukan di sisi client. Guardrail server adalah lapisan pengaman tambahan jika request dikirim langsung atau client bug.

### Image & Server

| Env | Default | Keterangan |
| --- | --- | --- |
| `IMAGE_NAME` | `ghcr.io/geber-suprabapak/project-robin` | Nama Docker image |
| `IMAGE_TAG` | `cpu-latest` | Tag image CPU yang dipakai Compose |
| `API_PORT` | `8000` | Port API di dalam container |
| `API_WORKERS` | `1` | Jumlah worker Uvicorn (production mode) |
| `ENVIRONMENT` | `production` | `development` atau `production` |
| `CORS_ALLOWED_ORIGINS` | `*` | Allowed origins untuk CORS |

---

## Model & Asset

### Model Face Recognition (AuraFace)

Model **tidak dibake** ke image Docker. File harus tersedia di `./models/` sebelum container start.

```
./models/
└── glintr100.onnx          ← mount ke /app/runtime-models/ di container
```

**Kenapa AuraFace?** Model card AuraFace menyatakan lisensi `apache-2.0` dan dilatih untuk skenario commercial setting. Untuk production, tetap lakukan evaluasi internal pada data institusi Anda karena model card juga mencatat bahwa performa dapat bervariasi antar etnis.

> ⚠️ **Hindari** InsightFace default pack (`buffalo_l`, `antelopev2`) untuk production tanpa commercial license eksplisit.

Jika mengganti model, override variabel berikut:

```ini
MODEL_PATH=/app/runtime-models/nama-model.onnx
MODEL_INPUT_SIZE=112
MODEL_DOWNLOAD_URL=https://...
MODEL_DOWNLOAD_CHECKSUM=sha256:...
```

### Face Detector (OpenCV DNN)

Face detector asset **sudah dibake** ke official Docker image saat build. Tidak perlu setup manual.

```
/app/models/face_detector/deploy.prototxt
/app/models/face_detector/res10_300x300_ssd_iter_140000.caffemodel
```

---

## Health Check

### `/live` — Liveness

Menandakan proses API hidup. **Tidak** mengecek model atau Qdrant. Cocok untuk Docker liveness probe.

```bash
curl http://localhost:8000/live
# 200 OK — proses hidup
```

### `/ready` dan `/health` — Readiness

Mengecek apakah API siap melayani request nyata. Mengembalikan `200` hanya jika **semua** komponen wajib siap:

- ✅ Model ONNX sudah loaded
- ✅ Face detector asset tersedia
- ✅ Qdrant reachable

Jika salah satu gagal, response `503`:

```json
{
  "status": "unhealthy",
  "model_loaded": true,
  "face_detector_ready": true,
  "gpu_available": false,
  "qdrant_connected": false
}
```

> Kondisi Qdrant down **tidak** disamarkan menjadi "user belum enroll" atau "user tidak ditemukan".

---

## Error Reference

### HTTP Status Codes

| Status | Kondisi |
| --- | --- |
| `200` | Request valid dan berhasil diproses |
| `400` | Image tidak valid, jumlah file enrollment salah, atau wajah tidak valid/terdeteksi |
| `401` | Credential Astra tidak ada/invalid atau user context tidak ada |
| `413` | Ukuran request melebihi `MAX_REQUEST_BYTES` |
| `422` | Body JSON tidak sesuai schema |
| `503` | Qdrant, model, atau dependency wajib belum siap |
| `500` | Error internal tak terduga |

### Field `status` pada Response Identify

| `status` | Artinya |
| --- | --- |
| `"ok"` | Wajah cocok dengan embedding user |
| `"not_found"` | User belum enrollment, atau similarity di bawah threshold |

---

## Development

Development berjalan lewat Docker Compose override yang me-mount source code lokal ke container.

### Setup Development

```bash
# 1. Siapkan .env
cp .env.example .env
# Isi konfigurasi Astra service boundary, Qdrant, dan model

# 2. Letakkan model ONNX
mkdir -p models
# Taruh glintr100.onnx di ./models/

# 3. Pull image resmi dan jalankan dengan override dev
docker compose pull
docker compose -f compose.yaml -f docker-compose.dev.yml up
```

Mode development me-mount `./src` ke dalam container sehingga perubahan kode langsung terreflect tanpa rebuild image.

### Pakai Image Tertentu (bukan latest)

```ini
# Di .env
IMAGE_NAME=ghcr.io/geber-suprabapak/project-robin
IMAGE_TAG=cpu-<short-sha>
```

---

## Deployment

### CPU (Default)

```bash
docker compose pull
docker compose up -d
```

**Volume yang dipakai:**

| Host | Container | Keterangan |
| --- | --- | --- |
| `./models` | `/app/runtime-models` | Model ONNX dari host |
| `./logs` | `/app/logs` | Output log runtime |

Gunakan `IMAGE_TAG=cpu-<short-sha>` untuk deployment pinned pada commit tertentu, atau `cpu-latest` untuk selalu mengikuti image terbaru dari branch `master`.

### Pinned Deployment (Recommended untuk Production)

```ini
# .env
IMAGE_TAG=cpu-abc1234
```

```bash
docker compose pull
docker compose up -d
```

---

## GPU Runtime

Image CPU adalah default. Untuk akselerasi NVIDIA GPU:

### Prasyarat

- NVIDIA driver terinstall di host
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) (`nvidia-docker2` atau Docker GPU runtime)

### Menjalankan GPU Runtime

```bash
# 1. Set tag image GPU di .env
GPU_IMAGE_TAG=gpu-<short-sha>

# 2. Jalankan dengan override GPU
docker compose -f compose.yaml -f docker-compose.gpu.yml up -d
```

Override GPU otomatis mengubah image ke tag GPU dan set `GPU_DEVICE_ID=0`.

---

## CI/CD

GitHub Actions adalah jalur build dan publish image resmi.

| Workflow | Trigger | Fungsi |
| --- | --- | --- |
| **CI/CD CPU** | Push/PR ke `master` | Lint, test (coverage ≥70%), build & publish `cpu-latest` + `cpu-<sha>` |
| **CI/CD GPU** | Manual (workflow dispatch) | Build & publish `gpu-<sha>` |

**CI pipeline untuk CPU:**
1. `uv lock --check` — verifikasi lockfile tidak drift
2. Sync dependency CPU
3. `ruff check` — lint
4. `pytest` dengan coverage gate 70%
5. Build Docker image CPU
6. Push `cpu-latest` dan `cpu-<short-sha>` ke GHCR (hanya saat push, bukan PR)

> Repository ini memakai `uv.lock` sebagai satu-satunya lockfile. Jangan tambahkan `requirements.txt`, Makefile, atau script build/publish lokal.

---

## Quality Check Lokal

Jalankan quality gate yang sama dengan CI sebelum push:

```bash
# Verifikasi lockfile
uv lock --check

# Lint
uv run ruff check src tests

# Test dengan coverage
uv run pytest --cov=src --cov-report=term-missing --cov-fail-under=70
```

---

## Privacy and Security

Project Robin processes biometric data. Read these before deploying:

- [Privacy Policy](./PRIVACY.md)
- [Security Policy](./SECURITY.md)

Project Robin is intended for 1:1 attendance verification only. It should not be used for mass surveillance, covert recognition, public-space identification, or 1:N watchlist matching.

---

## Smoke Mode (CI Only)

`SKIP_MODEL_LOAD=true` memungkinkan API start tanpa file ONNX. Dipakai hanya untuk CI smoke test.

> ⚠️ **Jangan** aktifkan di runtime nyata. Dengan `SKIP_MODEL_LOAD=true`, endpoint `/ready` akan gagal dan semua endpoint inference tidak bisa beroperasi.
