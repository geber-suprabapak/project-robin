# Project Robin

Project Robin adalah API backend untuk presensi berbasis face recognition. Aplikasi ini menerima foto wajah dari client, memverifikasi identitas user lewat token Supabase, membuat embedding wajah dengan model ONNX, lalu membandingkannya dengan embedding yang tersimpan di Qdrant.

Target utama aplikasi ini adalah runtime on-prem atau server internal sekolah/lembaga yang butuh verifikasi wajah 1:1 untuk presensi. Runtime default memakai image CPU agar mudah dijalankan di server biasa, dengan opsi image GPU untuk deployment yang butuh akselerasi.

## Ringkasan Fitur

- Verifikasi presensi wajah 1:1 untuk user yang sedang login.
- Enrollment wajah mandiri untuk menyimpan embedding user ke Qdrant.
- Status enrollment untuk mengecek apakah user sudah punya embedding.
- Validasi JWT Supabase di setiap endpoint fitur utama.
- Integrasi Supabase untuk validasi user dan data siswa.
- Integrasi Qdrant external untuk penyimpanan dan pencarian embedding.
- Readiness check yang membedakan proses hidup dan dependency siap.
- Docker image resmi dari GitHub Container Registry.

## Cara Kerja

Alur presensi memakai endpoint `POST /v1/identify`.

1. Client mengirim `Authorization: Bearer <supabase-jwt>` dan `image_base64`.
2. API memverifikasi JWT memakai `SUPABASE_JWT_SECRET`.
3. API memastikan `sub` dari JWT ada di tabel `user_profiles` Supabase.
4. Base64 image didecode, divalidasi secara minimal, lalu dicek harus berisi tepat satu wajah.
5. Face crop dipreprocess ke ukuran input model.
6. Model ONNX membuat embedding wajah.
7. Qdrant mengambil embedding milik user yang sama.
8. API menghitung cosine similarity 1:1.
9. Jika match, API mengambil data siswa dari Supabase dan mengembalikan hasil verifikasi.

Alur enrollment memakai endpoint `POST /v1/enroll`.

1. Client mengirim bearer token Supabase dan tepat 10 foto wajah.
2. API memverifikasi user dari Supabase.
3. Setiap foto dicek harus valid dan berisi tepat satu wajah.
4. Setiap foto diproses menjadi embedding.
5. Embedding lama user di Qdrant diganti dengan embedding baru.
6. API mengembalikan jumlah image yang berhasil diproses dan total embedding tersimpan.

## Arsitektur

Project Robin terdiri dari satu service API FastAPI.

Komponen runtime:

- `FastAPI`: HTTP API, routing, middleware, auth dependency, dan response handling.
- `ONNX Runtime`: menjalankan model face recognition dari file ONNX.
- `OpenCV DNN`: face detector untuk memastikan foto berisi satu wajah.
- `Supabase`: sumber data user profile, student profile, dan JWT secret.
- `Qdrant`: vector database untuk embedding wajah.

Qdrant tidak dijalankan oleh Docker Compose repository ini. Gunakan Qdrant external, baik self-hosted maupun Qdrant Cloud.

## Endpoint

| Method | Path | Auth | Fungsi |
| --- | --- | --- | --- |
| `GET` | `/` | Tidak | Metadata API |
| `GET` | `/live` | Tidak | Liveness process |
| `GET` | `/ready` | Tidak | Readiness dependency |
| `GET` | `/health` | Tidak | Alias readiness untuk backward compatibility |
| `GET` | `/v1/enroll/status` | Bearer JWT | Status enrollment user |
| `POST` | `/v1/enroll` | Bearer JWT | Enrollment wajah user |
| `POST` | `/v1/identify` | Bearer JWT | Verifikasi wajah untuk presensi |

### `POST /v1/identify`

Request:

```http
Authorization: Bearer <supabase-jwt>
Content-Type: application/json
```

```json
{
  "image_base64": "<base64-image>"
}
```

Response saat wajah cocok:

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

Response saat user belum enrollment:

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

### `POST /v1/enroll`

Request:

```http
Authorization: Bearer <supabase-jwt>
Content-Type: multipart/form-data
```

Field:

| Field | Tipe | Keterangan |
| --- | --- | --- |
| `files` | file[] | Tepat 10 foto wajah |

Response:

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

### `GET /v1/enroll/status`

Request:

```http
Authorization: Bearer <supabase-jwt>
```

Response:

```json
{
  "is_enrolled": true,
  "embedding_count": 10,
  "user_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

## Health Check

`/live` hanya menandakan proses API hidup. Endpoint ini tidak mengecek model, Supabase, atau Qdrant.

`/ready` dan `/health` mengecek apakah service siap melayani request presensi nyata. Readiness hanya `200` jika semua komponen wajib siap:

- model ONNX sudah loaded,
- face detector asset tersedia dan bisa diload,
- Supabase reachable,
- Qdrant reachable.

Jika salah satu dependency belum siap, readiness mengembalikan `503`. Kondisi Qdrant atau Supabase down tidak disamarkan menjadi user belum enroll atau user tidak ditemukan.

Contoh readiness gagal:

```json
{
  "status": "unhealthy",
  "model_loaded": true,
  "face_detector_ready": true,
  "gpu_available": false,
  "supabase_connected": true,
  "qdrant_connected": false
}
```

## Error Semantics

Endpoint fitur utama membedakan error bisnis dan error dependency.

| Status | Arti |
| --- | --- |
| `200` | Request valid dan berhasil diproses |
| `400` | Input image tidak valid, jumlah file enrollment salah, atau wajah tidak valid |
| `401` | Bearer token tidak ada, invalid, expired, atau user tidak ditemukan |
| `413` | Request terlalu besar berdasarkan `Content-Length` |
| `422` | JSON request tidak sesuai schema |
| `503` | Supabase, Qdrant, model, atau dependency wajib belum siap |
| `500` | Error internal tak terduga |

Untuk presensi, `status: "not_found"` hanya dipakai jika request berhasil diproses tetapi wajah tidak cocok atau user memang belum punya embedding. Jika Qdrant error, API mengembalikan `503`.

## Konfigurasi

Buat `.env` dari `.env.example`, lalu isi nilai yang sesuai deployment.

Konfigurasi utama:

| Env | Keterangan |
| --- | --- |
| `IMAGE_NAME` | Nama image Docker |
| `IMAGE_TAG` | Tag image CPU yang dipakai Compose |
| `MODEL_PATH` | Path model ONNX di dalam container |
| `SKIP_MODEL_LOAD` | `true` hanya untuk CI smoke test |
| `GPU_DEVICE_ID` | `-1` untuk CPU, `0` untuk GPU pertama |
| `SUPABASE_URL` | URL project Supabase |
| `SUPABASE_KEY` | Supabase anon key |
| `SUPABASE_SERVICE_ROLE_KEY` | Service role key untuk lookup server-side |
| `SUPABASE_JWT_SECRET` | Secret untuk verifikasi JWT client |
| `QDRANT_HOST` | Host atau URL Qdrant |
| `QDRANT_PORT` | Port REST Qdrant jika memakai host biasa |
| `QDRANT_HTTPS` | Gunakan HTTPS untuk Qdrant |
| `QDRANT_API_KEY` | API key Qdrant Cloud jika ada |
| `QDRANT_COLLECTION_NAME` | Nama collection embedding |
| `FACE_MATCH_THRESHOLD` | Threshold minimum similarity |

Guardrail request:

| Env | Default | Keterangan |
| --- | --- | --- |
| `MAX_REQUEST_BYTES` | `62914560` | Maksimum ukuran HTTP request |
| `MAX_IMAGE_BYTES` | `5242880` | Maksimum ukuran encoded image setelah decode/read |
| `MAX_IMAGE_WIDTH` | `4096` | Lebar maksimum image |
| `MAX_IMAGE_HEIGHT` | `4096` | Tinggi maksimum image |
| `MAX_IMAGE_PIXELS` | `16777216` | Total pixel maksimum |

Validasi size dan konversi image utama tetap sebaiknya dilakukan di client. Guardrail server ini hanya lapisan pengaman tambahan jika request dikirim langsung atau client bug.

## Model dan Asset

Model face recognition ONNX tidak dibake ke image Docker. Letakkan model di host directory:

```text
./models/arcface_r100_224x224.onnx
```

Compose production me-mount directory tersebut ke:

```text
/app/runtime-models
```

Default `MODEL_PATH`:

```text
/app/runtime-models/arcface_r100_224x224.onnx
```

Face detector OpenCV DNN dibake saat Docker build dan diverifikasi dengan checksum. Runtime tidak melakukan download model detector dari internet.

## Development

Development berjalan lewat Docker Compose override. Compose tetap memakai image resmi dari GHCR, lalu source code lokal di-mount hanya saat override development dipakai.

1. Buat `.env` dari `.env.example`.
2. Isi konfigurasi model, Supabase, dan Qdrant external.
3. Letakkan model ONNX di `./models`.
4. Jalankan:

```bash
docker compose pull
docker compose -f docker-compose.yml -f docker-compose.dev.yml up
```

Default image:

```text
ghcr.io/lunaradevs/project-robin:cpu-latest
```

Untuk memakai image tertentu:

```ini
IMAGE_NAME=ghcr.io/lunaradevs/project-robin
IMAGE_TAG=cpu-<short-sha>
```

## Deployment

Server on-prem CPU-only cukup pull image resmi dan menjalankan Compose dengan `.env` production.

```bash
docker compose pull
docker compose up -d
```

Compose production tidak me-mount source code. Volume yang dipakai:

| Host | Container | Keterangan |
| --- | --- | --- |
| `./models` | `/app/runtime-models` | Model ONNX dari host |
| `./logs` | `/app/logs` | Output log runtime |

Gunakan `IMAGE_TAG=cpu-<short-sha>` untuk deployment pinned, atau `IMAGE_TAG=cpu-latest` untuk mengikuti image terbaru dari branch `master`.

## GPU Runtime

Image CPU adalah default. Untuk runtime GPU manual:

1. Pastikan host punya NVIDIA driver dan runtime Docker yang mendukung GPU.
2. Set tag image GPU:

```ini
GPU_IMAGE_TAG=gpu-<short-sha>
```

3. Jalankan Compose dengan override GPU:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
```

Override GPU mengubah image ke tag GPU dan mengatur `GPU_DEVICE_ID=0`.

## CI/CD

GitHub Actions adalah jalur build dan publish image resmi.

- `CI/CD CPU`: berjalan otomatis untuk pull request dan push ke `master`; menjalankan `uv lock --check`, sync dependency CPU, lint, test dengan coverage gate, build image CPU, lalu publish `cpu-latest` dan `cpu-<short-sha>` saat push.
- `CI/CD GPU`: berjalan manual via workflow dispatch; build dan publish image `gpu-<short-sha>`.

Repository ini memakai `uv.lock` sebagai satu-satunya lockfile dependency. Jangan tambahkan `requirements.txt`, script setup lokal, script run lokal, atau Makefile untuk build/publish.

## Smoke Mode

`SKIP_MODEL_LOAD=true` hanya untuk CI smoke test agar aplikasi bisa start tanpa file ONNX. Jangan aktifkan ini untuk runtime nyata karena `/ready` akan gagal dan endpoint inference membutuhkan model yang sudah diload.

## Local Quality Check

Jalankan quality gate yang sama dengan CI:

```bash
uv lock --check
uv run ruff check src tests
uv run pytest --cov=src --cov-report=term-missing --cov-fail-under=70
```
