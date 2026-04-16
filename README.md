# Project Robin

Project Robin adalah API presensi berbasis face recognition yang dibangun untuk proses verifikasi wajah berkecepatan tinggi. API ini menerima enrollment wajah, memverifikasi wajah pengguna saat presensi, lalu menghubungkan hasilnya dengan Supabase dan Qdrant external.

Runtime utama memakai CPU image agar cocok untuk server on-prem, dengan opsi GPU manual untuk kebutuhan akselerasi. Repository ini mengikuti pendekatan CI/CD-first, Docker-first, dan uv-only.

## What It Does

- Memproses presensi face recognition dengan latency rendah.
- Mengelola enrollment wajah mandiri untuk setiap user.
- Melakukan verifikasi 1:1 terhadap embedding wajah user yang sedang login.
- Menyimpan dan membaca embedding dari Qdrant external.
- Mengambil profil siswa dan user dari Supabase.
- Menyediakan image Docker resmi dari GitHub Container Registry.

## Development

Development berjalan lewat Docker Compose. Compose tidak melakukan local build; image di-pull dari GitHub Container Registry dan source code lokal di-mount ke container.

1. Buat `.env` dari `.env.example`.
2. Isi konfigurasi model, Supabase, dan Qdrant external.
3. Letakkan model ONNX di `./models`.
4. Jalankan stack:

```bash
docker compose pull
docker compose up
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

## Runtime

Service utama Compose adalah `api` dan memakai CPU image secara default. Qdrant tidak dijalankan oleh Compose; arahkan `QDRANT_HOST`, `QDRANT_PORT`, `QDRANT_HTTPS`, dan `QDRANT_API_KEY` ke instance external.

Endpoint:

| Method | Path | Description |
| --- | --- | --- |
| GET | `/` | API metadata |
| GET | `/health` | Health check |
| POST | `/v1/identify` | Verifikasi wajah untuk presensi |
| POST | `/v1/enroll` | Enrollment wajah user |
| GET | `/v1/enroll/status` | Status enrollment user |

## CI/CD

GitHub Actions adalah satu-satunya jalur build dan publish image resmi. CPU dan GPU dipisah menjadi dua workflow supaya build CPU tetap cepat di hosted runner.

- `CI/CD CPU`: berjalan otomatis untuk pull request dan push ke `master`; cek `uv.lock`, sync CPU dependencies, lint, test, build image CPU, lalu publish `cpu-latest` dan `cpu-<short-sha>` saat push.
- `CI/CD GPU`: berjalan manual via workflow dispatch; build dan publish image `gpu-<short-sha>` di hosted runner.

CI memakai `uv.lock` sebagai satu-satunya dependency lock source. Jangan tambahkan `requirements.txt`, local setup script, local run script, atau Makefile untuk build/publish.

Untuk runtime GPU manual, set `GPU_IMAGE_TAG=gpu-<short-sha>` lalu jalankan Compose dengan override `docker-compose.gpu.yml`.

## Smoke Mode

`SKIP_MODEL_LOAD=true` hanya untuk CI smoke test agar aplikasi bisa start tanpa file ONNX. Jangan aktifkan ini untuk runtime nyata karena endpoint inference membutuhkan model yang sudah diload.

## Deployment

Server on-prem CPU-only cukup pull image resmi dan menjalankan Compose dengan `.env` production.

```bash
docker compose pull
docker compose up -d
```

Gunakan `IMAGE_TAG=cpu-<short-sha>` untuk deployment yang pinned, atau `IMAGE_TAG=cpu-latest` untuk mengikuti release terbaru dari branch `master`.
