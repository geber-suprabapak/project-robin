# Face Cropping & File Size Information

## ✅ Auto Face Cropping

Sistem sekarang **otomatis crop wajah** dari gambar!

### Proses:
1. **Detect face** menggunakan DNN (ResNet SSD)
2. **Crop face region** dengan margin 20%
3. **Resize** ke 112x112 untuk model ArcFace
4. **Extract embedding** 512-dim

### Keuntungan:
- ✅ Foto fullbody/landscape tetap bisa dipakai
- ✅ Background noise dihilangkan
- ✅ Fokus hanya ke wajah
- ✅ Akurasi lebih tinggi

---

## 📏 File Size Limits

### Default FastAPI Limits:
- **Max file size per upload**: ~16 MB (default)
- **Max total request size**: ~16 MB

### Rekomendasi File Size:
- **Optimal**: 500 KB - 2 MB per image
- **Max recommended**: 5 MB per image
- **Total untuk 15 images**: ~7.5 - 30 MB

### Image Specs:
- **Format**: JPG, PNG
- **Resolution**: 640x480 hingga 1920x1080
- **Quality**: Medium-High (80-90% JPEG quality)

---

## 🎯 Best Practices

### Untuk Enrollment:
1. **Gunakan foto dengan resolusi wajar** (tidak perlu 4K)
2. **Compress jika perlu** (gunakan 80-90% quality)
3. **Pastikan wajah terlihat jelas** (tidak blur)
4. **Tidak perlu crop manual** - sistem auto crop!

### Contoh Good Photos:
```
✅ Foto selfie HP (1-3 MB)
✅ Foto webcam (500 KB - 1 MB)
✅ Foto landscape dengan wajah jelas (2-5 MB)
✅ Foto fullbody asal wajah terlihat (2-5 MB)
```

### Contoh Bad Photos:
```
❌ Foto blur/gelap
❌ Foto terlalu jauh (wajah < 10% frame)
❌ Foto dengan multiple faces
❌ Foto dengan occlusion berat
```

---

## 🔧 Jika Perlu Ubah Limit

Edit `main.py` atau `config.py`:

```python
# Tambahkan di app initialization
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Increase max body size (default 16MB)
app.add_middleware(
    CORSMiddleware,
    max_age=3600,
)

# Atau via uvicorn
# uvicorn main:app --limit-max-requests 100 --timeout-keep-alive 30
```

Untuk production dengan Nginx:
```nginx
client_max_body_size 50M;  # Max 50MB total
```
