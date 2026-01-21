# Migration Instructions

## Langkah-langkah Menjalankan Migration

### 1. Buka Supabase SQL Editor
- Login ke dashboard Supabase
- Pilih project Anda
- Klik "SQL Editor" di sidebar kiri

### 2. Copy-Paste SQL Migration
- Buka file: `sql/face_embeddings_user_id_schema.sql`
- Copy seluruh isinya
- Paste ke SQL Editor
- Klik "Run" atau tekan Ctrl+Enter

### 3. Verifikasi Migration Berhasil
Jalankan query berikut untuk memastikan struktur tabel sudah benar:

```sql
-- Check kolom yang ada di face_embeddings
SELECT column_name, data_type, is_nullable
FROM information_schema.columns
WHERE table_name = 'face_embeddings'
ORDER BY ordinal_position;
```

**Expected columns:**
- `id` (uuid)
- `user_id` (uuid, NOT NULL)
- `embedding` (vector)
- `image_index` (integer)
- `image_quality_score` (double precision)
- `captured_at` (timestamp with time zone)
- `created_at` (timestamp with time zone)
- `updated_at` (timestamp with time zone)

**Kolom yang TIDAK ada lagi:**
- ❌ `nis` (sudah dihapus)
- ❌ `camera_id` (sudah dihapus)

### 4. Test RPC Functions
```sql
-- Test get_student_by_user_id
SELECT * FROM get_student_by_user_id('your-user-uuid-here');

-- Test count_user_embeddings
SELECT count_user_embeddings('your-user-uuid-here');
```

### 5. Restart API Server
Setelah migration berhasil, restart server:
```powershell
# Di terminal project-robin
uv run .\main.py
```

## Troubleshooting

### Error: "column nis does not exist"
✅ **Fixed!** Migration sekarang sudah menambahkan kolom `image_index` terlebih dahulu sebelum menghapus `nis`.

### Error: "user_id cannot be null"
Pastikan semua user di `user_profiles` memiliki `user_id` yang valid. Jika ada data lama tanpa `user_id`, migration akan menghapusnya otomatis.

### Error: "function already exists"
Ini normal. Migration menggunakan `CREATE OR REPLACE FUNCTION` jadi akan overwrite function yang lama.
