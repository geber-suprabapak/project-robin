-- ============================================================================
-- Multi-Image Face Embeddings Schema - User ID Based Strategy
-- ============================================================================
-- Migration untuk menggunakan user_id sebagai foreign key utama
-- Menghapus ketergantungan pada kolom nis
--
-- PENTING: Jalankan migration ini SETELAH face_embeddings_schema.sql
-- dan face_embeddings_multi_schema.sql
-- ============================================================================

-- ============================================================================
-- Step 1: Add image_index column if not exists
-- ============================================================================
ALTER TABLE face_embeddings 
ADD COLUMN IF NOT EXISTS image_index INT DEFAULT 1;

-- Add constraint to limit image_index between 1 and 20
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.constraint_column_usage 
        WHERE constraint_name = 'face_embeddings_image_index_check'
    ) THEN
        ALTER TABLE face_embeddings 
        ADD CONSTRAINT face_embeddings_image_index_check 
        CHECK (image_index >= 1 AND image_index <= 20);
    END IF;
END $$;

-- ============================================================================
-- Step 2: Drop existing constraints dan indexes yang menggunakan nis
-- ============================================================================
DROP INDEX IF EXISTS face_embeddings_nis_image_idx_unique;
DROP INDEX IF EXISTS face_embeddings_nis_camera_unique;
DROP INDEX IF EXISTS idx_face_embeddings_nis;
DROP INDEX IF EXISTS idx_face_embeddings_nis_created;

-- ============================================================================
-- Step 3: Hapus kolom nis dari tabel
-- ============================================================================
ALTER TABLE face_embeddings DROP COLUMN IF EXISTS nis;
ALTER TABLE face_embeddings DROP COLUMN IF EXISTS camera_id;

-- ============================================================================
-- Step 4: Buat user_id NOT NULL (wajib ada)
-- ============================================================================
-- Hapus data lama yang tidak punya user_id
DELETE FROM face_embeddings WHERE user_id IS NULL;

-- Ubah user_id menjadi NOT NULL
ALTER TABLE face_embeddings ALTER COLUMN user_id SET NOT NULL;

-- ============================================================================
-- Step 5: Index baru berdasarkan user_id
-- ============================================================================
-- Index untuk lookup by user_id
CREATE INDEX IF NOT EXISTS idx_face_embeddings_user_id ON face_embeddings(user_id);

-- Index untuk ordered retrieval per user
CREATE INDEX IF NOT EXISTS idx_face_embeddings_user_created 
    ON face_embeddings(user_id, created_at DESC);

-- Unique constraint: satu embedding per user + image_index
CREATE UNIQUE INDEX IF NOT EXISTS face_embeddings_user_image_idx_unique 
    ON face_embeddings(user_id, image_index);

-- ============================================================================
-- Step 6: Update find_face_match - return user_id only, backend lookup
-- ============================================================================
CREATE OR REPLACE FUNCTION find_face_match(
    query_embedding vector(512),
    match_threshold FLOAT DEFAULT 0.6,
    match_count INT DEFAULT 1
)
RETURNS TABLE (
    user_id UUID,
    confidence FLOAT,
    distance FLOAT
)
LANGUAGE plpgsql
SECURITY DEFINER SET search_path = public
AS $$
BEGIN
    -- Return distinct users with their BEST matching embedding
    RETURN QUERY
    WITH ranked_matches AS (
        SELECT 
            fe.user_id,
            (1.0 - (fe.embedding <=> query_embedding))::FLOAT as confidence,
            (fe.embedding <=> query_embedding)::FLOAT as distance,
            ROW_NUMBER() OVER (PARTITION BY fe.user_id ORDER BY fe.embedding <=> query_embedding ASC) as rn
        FROM face_embeddings fe
        WHERE (1.0 - (fe.embedding <=> query_embedding)) >= match_threshold
    )
    SELECT 
        rm.user_id,
        rm.confidence,
        rm.distance
    FROM ranked_matches rm
    WHERE rm.rn = 1  -- Only keep the best match per user
    ORDER BY rm.distance ASC
    LIMIT match_count;
END;
$$;

-- Re-grant execute permission
GRANT EXECUTE ON FUNCTION find_face_match(vector, FLOAT, INT) TO service_role;
GRANT EXECUTE ON FUNCTION find_face_match(vector, FLOAT, INT) TO authenticated;

-- ============================================================================
-- Step 7: Update insert_face_embedding - use user_id only
-- ============================================================================
CREATE OR REPLACE FUNCTION insert_face_embedding(
    p_user_id UUID,
    p_embedding vector(512),
    p_image_index INT,
    p_quality_score FLOAT DEFAULT NULL
)
RETURNS UUID
LANGUAGE plpgsql
SECURITY DEFINER SET search_path = public
AS $$
DECLARE
    v_id UUID;
BEGIN
    -- Validate image_index
    IF p_image_index < 1 OR p_image_index > 20 THEN
        RAISE EXCEPTION 'image_index must be between 1 and 20';
    END IF;
    
    -- Insert or update embedding at specific image_index
    INSERT INTO face_embeddings (
        user_id, 
        embedding, 
        image_index,
        image_quality_score
    ) VALUES (
        p_user_id,
        p_embedding,
        p_image_index,
        p_quality_score
    )
    ON CONFLICT (user_id, image_index) 
    DO UPDATE SET
        embedding = EXCLUDED.embedding,
        image_quality_score = EXCLUDED.image_quality_score,
        captured_at = NOW(),
        updated_at = NOW()
    RETURNING id INTO v_id;
    
    RETURN v_id;
END;
$$;

-- Grant execute permission
GRANT EXECUTE ON FUNCTION insert_face_embedding(UUID, vector, INT, FLOAT) TO service_role;
GRANT EXECUTE ON FUNCTION insert_face_embedding(UUID, vector, INT, FLOAT) TO authenticated;

-- ============================================================================
-- Step 8: Update delete function - use user_id
-- ============================================================================
CREATE OR REPLACE FUNCTION delete_user_embeddings(p_user_id UUID)
RETURNS INT
LANGUAGE plpgsql
SECURITY DEFINER SET search_path = public
AS $$
DECLARE
    deleted_count INT;
BEGIN
    DELETE FROM face_embeddings WHERE user_id = p_user_id;
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$;

-- Grant execute permission
GRANT EXECUTE ON FUNCTION delete_user_embeddings(UUID) TO service_role;
GRANT EXECUTE ON FUNCTION delete_user_embeddings(UUID) TO authenticated;

-- ============================================================================
-- Step 9: Update count function - use user_id
-- ============================================================================
CREATE OR REPLACE FUNCTION count_user_embeddings(p_user_id UUID)
RETURNS INT
LANGUAGE plpgsql
SECURITY DEFINER SET search_path = public
AS $$
DECLARE
    embedding_count INT;
BEGIN
    SELECT COUNT(*) INTO embedding_count FROM face_embeddings WHERE user_id = p_user_id;
    RETURN embedding_count;
END;
$$;

-- Grant execute permission
GRANT EXECUTE ON FUNCTION count_user_embeddings(UUID) TO service_role;
GRANT EXECUTE ON FUNCTION count_user_embeddings(UUID) TO authenticated;

-- ============================================================================
-- Step 10: Function untuk lookup student info by user_id
-- ============================================================================
CREATE OR REPLACE FUNCTION get_student_by_user_id(p_user_id UUID)
RETURNS TABLE (
    nis TEXT,
    nama TEXT,
    kelas TEXT,
    absen INT
)
LANGUAGE plpgsql
SECURITY DEFINER SET search_path = public
AS $$
BEGIN
    RETURN QUERY
    SELECT 
        up.nis,
        bs.nama,
        bs.kelas,
        bs.absen
    FROM user_profiles up
    INNER JOIN biodata_siswa bs ON up.nis = bs.nis::TEXT
    WHERE up.user_id = p_user_id
    LIMIT 1;
END;
$$;

-- Grant execute permission
GRANT EXECUTE ON FUNCTION get_student_by_user_id(UUID) TO service_role;
GRANT EXECUTE ON FUNCTION get_student_by_user_id(UUID) TO authenticated;

-- ============================================================================
-- Drop old functions yang sudah tidak dipakai
-- ============================================================================
DROP FUNCTION IF EXISTS delete_student_embeddings(TEXT);
DROP FUNCTION IF EXISTS count_student_embeddings(TEXT);
DROP FUNCTION IF EXISTS insert_face_embedding(TEXT, UUID, vector, INT, TEXT, FLOAT);
DROP FUNCTION IF EXISTS upsert_face_embedding(TEXT, UUID, vector, TEXT, FLOAT);

-- ============================================================================
-- Step 11: 1:1 Face Verification (verify against specific user only)
-- ============================================================================
CREATE OR REPLACE FUNCTION verify_face_1to1(
    p_user_id UUID,
    query_embedding vector(512),
    match_threshold FLOAT DEFAULT 0.6
)
RETURNS TABLE (
    verified BOOLEAN,
    confidence FLOAT,
    distance FLOAT,
    best_match_index INT
)
LANGUAGE plpgsql
SECURITY DEFINER SET search_path = public
AS $$
BEGIN
    -- Compare query embedding against user's enrolled embeddings only
    -- Returns the best matching result if above threshold
    RETURN QUERY
    SELECT 
        (1.0 - (fe.embedding <=> query_embedding))::FLOAT >= match_threshold AS verified,
        (1.0 - (fe.embedding <=> query_embedding))::FLOAT AS confidence,
        (fe.embedding <=> query_embedding)::FLOAT AS distance,
        fe.image_index AS best_match_index
    FROM face_embeddings fe
    WHERE fe.user_id = p_user_id
    ORDER BY fe.embedding <=> query_embedding ASC
    LIMIT 1;
END;
$$;

-- Grant execute permission
GRANT EXECUTE ON FUNCTION verify_face_1to1(UUID, vector, FLOAT) TO service_role;
GRANT EXECUTE ON FUNCTION verify_face_1to1(UUID, vector, FLOAT) TO authenticated;

-- ============================================================================
-- Comments
-- ============================================================================
COMMENT ON FUNCTION find_face_match(vector, FLOAT, INT) IS 'Find matching faces, returns user_id only for server-side lookup';
COMMENT ON FUNCTION insert_face_embedding(UUID, vector, INT, FLOAT) IS 'Insert face embedding for a user';
COMMENT ON FUNCTION delete_user_embeddings(UUID) IS 'Delete all embeddings for a user';
COMMENT ON FUNCTION count_user_embeddings(UUID) IS 'Count total embeddings for a user';
COMMENT ON FUNCTION get_student_by_user_id(UUID) IS 'Lookup student info by user_id from user_profiles and biodata_siswa';
COMMENT ON FUNCTION verify_face_1to1(UUID, vector, FLOAT) IS '1:1 face verification - compare embedding against specific user embeddings only';

