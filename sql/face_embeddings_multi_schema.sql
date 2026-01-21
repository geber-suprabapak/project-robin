-- ============================================================================
-- Multi-Image Face Embeddings Schema Update
-- ============================================================================
-- This migration allows storing multiple embeddings per student for improved
-- recognition accuracy. Each student can have 10-20 face images enrolled.
--
-- Run this AFTER the initial face_embeddings_schema.sql
-- ============================================================================

-- ============================================================================
-- Step 1: Drop the unique constraint that limits one embedding per nis+camera
-- ============================================================================
DROP INDEX IF EXISTS face_embeddings_nis_camera_unique;

-- ============================================================================
-- Step 2: Add new index for better query performance
-- ============================================================================
-- Index on (nis, created_at) for ordered retrieval of embeddings per student
CREATE INDEX IF NOT EXISTS idx_face_embeddings_nis_created 
    ON face_embeddings(nis, created_at DESC);

-- ============================================================================
-- Step 3: Add image_index column to track which image number this is (1-20)
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
-- Step 4: New unique constraint - one embedding per nis + image_index
-- ============================================================================
CREATE UNIQUE INDEX IF NOT EXISTS face_embeddings_nis_image_idx_unique 
    ON face_embeddings(nis, image_index);

-- ============================================================================
-- Step 5: Update find_face_match to return best match across ALL embeddings
-- ============================================================================
CREATE OR REPLACE FUNCTION find_face_match(
    query_embedding vector(512),
    match_threshold FLOAT DEFAULT 0.6,
    match_count INT DEFAULT 1
)
RETURNS TABLE (
    nis TEXT,
    nama TEXT,
    kelas TEXT,
    user_id UUID,
    confidence FLOAT,
    distance FLOAT
)
LANGUAGE plpgsql
SECURITY DEFINER SET search_path = public
AS $$
BEGIN
    -- Return distinct students with their BEST matching embedding
    -- This ensures multi-image enrollment provides better matching
    RETURN QUERY
    WITH ranked_matches AS (
        SELECT 
            fe.nis,
            bs.nama,
            bs.kelas,
            fe.user_id,
            (1.0 - (fe.embedding <=> query_embedding))::FLOAT as confidence,
            (fe.embedding <=> query_embedding)::FLOAT as distance,
            ROW_NUMBER() OVER (PARTITION BY fe.nis ORDER BY fe.embedding <=> query_embedding ASC) as rn
        FROM face_embeddings fe
        INNER JOIN biodata_siswa bs ON fe.nis = bs.nis::TEXT
        WHERE (1.0 - (fe.embedding <=> query_embedding)) >= match_threshold
    )
    SELECT 
        rm.nis,
        rm.nama,
        rm.kelas,
        rm.user_id,
        rm.confidence,
        rm.distance
    FROM ranked_matches rm
    WHERE rm.rn = 1  -- Only keep the best match per student
    ORDER BY rm.distance ASC
    LIMIT match_count;
END;
$$;

-- ============================================================================
-- Step 6: New function to insert multiple embeddings for enrollment
-- ============================================================================
CREATE OR REPLACE FUNCTION insert_face_embedding(
    p_nis TEXT,
    p_user_id UUID,
    p_embedding vector(512),
    p_image_index INT,
    p_camera_id TEXT DEFAULT 'enrollment',
    p_quality_score FLOAT DEFAULT NULL
)
RETURNS UUID
LANGUAGE plpgsql
SECURITY DEFINER SET search_path = public
AS $$
DECLARE
    v_id UUID;
BEGIN
    -- Insert or update embedding at specific image_index
    INSERT INTO face_embeddings (
        nis, 
        user_id, 
        embedding, 
        image_index,
        camera_id, 
        image_quality_score
    ) VALUES (
        p_nis,
        p_user_id,
        p_embedding,
        p_image_index,
        p_camera_id,
        p_quality_score
    )
    ON CONFLICT (nis, image_index) 
    DO UPDATE SET
        embedding = EXCLUDED.embedding,
        user_id = EXCLUDED.user_id,
        camera_id = EXCLUDED.camera_id,
        image_quality_score = EXCLUDED.image_quality_score,
        captured_at = NOW(),
        updated_at = NOW()
    RETURNING id INTO v_id;
    
    RETURN v_id;
END;
$$;

-- Grant execute permission
GRANT EXECUTE ON FUNCTION insert_face_embedding(TEXT, UUID, vector, INT, TEXT, FLOAT) TO service_role;
GRANT EXECUTE ON FUNCTION insert_face_embedding(TEXT, UUID, vector, INT, TEXT, FLOAT) TO authenticated;

-- ============================================================================
-- Step 7: Function to delete all embeddings for a student (for re-enrollment)
-- ============================================================================
CREATE OR REPLACE FUNCTION delete_student_embeddings(p_nis TEXT)
RETURNS INT
LANGUAGE plpgsql
SECURITY DEFINER SET search_path = public
AS $$
DECLARE
    deleted_count INT;
BEGIN
    DELETE FROM face_embeddings WHERE nis = p_nis;
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$;

-- Grant execute permission
GRANT EXECUTE ON FUNCTION delete_student_embeddings(TEXT) TO service_role;
GRANT EXECUTE ON FUNCTION delete_student_embeddings(TEXT) TO authenticated;

-- ============================================================================
-- Step 8: Function to count embeddings for a student
-- ============================================================================
CREATE OR REPLACE FUNCTION count_student_embeddings(p_nis TEXT)
RETURNS INT
LANGUAGE plpgsql
SECURITY DEFINER SET search_path = public
AS $$
DECLARE
    embedding_count INT;
BEGIN
    SELECT COUNT(*) INTO embedding_count FROM face_embeddings WHERE nis = p_nis;
    RETURN embedding_count;
END;
$$;

-- Grant execute permission
GRANT EXECUTE ON FUNCTION count_student_embeddings(TEXT) TO service_role;
GRANT EXECUTE ON FUNCTION count_student_embeddings(TEXT) TO authenticated;

-- ============================================================================
-- Comments
-- ============================================================================
COMMENT ON COLUMN face_embeddings.image_index IS 'Index of the enrolled image (1-20) for multi-image enrollment';
COMMENT ON FUNCTION insert_face_embedding(TEXT, UUID, vector, INT, TEXT, FLOAT) IS 'Insert single face embedding with image index';
COMMENT ON FUNCTION delete_student_embeddings(TEXT) IS 'Delete all embeddings for a student';
COMMENT ON FUNCTION count_student_embeddings(TEXT) IS 'Count total embeddings for a student';
