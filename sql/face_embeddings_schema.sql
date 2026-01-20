-- ============================================================================
-- Face Embeddings Table Extension for Face Recognition API
-- ============================================================================
-- This schema extends the existing Project Chronos database to support
-- face recognition functionality by storing student face embeddings.
--
-- Prerequisites:
-- 1. Install pgvector extension: CREATE EXTENSION IF NOT EXISTS vector;
-- 2. Run the main schema_latest_latest.sql first
-- ============================================================================

-- Enable pgvector extension for vector similarity search
CREATE EXTENSION IF NOT EXISTS vector;

-- ============================================================================
-- Table: face_embeddings
-- Description: Stores face embedding vectors for students
-- ============================================================================
CREATE TABLE IF NOT EXISTS face_embeddings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
    nis TEXT NOT NULL,
    user_id UUID,
    
    -- Face embedding vector (512-dimensional for ArcFace)
    embedding vector(512) NOT NULL,
    
    -- Metadata
    camera_id TEXT,
    captured_at TIMESTAMPTZ DEFAULT NOW() NOT NULL,
    image_quality_score FLOAT,
    
    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW() NOT NULL,
    updated_at TIMESTAMPTZ DEFAULT NOW() NOT NULL,
    
    -- Foreign keys
    CONSTRAINT fk_face_embeddings_user_id FOREIGN KEY (user_id) 
        REFERENCES auth.users(id) ON DELETE CASCADE,
    CONSTRAINT fk_face_embeddings_nis FOREIGN KEY (nis) 
        REFERENCES biodata_siswa(nis) ON DELETE CASCADE
);

-- ============================================================================
-- Indexes for Performance
-- ============================================================================

-- Index on NIS for fast lookup
CREATE INDEX IF NOT EXISTS idx_face_embeddings_nis ON face_embeddings(nis);

-- Index on user_id
CREATE INDEX IF NOT EXISTS idx_face_embeddings_user_id ON face_embeddings(user_id);

-- HNSW index for fast vector similarity search using cosine distance
-- This enables efficient nearest neighbor search on embeddings
CREATE INDEX IF NOT EXISTS idx_face_embeddings_vector_cosine 
    ON face_embeddings 
    USING hnsw (embedding vector_cosine_ops);

-- Alternative: IVFFlat index (faster build time, slightly slower query)
-- Uncomment if HNSW is too slow to build:
-- CREATE INDEX IF NOT EXISTS idx_face_embeddings_vector_ivfflat 
--     ON face_embeddings 
--     USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- ============================================================================
-- Unique Constraint: One embedding per student per camera
-- ============================================================================
CREATE UNIQUE INDEX IF NOT EXISTS face_embeddings_nis_camera_unique 
    ON face_embeddings(nis, camera_id);

-- ============================================================================
-- Trigger for auto-updating updated_at timestamp
-- ============================================================================
DROP TRIGGER IF EXISTS update_face_embeddings_updated_at ON face_embeddings;
CREATE TRIGGER update_face_embeddings_updated_at
    BEFORE UPDATE ON face_embeddings
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ============================================================================
-- Row Level Security (RLS)
-- ============================================================================
ALTER TABLE face_embeddings ENABLE ROW LEVEL SECURITY;

-- Service role can manage all embeddings (for API)
DO $$ 
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_policies 
    WHERE schemaname='public' 
      AND tablename='face_embeddings' 
      AND policyname='Service role can manage embeddings'
  ) THEN
    CREATE POLICY "Service role can manage embeddings" ON face_embeddings 
      FOR ALL 
      USING (auth.jwt() ->> 'role' = 'service_role');
  END IF;
END $$;

-- Users can view their own embeddings
DO $$ 
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_policies 
    WHERE schemaname='public' 
      AND tablename='face_embeddings' 
      AND policyname='Users can view own embeddings'
  ) THEN
    CREATE POLICY "Users can view own embeddings" ON face_embeddings 
      FOR SELECT 
      USING (auth.uid() = user_id);
  END IF;
END $$;

-- ============================================================================
-- RPC Functions for Face Matching
-- ============================================================================

-- Function to find nearest face match using cosine similarity
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
    RETURN QUERY
    SELECT 
        fe.nis,
        bs.nama,
        bs.kelas,
        fe.user_id,
        -- Convert cosine distance to confidence (similarity)
        (1.0 - (fe.embedding <=> query_embedding))::FLOAT as confidence,
        (fe.embedding <=> query_embedding)::FLOAT as distance
    FROM face_embeddings fe
    INNER JOIN biodata_siswa bs ON fe.nis = bs.nis::TEXT
    WHERE (1.0 - (fe.embedding <=> query_embedding)) >= match_threshold
    ORDER BY fe.embedding <=> query_embedding ASC
    LIMIT match_count;
END;
$$;

-- Grant execute permission to service role
GRANT EXECUTE ON FUNCTION find_face_match(vector, FLOAT, INT) TO service_role;
GRANT EXECUTE ON FUNCTION find_face_match(vector, FLOAT, INT) TO authenticated;

-- Function to store or update face embedding
CREATE OR REPLACE FUNCTION upsert_face_embedding(
    p_nis TEXT,
    p_user_id UUID,
    p_embedding vector(512),
    p_camera_id TEXT DEFAULT NULL,
    p_quality_score FLOAT DEFAULT NULL
)
RETURNS UUID
LANGUAGE plpgsql
SECURITY DEFINER SET search_path = public
AS $$
DECLARE
    v_id UUID;
BEGIN
    -- Insert or update embedding
    INSERT INTO face_embeddings (
        nis, 
        user_id, 
        embedding, 
        camera_id, 
        image_quality_score
    ) VALUES (
        p_nis,
        p_user_id,
        p_embedding,
        p_camera_id,
        p_quality_score
    )
    ON CONFLICT (nis, camera_id) 
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
GRANT EXECUTE ON FUNCTION upsert_face_embedding(TEXT, UUID, vector, TEXT, FLOAT) TO service_role;
GRANT EXECUTE ON FUNCTION upsert_face_embedding(TEXT, UUID, vector, TEXT, FLOAT) TO authenticated;

-- ============================================================================
-- Grants and Permissions
-- ============================================================================
GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE face_embeddings TO authenticated, service_role;

-- ============================================================================
-- Comments for documentation
-- ============================================================================
COMMENT ON TABLE face_embeddings IS 'Face embedding vectors for student identification';
COMMENT ON COLUMN face_embeddings.embedding IS '512-dimensional face embedding vector from ArcFace model';
COMMENT ON COLUMN face_embeddings.camera_id IS 'Identifier of camera/kiosk where face was captured';
COMMENT ON COLUMN face_embeddings.image_quality_score IS 'Optional quality score of captured face image';

COMMENT ON FUNCTION find_face_match(vector, FLOAT, INT) IS 'Find nearest matching faces using cosine similarity';
COMMENT ON FUNCTION upsert_face_embedding(TEXT, UUID, vector, TEXT, FLOAT) IS 'Insert or update face embedding for a student';

-- ============================================================================
-- Sample Usage
-- ============================================================================
-- 
-- 1. Insert a face embedding:
-- SELECT upsert_face_embedding(
--     '12345678',                    -- NIS
--     'user-uuid-here',              -- user_id
--     '[0.1, 0.2, ...]'::vector(512), -- embedding
--     'kiosk_001',                   -- camera_id
--     0.95                           -- quality_score
-- );
--
-- 2. Search for matching face:
-- SELECT * FROM find_face_match(
--     '[0.1, 0.2, ...]'::vector(512), -- query embedding
--     0.6,                            -- threshold
--     1                               -- max results
-- );
--
-- ============================================================================
