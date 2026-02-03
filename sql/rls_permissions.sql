-- ============================================================================
-- Row Level Security & Table Permissions for Backend Access
-- ============================================================================
-- Script ini memberikan permission kepada service_role untuk mengakses
-- tabel-tabel yang diperlukan oleh backend application.
--
-- Jalankan script ini di Supabase SQL Editor dengan role postgres/admin.
-- ============================================================================

-- ============================================================================
-- Step 1: Grant table permissions to service_role
-- ============================================================================

-- Grant permissions pada tabel user_profiles
GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE public.user_profiles TO service_role;
GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE public.user_profiles TO authenticated;

-- Grant permissions pada tabel biodata_siswa (untuk RPC get_student_by_user_id)
GRANT SELECT ON TABLE public.biodata_siswa TO service_role;
GRANT SELECT ON TABLE public.biodata_siswa TO authenticated;

-- ============================================================================
-- Step 2: Enable RLS on tables (if not already enabled)
-- ============================================================================

ALTER TABLE public.user_profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.biodata_siswa ENABLE ROW LEVEL SECURITY;

-- ============================================================================
-- Step 3: Create RLS Policies for user_profiles
-- ============================================================================

-- Drop existing policies if any (to recreate cleanly)
DROP POLICY IF EXISTS "Allow service_role full access to user_profiles" ON public.user_profiles;
DROP POLICY IF EXISTS "Allow authenticated users to read own profile" ON public.user_profiles;
DROP POLICY IF EXISTS "Allow authenticated users to update own profile" ON public.user_profiles;

-- Policy: service_role can do everything
CREATE POLICY "Allow service_role full access to user_profiles"
ON public.user_profiles
FOR ALL
TO service_role
USING (true)
WITH CHECK (true);

-- Policy: Authenticated users can read their own profile
CREATE POLICY "Allow authenticated users to read own profile"
ON public.user_profiles
FOR SELECT
TO authenticated
USING (auth.uid() = user_id);

-- Policy: Authenticated users can update their own profile
CREATE POLICY "Allow authenticated users to update own profile"
ON public.user_profiles
FOR UPDATE
TO authenticated
USING (auth.uid() = user_id)
WITH CHECK (auth.uid() = user_id);

-- ============================================================================
-- Step 4: Create RLS Policies for biodata_siswa
-- ============================================================================

-- Drop existing policies if any
DROP POLICY IF EXISTS "Allow service_role full access to biodata_siswa" ON public.biodata_siswa;
DROP POLICY IF EXISTS "Allow authenticated users to read biodata_siswa" ON public.biodata_siswa;

-- Policy: service_role can do everything
CREATE POLICY "Allow service_role full access to biodata_siswa"
ON public.biodata_siswa
FOR ALL
TO service_role
USING (true)
WITH CHECK (true);

-- Policy: Authenticated users can read biodata (public data)
CREATE POLICY "Allow authenticated users to read biodata_siswa"
ON public.biodata_siswa
FOR SELECT
TO authenticated
USING (true);

-- ============================================================================
-- Step 5: Ensure sequences are accessible (if using serial/bigserial)
-- ============================================================================

-- Grant usage on sequences if any
GRANT USAGE ON ALL SEQUENCES IN SCHEMA public TO service_role;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA public TO authenticated;

-- ============================================================================
-- Step 6: Create RPC Functions
-- ============================================================================

-- Function untuk lookup student info by user_id
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

-- Grant execute permission for get_student_by_user_id
GRANT EXECUTE ON FUNCTION get_student_by_user_id(UUID) TO service_role;
GRANT EXECUTE ON FUNCTION get_student_by_user_id(UUID) TO authenticated;

COMMENT ON FUNCTION get_student_by_user_id(UUID) IS 'Lookup student info by user_id from user_profiles and biodata_siswa';

-- ============================================================================
-- Step 7: Additional helper - bypass RLS for service_role
-- ============================================================================
-- IMPORTANT: Supabase service_role should bypass RLS by default,
-- but if it doesn't work, you can run this as superuser:

-- ALTER ROLE service_role BYPASSRLS; -- Only run if needed, requires superuser

-- ============================================================================
-- VERIFICATION: Test queries (run these to verify access)
-- ============================================================================

-- Test 1: Can service_role select from user_profiles?
-- SELECT * FROM user_profiles LIMIT 1;

-- Test 2: Can service_role call get_student_by_user_id?
-- SELECT * FROM get_student_by_user_id('your-test-uuid-here');

-- ============================================================================
-- Notes:
-- ============================================================================
-- 1. service_role adalah role khusus Supabase yang seharusnya bypass RLS
-- 2. Jika masih error, pastikan SUPABASE_SERVICE_ROLE_KEY digunakan (bukan anon key)
-- 3. Untuk debug, coba disable RLS sementara:
--    ALTER TABLE user_profiles DISABLE ROW LEVEL SECURITY;
-- 4. Re-enable setelah testing:
--    ALTER TABLE user_profiles ENABLE ROW LEVEL SECURITY;
-- ============================================================================
