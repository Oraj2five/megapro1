-- ============================================================================
-- SMART BPI — Daily Sales Module
-- Migration: 002_create_daily_sales.sql
-- Target Database: Supabase PostgreSQL 17.x
-- Description: Creates public.daily_sales with user isolation, non-negative
--              revenue constraints, unique (user_id, sale_date) constraint,
--              composite index, and strict Row Level Security (RLS).
-- ============================================================================

-- 1. Ensure required cryptographic extension for UUID generation
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- 2. Create daily_sales table
CREATE TABLE IF NOT EXISTS public.daily_sales (
    -- Unique identifier for the daily sales entry
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Owner identification: Verified Firebase Authentication UID
    user_id TEXT NOT NULL,

    -- Calendar date of the sales record
    sale_date DATE NOT NULL,

    -- Daily revenue in INR: strictly constrained to non-negative values
    daily_revenue NUMERIC(15, 2) NOT NULL,
    CONSTRAINT chk_daily_revenue_non_negative CHECK (daily_revenue >= 0),

    -- Audit timestamps (server clock)
    created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),

    -- Concurrency & Idempotency: Prevent duplicate or conflicting records
    -- for the same business user and sales date at database level.
    CONSTRAINT uq_daily_sales_user_date UNIQUE (user_id, sale_date)
);

-- 3. Performance Indexes
-- Composite index for fast user lookup and date-ordered retrieval:
-- SELECT * FROM public.daily_sales WHERE user_id = $1 ORDER BY sale_date DESC;
CREATE INDEX IF NOT EXISTS idx_daily_sales_user_date
ON public.daily_sales (user_id, sale_date DESC);

-- 4. Row Level Security (RLS) Configuration
-- RLS is ENABLED to lock down direct access.
-- No policies are granted to 'anon' or 'authenticated' roles.
-- All access is mediated strictly through the FastAPI backend using the Supabase service-role key,
-- with strict user_id filtering on every query.
ALTER TABLE public.daily_sales ENABLE ROW LEVEL SECURITY;

-- 5. Schema Documentation Comments
COMMENT ON TABLE public.daily_sales IS 'Stores authenticated manual daily sales entries for SMART BPI sales forecasting.';
COMMENT ON COLUMN public.daily_sales.id IS 'Unique UUID for this daily sales record.';
COMMENT ON COLUMN public.daily_sales.user_id IS 'Verified Firebase Authentication UID of the business owner.';
COMMENT ON COLUMN public.daily_sales.sale_date IS 'Calendar date of sales (YYYY-MM-DD).';
COMMENT ON COLUMN public.daily_sales.daily_revenue IS 'Total revenue in INR for this date, must be non-negative.';
COMMENT ON COLUMN public.daily_sales.created_at IS 'Server timestamp when record was created.';
COMMENT ON COLUMN public.daily_sales.updated_at IS 'Server timestamp when record was last updated.';
