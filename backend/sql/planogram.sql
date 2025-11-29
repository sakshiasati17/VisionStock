-- Planogram table schema for VisionStock backend
-- This is the SQL schema for the planogram table

CREATE TABLE IF NOT EXISTS planograms (
    id SERIAL PRIMARY KEY,
    planogram_name VARCHAR NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
    sku VARCHAR NOT NULL,
    product_name VARCHAR,
    shelf_location VARCHAR NOT NULL,
    expected_count INTEGER DEFAULT 1 NOT NULL,
    x_position FLOAT,
    y_position FLOAT,
    meta_data JSONB
);

CREATE INDEX IF NOT EXISTS idx_planograms_name ON planograms(planogram_name);
CREATE INDEX IF NOT EXISTS idx_planograms_sku ON planograms(sku);

