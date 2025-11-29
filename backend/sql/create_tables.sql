-- SQL script to create all database tables for VisionStock
-- Run this if you prefer SQL over Python init_database.py

-- Detections table
CREATE TABLE IF NOT EXISTS detections (
    id SERIAL PRIMARY KEY,
    image_path VARCHAR NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
    sku VARCHAR,
    class_name VARCHAR NOT NULL,
    confidence FLOAT NOT NULL,
    x_center FLOAT NOT NULL,
    y_center FLOAT NOT NULL,
    width FLOAT NOT NULL,
    height FLOAT NOT NULL,
    shelf_location VARCHAR,
    meta_data JSONB,
    model_version_id INTEGER
);

CREATE INDEX IF NOT EXISTS idx_detections_sku ON detections(sku);
CREATE INDEX IF NOT EXISTS idx_detections_timestamp ON detections(timestamp);
CREATE INDEX IF NOT EXISTS idx_detections_shelf_location ON detections(shelf_location);

-- Planograms table
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

-- Discrepancies table
CREATE TABLE IF NOT EXISTS discrepancies (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
    planogram_id INTEGER NOT NULL,
    sku VARCHAR NOT NULL,
    shelf_location VARCHAR NOT NULL,
    discrepancy_type VARCHAR NOT NULL,
    expected_count INTEGER NOT NULL,
    detected_count INTEGER NOT NULL,
    confidence_score FLOAT,
    meta_data JSONB,
    FOREIGN KEY (planogram_id) REFERENCES planograms(id)
);

CREATE INDEX IF NOT EXISTS idx_discrepancies_sku ON discrepancies(sku);
CREATE INDEX IF NOT EXISTS idx_discrepancies_type ON discrepancies(discrepancy_type);
CREATE INDEX IF NOT EXISTS idx_discrepancies_timestamp ON discrepancies(timestamp);

-- Model versions table
CREATE TABLE IF NOT EXISTS model_versions (
    id SERIAL PRIMARY KEY,
    version_name VARCHAR NOT NULL UNIQUE,
    model_type VARCHAR NOT NULL,
    model_path VARCHAR NOT NULL,
    base_model VARCHAR,
    training_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
    epochs INTEGER,
    dataset_path VARCHAR,
    is_active INTEGER DEFAULT 0 NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
    meta_data JSONB
);

CREATE INDEX IF NOT EXISTS idx_model_versions_name ON model_versions(version_name);
CREATE INDEX IF NOT EXISTS idx_model_versions_type ON model_versions(model_type);

-- Model metrics table
CREATE TABLE IF NOT EXISTS model_metrics (
    id SERIAL PRIMARY KEY,
    model_version_id INTEGER NOT NULL,
    evaluation_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
    map50 FLOAT NOT NULL,
    map50_95 FLOAT NOT NULL,
    precision FLOAT NOT NULL,
    recall FLOAT NOT NULL,
    f1_score FLOAT NOT NULL,
    inference_time_ms FLOAT,
    test_dataset_path VARCHAR,
    num_test_images INTEGER,
    meta_data JSONB,
    FOREIGN KEY (model_version_id) REFERENCES model_versions(id)
);

CREATE INDEX IF NOT EXISTS idx_model_metrics_version ON model_metrics(model_version_id);
CREATE INDEX IF NOT EXISTS idx_model_metrics_date ON model_metrics(evaluation_date);

-- Add foreign key for detections.model_version_id
ALTER TABLE detections 
ADD CONSTRAINT fk_detections_model_version 
FOREIGN KEY (model_version_id) REFERENCES model_versions(id);

