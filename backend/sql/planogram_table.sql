-- Planogram table schema and sample data
-- This defines the expected product layout on shelves

-- Create planogram table (if not using create_tables.sql)
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

-- Sample planogram data
-- Replace with your actual planogram data

INSERT INTO planograms (planogram_name, sku, product_name, shelf_location, expected_count, x_position, y_position) VALUES
('Shelf_A1', 'COKE_500ML', 'Coca Cola 500ml', 'A1', 6, 0.2, 0.3),
('Shelf_A1', 'PEPSI_500ML', 'Pepsi 500ml', 'A1', 4, 0.5, 0.3),
('Shelf_A1', 'SPRITE_500ML', 'Sprite 500ml', 'A1', 3, 0.8, 0.3),
('Shelf_A2', 'WATER_500ML', 'Water Bottle 500ml', 'A2', 8, 0.3, 0.5),
('Shelf_A2', 'JUICE_500ML', 'Orange Juice 500ml', 'A2', 5, 0.7, 0.5)
ON CONFLICT DO NOTHING;

-- Query to get planogram for a specific shelf
-- SELECT * FROM planograms WHERE shelf_location = 'A1';

-- Query to get expected count for a SKU
-- SELECT expected_count FROM planograms WHERE sku = 'COKE_500ML' AND shelf_location = 'A1';

