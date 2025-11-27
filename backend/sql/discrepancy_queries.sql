-- SQL queries for finding discrepancies between planogram and detections

-- 1. Find missing products (expected but not detected)
SELECT 
    p.planogram_name,
    p.sku,
    p.product_name,
    p.shelf_location,
    p.expected_count,
    COALESCE(COUNT(d.id), 0) as detected_count,
    p.expected_count - COALESCE(COUNT(d.id), 0) as missing_count,
    'missing' as discrepancy_type
FROM planograms p
LEFT JOIN detections d ON p.sku = d.sku AND p.shelf_location = d.shelf_location
WHERE p.expected_count > COALESCE(COUNT(d.id), 0)
GROUP BY p.id, p.planogram_name, p.sku, p.product_name, p.shelf_location, p.expected_count
HAVING p.expected_count > COALESCE(COUNT(d.id), 0);

-- 2. Find extra products (detected but not in planogram)
SELECT 
    d.sku,
    d.class_name,
    d.shelf_location,
    COUNT(d.id) as detected_count,
    0 as expected_count,
    COUNT(d.id) as extra_count,
    'extra' as discrepancy_type
FROM detections d
LEFT JOIN planograms p ON d.sku = p.sku AND d.shelf_location = p.shelf_location
WHERE p.id IS NULL
GROUP BY d.sku, d.class_name, d.shelf_location;

-- 3. Find low stock (detected count less than expected)
SELECT 
    p.planogram_name,
    p.sku,
    p.product_name,
    p.shelf_location,
    p.expected_count,
    COUNT(d.id) as detected_count,
    p.expected_count - COUNT(d.id) as shortage,
    'low_stock' as discrepancy_type
FROM planograms p
LEFT JOIN detections d ON p.sku = d.sku AND p.shelf_location = d.shelf_location
GROUP BY p.id, p.planogram_name, p.sku, p.product_name, p.shelf_location, p.expected_count
HAVING COUNT(d.id) < p.expected_count AND COUNT(d.id) > 0;

-- 4. Find misplaced products (wrong shelf location)
SELECT 
    d.sku,
    d.class_name,
    d.shelf_location as detected_location,
    p.shelf_location as expected_location,
    COUNT(d.id) as count,
    'misplaced' as discrepancy_type
FROM detections d
JOIN planograms p ON d.sku = p.sku
WHERE d.shelf_location != p.shelf_location
GROUP BY d.sku, d.class_name, d.shelf_location, p.shelf_location;

-- 5. Summary of all discrepancies for a specific planogram
SELECT 
    p.planogram_name,
    p.sku,
    p.product_name,
    p.shelf_location,
    p.expected_count,
    COALESCE(COUNT(d.id), 0) as detected_count,
    CASE 
        WHEN COALESCE(COUNT(d.id), 0) = 0 THEN 'missing'
        WHEN COALESCE(COUNT(d.id), 0) < p.expected_count THEN 'low_stock'
        WHEN COALESCE(COUNT(d.id), 0) > p.expected_count THEN 'overstock'
        ELSE 'correct'
    END as status
FROM planograms p
LEFT JOIN detections d ON p.sku = d.sku AND p.shelf_location = d.shelf_location
WHERE p.planogram_name = 'Shelf_A1'  -- Replace with your planogram name
GROUP BY p.id, p.planogram_name, p.sku, p.product_name, p.shelf_location, p.expected_count;

-- 6. Get discrepancy statistics
SELECT 
    discrepancy_type,
    COUNT(*) as count,
    AVG(expected_count - detected_count) as avg_difference
FROM discrepancies
GROUP BY discrepancy_type;

-- 7. Recent discrepancies (last 24 hours)
SELECT 
    d.*,
    p.product_name
FROM discrepancies d
JOIN planograms p ON d.planogram_id = p.id
WHERE d.timestamp >= NOW() - INTERVAL '24 hours'
ORDER BY d.timestamp DESC;

