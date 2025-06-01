CREATE TABLE IF NOT EXISTS alerts (
    id SERIAL PRIMARY KEY,
    camera_id VARCHAR,
    camera_name VARCHAR,
    timestamp BIGINT,
    image_url VARCHAR,
    distance FLOAT,
    description TEXT
);
