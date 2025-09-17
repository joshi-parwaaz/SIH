-- Initialize database for Ocean Hazard Analysis System
-- This script creates tables for storing hazard reports and system logs

-- Create extension for UUID generation
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create hazard_reports table
CREATE TABLE IF NOT EXISTS hazard_reports (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source VARCHAR(50) NOT NULL,
    text TEXT NOT NULL,
    hazard_type VARCHAR(50) NOT NULL,
    severity VARCHAR(20) NOT NULL,
    urgency VARCHAR(20) NOT NULL,
    sentiment VARCHAR(20) NOT NULL,
    location VARCHAR(100),
    confidence DECIMAL(3,2) NOT NULL,
    misinformation BOOLEAN DEFAULT FALSE,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- Indexes for better query performance
    CONSTRAINT valid_confidence CHECK (confidence >= 0.0 AND confidence <= 1.0)
);

-- Create indexes
CREATE INDEX idx_hazard_reports_timestamp ON hazard_reports(timestamp);
CREATE INDEX idx_hazard_reports_source ON hazard_reports(source);
CREATE INDEX idx_hazard_reports_hazard_type ON hazard_reports(hazard_type);
CREATE INDEX idx_hazard_reports_location ON hazard_reports(location);
CREATE INDEX idx_hazard_reports_confidence ON hazard_reports(confidence);

-- Create system_logs table
CREATE TABLE IF NOT EXISTS system_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    level VARCHAR(20) NOT NULL,
    message TEXT NOT NULL,
    component VARCHAR(50),
    timestamp TIMESTAMP WITH TIME Z

DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB
);

-- Create index on system_logs
CREATE INDEX idx_system_logs_timestamp ON system_logs(timestamp);
CREATE INDEX idx_system_logs_level ON system_logs(level);
CREATE INDEX idx_system_logs_component ON system_logs(component);

-- Create api_requests table for monitoring
CREATE TABLE IF NOT EXISTS api_requests (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    endpoint VARCHAR(100) NOT NULL,
    method VARCHAR(10) NOT NULL,
    status_code INTEGER NOT NULL,
    response_time_ms INTEGER,
    ip_address INET,
    user_agent TEXT,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for api_requests
CREATE INDEX idx_api_requests_timestamp ON api_requests(timestamp);
CREATE INDEX idx_api_requests_endpoint ON api_requests(endpoint);
CREATE INDEX idx_api_requests_status_code ON api_requests(status_code);

-- Insert sample data for testing
INSERT INTO hazard_reports (source, text, hazard_type, severity, urgency, sentiment, location, confidence, misinformation) VALUES
('INCOIS', 'Tsunami warning issued for coastal Tamil Nadu', 'tsunami', 'high', 'immediate', 'concern', 'tamil_nadu', 0.95, FALSE),
('Twitter', 'Heavy flooding in Mumbai due to monsoon rains', 'flood', 'medium', 'high', 'neutral', 'maharashtra', 0.78, FALSE),
('YouTube', 'Cyclone Biparjoy updates: Storm approaches Gujarat coast', 'cyclone', 'high', 'high', 'concern', 'gujarat', 0.89, FALSE);

-- Create view for recent high-priority alerts
CREATE OR REPLACE VIEW high_priority_alerts AS
SELECT 
    id,
    source,
    text,
    hazard_type,
    severity,
    urgency,
    location,
    confidence,
    timestamp
FROM hazard_reports 
WHERE 
    confidence >= 0.7 
    AND urgency IN ('high', 'immediate')
    AND timestamp >= NOW() - INTERVAL '24 hours'
ORDER BY confidence DESC, timestamp DESC;

-- Grant permissions to hazard_user
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO hazard_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO hazard_user;
GRANT ALL PRIVILEGES ON ALL FUNCTIONS IN SCHEMA public TO hazard_user;