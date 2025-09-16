// MongoDB initialization script for Ocean Hazard Prediction System

// Create database
db = db.getSiblingDB('ocean_hazard_db');

// Create application user
db.createUser({
  user: 'ocean_hazard_app',
  pwd: 'ocean_hazard_app_2024',
  roles: [
    {
      role: 'readWrite',
      db: 'ocean_hazard_db'
    }
  ]
});

// Create collections with indexes
print('Creating collections and indexes...');

// Historical events collection
db.createCollection('historical_events');
db.historical_events.createIndex({ 'location': '2dsphere' });
db.historical_events.createIndex({ 'event_date': 1 });
db.historical_events.createIndex({ 'hazard_type': 1 });
db.historical_events.createIndex({ 'magnitude': 1 });

// Sensor data collection
db.createCollection('sensor_data');
db.sensor_data.createIndex({ 'location': '2dsphere' });
db.sensor_data.createIndex({ 'timestamp': 1 });
db.sensor_data.createIndex({ 'sensor_type': 1 });

// Risk assessments collection
db.createCollection('risk_assessments');
db.risk_assessments.createIndex({ 'location': '2dsphere' });
db.risk_assessments.createIndex({ 'assessment_date': 1 });
db.risk_assessments.createIndex({ 'risk_level': 1 });
db.risk_assessments.createIndex({ 'hazard_type': 1 });

// Hotspots collection
db.createCollection('hotspots');
db.hotspots.createIndex({ 'location': '2dsphere' });
db.hotspots.createIndex({ 'identified_at': 1 });
db.hotspots.createIndex({ 'risk_score': 1 });

// Alerts collection
db.createCollection('alerts');
db.alerts.createIndex({ 'location': '2dsphere' });
db.alerts.createIndex({ 'created_at': 1 });
db.alerts.createIndex({ 'alert_level': 1 });
db.alerts.createIndex({ 'status': 1 });

// Model metadata collection
db.createCollection('model_metadata');
db.model_metadata.createIndex({ 'model_name': 1 });
db.model_metadata.createIndex({ 'created_at': 1 });
db.model_metadata.createIndex({ 'model_type': 1 });

// User feedback collection
db.createCollection('user_feedback');
db.user_feedback.createIndex({ 'location': '2dsphere' });
db.user_feedback.createIndex({ 'submitted_at': 1 });
db.user_feedback.createIndex({ 'feedback_type': 1 });

// API usage logs collection
db.createCollection('api_logs');
db.api_logs.createIndex({ 'timestamp': 1 });
db.api_logs.createIndex({ 'endpoint': 1 });
db.api_logs.createIndex({ 'user_id': 1 });

// Insert sample data
print('Inserting sample data...');

// Sample historical events
db.historical_events.insertMany([
  {
    event_id: 'tsunami_2004_indian_ocean',
    hazard_type: 'tsunami',
    location: {
      type: 'Point',
      coordinates: [95.979, 3.295] // Indian Ocean
    },
    event_date: new Date('2004-12-26'),
    magnitude: 9.1,
    description: '2004 Indian Ocean earthquake and tsunami',
    affected_areas: ['Indonesia', 'Thailand', 'Sri Lanka', 'India'],
    casualties: 230000,
    economic_impact: 15000000000
  },
  {
    event_id: 'tsunami_2011_japan',
    hazard_type: 'tsunami',
    location: {
      type: 'Point',
      coordinates: [142.373, 38.297] // Japan
    },
    event_date: new Date('2011-03-11'),
    magnitude: 9.1,
    description: '2011 Tōhoku earthquake and tsunami',
    affected_areas: ['Japan'],
    casualties: 19759,
    economic_impact: 235000000000
  },
  {
    event_id: 'hurricane_katrina_2005',
    hazard_type: 'storm_surge',
    location: {
      type: 'Point',
      coordinates: [-89.6, 29.15] // Gulf of Mexico
    },
    event_date: new Date('2005-08-29'),
    magnitude: 5, // Category 5
    description: 'Hurricane Katrina storm surge',
    affected_areas: ['Louisiana', 'Mississippi', 'Alabama'],
    casualties: 1833,
    economic_impact: 125000000000
  }
]);

// Sample sensor data
db.sensor_data.insertMany([
  {
    sensor_id: 'buoy_001_pacific',
    sensor_type: 'ocean_buoy',
    location: {
      type: 'Point',
      coordinates: [-156.5, 19.5] // Hawaii
    },
    timestamp: new Date(),
    readings: {
      wave_height: 2.3,
      sea_temperature: 26.8,
      wind_speed: 12.5,
      atmospheric_pressure: 1013.2,
      water_depth: 4200
    },
    status: 'active'
  },
  {
    sensor_id: 'seismic_001_japan',
    sensor_type: 'seismometer',
    location: {
      type: 'Point',
      coordinates: [139.6917, 35.6895] // Tokyo
    },
    timestamp: new Date(),
    readings: {
      magnitude: 2.1,
      depth: 10,
      frequency: 5.2
    },
    status: 'active'
  }
]);

print('Database initialization completed successfully!');
print('Created collections: historical_events, sensor_data, risk_assessments, hotspots, alerts, model_metadata, user_feedback, api_logs');
print('Sample data inserted for testing purposes.');