# Ocean Hazard Analysis System - Pro

Advanced real-time ocean hazard detection system with custom Twitter scraper and ML pipeline.

## Key Features

- 4+ Data Sources: Government feeds, Custom Twitter scraper, Regional news, YouTube
- Custom Twitter Integration: No API keys needed - uses Nitter instances with RSS fallback
- ML-Powered Analysis: Relevance classification, information extraction, anomaly detection
- Real-time API: FastAPI server with comprehensive JSON responses
- India-Focused: Geographic filtering for coastal regions and disaster-prone areas
- Production Ready: Docker containerization with nginx proxy

## Quick Start

### Docker (Recommended)
`ash
# Clone repository
git clone <repo-url>
cd 1pro

# Start system
docker-compose up --build

# API available at http://localhost:8000
# Documentation at http://localhost:8000/docs
`

### Local Development
`ash
# Setup environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Start server
uvicorn app.main:app --reload
`

## API Usage

- GET /analyze - Returns comprehensive hazard analysis from all sources
- GET /health - Health check
- GET /docs - Interactive documentation

## Custom Twitter Scraper

Works without API keys using:
- Primary: Nitter instances (nitter.net, nitter.it)
- Fallback: RSS-based Google search
- Auto-failover with smart disaster keyword filtering

## Performance

- Fast Mode: 10-30 seconds (development)
- Production Mode: 30-60 seconds (full ML)
- Twitter Success Rate: 95%+ (with fallbacks)

Built for disaster preparedness across India's coastal regions.
