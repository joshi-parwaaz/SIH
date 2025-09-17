# Ocean Hazard Analysis System - Model 1.5

A real-time ocean hazard detection and analysis system that combines data from multiple sources with NLP/ML pipeline to provide structured hazard information.

## 🌊 Features

- **Multi-source Data Collection**: INCOIS XML feeds, Twitter, YouTube
- **ML-powered Analysis**: Relevance classification and information extraction
- **Real-time API**: FastAPI server with structured JSON output
- **Comprehensive Coverage**: Tsunami, flood, cyclone, storm surge detection
- **India-focused**: Geographic filtering and multilingual support

## 📂 Project Structure

```
Model1.5/
├── scrapers/           # Data collection modules
│   ├── incois_scraper.py    # INCOIS XML feed scraper
│   ├── twitter_scraper.py   # Twitter via snscrape
│   └── youtube_scraper.py   # YouTube API integration
├── models/             # ML/NLP processing
│   ├── relevance_classifier.py  # Content relevance detection
│   └── extractor.py            # Information extraction
├── app/               # Web API and orchestration
│   ├── schema.py           # Pydantic data models
│   ├── pipeline.py         # Main processing pipeline
│   └── main.py            # FastAPI server
├── tests/             # Test suite
│   └── test_pipeline.py    # Comprehensive tests
├── requirements.txt   # Python dependencies
└── README.md         # This file
```

## 🚀 Quick Start

### Option 1: Docker (Recommended)

**Prerequisites:**
- Docker Desktop installed
- Docker Compose available

```bash
# Clone and navigate to project
cd Model1.5

# Copy environment template (optional)
cp .env.example .env
# Edit .env file with your API keys

# Start the system (development mode)
docker-compose up -d

# Or use the deployment script
./deploy.bat start --dev     # Windows
./scripts/deploy.sh start --dev  # Linux/Mac

# Visit the API
# http://localhost:8000 - Main API
# http://localhost:8000/docs - Interactive documentation
```

**Production Deployment:**
```bash
# Start with all services (nginx, redis, postgres)
./deploy.bat start --prod     # Windows
./scripts/deploy.sh start --prod  # Linux/Mac

# Access via nginx reverse proxy
# http://localhost - Main application
# http://localhost:8000 - Direct API access
```

### Option 2: Local Development

**Prerequisites:**
- Python 3.11+
- Virtual environment

```bash
# Navigate to Model1.5 directory
cd Model1.5

# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Optional: Configure YouTube API

```bash
# Get free API key from Google Cloud Console
# Set environment variable
set YOUTUBE_API_KEY=your-api-key-here  # Windows
# export YOUTUBE_API_KEY="your-api-key-here"  # Linux/Mac
```

### 3. Run the Server

```bash
# Start FastAPI server
python app/main.py

# Or using uvicorn directly
uvicorn app.main:app --reload
```

### 4. Test the API

Visit: http://127.0.0.1:8000

**Main endpoints:**
- `GET /analyze` - Run complete hazard analysis
- `GET /health` - Health check
- `GET /status` - Detailed system status
- `GET /docs` - Interactive API documentation

## 🐳 Docker Commands

### Basic Operations
```bash
# Start system
docker-compose up -d

# Stop system  
docker-compose down

# View logs
docker-compose logs -f hazard-api

# Check status
docker-compose ps
```

### Development vs Production

**Development Mode (API Only):**
```bash
docker-compose up -d
# Access: http://localhost:8000
```

**Production Mode (Full Stack):**
```bash
docker-compose --profile production --profile cache --profile database up -d
# Access: http://localhost (nginx proxy)
# Direct API: http://localhost:8000
```

### Using Deployment Scripts

**Windows:**
```cmd
deploy.bat start --dev     # Development mode
deploy.bat start --prod    # Production mode
deploy.bat logs            # View logs
deploy.bat status          # Check status
deploy.bat clean           # Clean up
```

**Linux/Mac:**
```bash
./scripts/deploy.sh start --dev   # Development mode
./scripts/deploy.sh start --prod  # Production mode
./scripts/deploy.sh logs          # View logs
./scripts/deploy.sh status        # Check status
./scripts/deploy.sh clean         # Clean up
```

## 📊 API Usage Examples

### Analyze Hazards

```bash
curl http://127.0.0.1:8000/analyze
```

**Sample Response:**
```json
{
  "reports": [
    {
      "source": "INCOIS",
      "text": "Tsunami warning issued for coastal areas",
      "hazard_type": "tsunami",
      "severity": "high",
      "urgency": "immediate",
      "sentiment": "concern",
      "misinformation": false,
      "location": "coastal_area",
      "confidence": 0.85,
      "timestamp": "2025-01-15T10:30:00Z"
    }
  ],
  "total_sources_checked": 3,
  "relevant_reports": 1,
  "processing_time_seconds": 2.34,
  "timestamp": "2025-01-15T10:30:00Z"
}
```

### System Status

```bash
curl http://127.0.0.1:8000/status
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file from the template:
```bash
cp .env.example .env
```

**Key Configuration:**
- `YOUTUBE_API_KEY` - YouTube Data API key (optional)
- `POSTGRES_PASSWORD` - Database password (production)
- `LOG_LEVEL` - Logging level (INFO, DEBUG, ERROR)
- `DEBUG` - Enable debug mode (true/false)

### Data Sources

1. **INCOIS** (Always enabled)
   - Official Indian ocean warning system
   - No API key required
   - XML feed parsing

2. **Twitter** (Auto-detected)
   - Uses snscrape (no API key needed)
   - Geographic filtering for India
   - Rate-limited to avoid blocking

3. **YouTube** (Optional)
   - Requires free Google API key
   - Video title/description analysis
   - Regional filtering (India)

### ML Models

- **Relevance Classifier**: Uses multilingual BERT for content filtering
- **Information Extractor**: Rule-based + pattern matching for structured data
- **Fallback System**: Keyword-based processing if ML models unavailable

## 🐳 Docker Architecture

### Services Overview

**Core Service:**
- `hazard-api` - Main FastAPI application

**Production Services:**
- `nginx` - Reverse proxy with rate limiting
- `redis` - Caching layer (future feature)
- `postgres` - Data persistence (future feature)

### Container Features

- **Multi-stage builds** for optimized image size
- **Non-root user** for security
- **Health checks** for reliability  
- **Volume mounts** for persistent data
- **Network isolation** with custom bridge network

### Production Deployment

```bash
# Full production stack
docker-compose --profile production --profile cache --profile database up -d

# Services included:
# - API server (port 8000)
# - Nginx proxy (port 80/443)  
# - Redis cache (port 6379)
# - PostgreSQL database (port 5432)
```

## 🧪 Testing

```bash
# Run all tests
python tests/test_pipeline.py

# Run with pytest (if installed)
pytest tests/ -v
```

**Test Coverage:**
- ✅ Scraper functionality
- ✅ ML model accuracy
- ✅ Pipeline integration
- ✅ API endpoints
- ✅ Error handling

## 📝 Dependencies

### Core Requirements
- `fastapi` - Web API framework
- `requests` - HTTP requests for scrapers
- `transformers` - ML models for classification
- `snscrape` - Twitter scraping without API
- `pydantic` - Data validation and serialization

### Optional Enhancements
- `torch` - Deep learning backend
- `accelerate` - Faster transformer inference
- `pandas` - Data manipulation (future features)

## 🛠️ Development

### Adding New Sources

1. Create scraper in `scrapers/`
2. Add to `pipeline.py` source configs
3. Add tests in `test_pipeline.py`

### Improving ML Models

1. Update `models/relevance_classifier.py` for better accuracy
2. Enhance `models/extractor.py` with more patterns
3. Add confidence scoring improvements

### API Extensions

1. Add new endpoints in `app/main.py`
2. Define response models in `app/schema.py`
3. Update documentation

## 🐛 Troubleshooting

### Common Issues

**Import Errors**
```bash
# Ensure you're in the right directory
cd Model1.5
python -c "import app.main"  # Should work without errors
```

**Missing Dependencies**
```bash
# Reinstall with specific versions
pip install -r requirements.txt --force-reinstall
```

**Twitter Scraping Issues**
```bash
# snscrape sometimes needs updates
pip install --upgrade snscrape
```

**ML Model Loading Errors**
```bash
# First run downloads models - ensure internet connection
# Models cached in ~/.cache/huggingface/
```

### Performance Optimization

1. **Faster Startup**: Pre-load ML models
2. **Better Caching**: Cache scraper results
3. **Async Processing**: Parallel source fetching
4. **Resource Limits**: Configure max items per source

## 📈 Monitoring

The system provides built-in monitoring:

- **Health Checks**: `/health` endpoint
- **Metrics**: Request counting, uptime tracking
- **Source Status**: Individual source health monitoring
- **Error Logging**: Comprehensive error tracking

## 🔮 Future Enhancements

1. **Real-time Processing**: WebSocket connections
2. **Database Integration**: Historical data storage
3. **Advanced ML**: Fine-tuned models for Indian context
4. **Mobile API**: Optimized endpoints for mobile apps
5. **Alert System**: Push notifications for critical hazards

## 📄 License

This project is part of the Smart India Hackathon 2024 submission.

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

---

**Happy Coding! 🌊💻**

For questions or issues, please check the API documentation at `/docs` when the server is running.