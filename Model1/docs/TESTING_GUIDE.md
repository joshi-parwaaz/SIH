# 🧪 Model1 Testing Guide

## 📋 Pre-Deployment Testing Checklist

Before integrating Model1 into the main application, follow this comprehensive testing checklist to ensure everything works properly.

### ✅ **Phase 1: Environment Setup**

1. **Dependencies Installation**
   ```bash
   cd Model1
   pip install -r requirements.txt
   pip install pytest pytest-asyncio httpx  # Testing dependencies
   ```

2. **Configuration Setup**
   ```bash
   cp .env.example .env
   # Edit .env with your database URLs and API keys
   ```

3. **Database Setup**
   ```bash
   # Start MongoDB (required for testing)
   docker-compose up -d mongodb
   ```

### ✅ **Phase 2: Unit Testing**

Run individual component tests to verify core functionality:

```bash
# Run all unit tests
pytest tests/test_ml_components.py -v

# Run specific test categories
pytest tests/test_ml_components.py::TestPreprocessingPipeline -v
pytest tests/test_ml_components.py::TestNLPAnalysisEngine -v
pytest tests/test_ml_components.py::TestGeolocationExtractor -v
pytest tests/test_ml_components.py::TestAnomalyDetectionEngine -v
```

**Expected Results:**
- ✅ All preprocessing tests pass (language detection, text normalization)
- ✅ NLP analysis tests pass (hazard classification, sentiment analysis)
- ✅ Geolocation extraction tests pass (location matching, coordinate validation)
- ✅ Anomaly detection tests pass (spatial clustering, temporal spikes)

### ✅ **Phase 3: API Testing**

Test the FastAPI endpoints and integration:

```bash
# Run API tests
pytest tests/test_api.py -v

# Test specific endpoints
pytest tests/test_api.py::TestAPIEndpoints::test_process_single_report_valid -v
pytest tests/test_api.py::TestAPIEndpoints::test_multilingual_processing -v
```

**Expected Results:**
- ✅ Health endpoint responds correctly
- ✅ Single report processing works
- ✅ Batch processing handles multiple reports
- ✅ Multilingual input is processed correctly
- ✅ Error handling works properly

### ✅ **Phase 4: System Integration Testing**

Test the complete system end-to-end:

```bash
# Start the complete system
docker-compose up -d

# Wait for services to be ready (30-60 seconds)
sleep 60

# Run integration tests
python tests/test_system_integration.py
```

### ✅ **Phase 5: Performance Testing**

Test system performance and scalability:

```bash
# Run performance tests
pytest tests/test_api.py::TestAPIPerformance -v

# Load testing (optional - requires additional tools)
python tests/load_test.py
```

### ✅ **Phase 6: Manual Testing Scenarios**

#### **Scenario 1: Multilingual Hazard Detection**

Test with various Indian languages:

```bash
# Test Hindi
curl -X POST "http://localhost:8000/api/v1/process/single" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "चेन्नई के तट पर सुनामी की चेतावनी। तुरंत निकलें।",
    "source": "government_alert",
    "timestamp": "2024-01-01T10:00:00Z"
  }'

# Test English
curl -X POST "http://localhost:8000/api/v1/process/single" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Tsunami warning issued for Chennai coast. Evacuate immediately.",
    "source": "twitter",
    "timestamp": "2024-01-01T10:00:00Z"
  }'

# Test Code-mixed (Hindi-English)
curl -X POST "http://localhost:8000/api/v1/process/single" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Mumbai beach pe bahut dangerous waves aa rahe hain! Please stay away 🌊",
    "source": "user_report",
    "timestamp": "2024-01-01T10:00:00Z"
  }'
```

**Expected Results:**
- ✅ Language is correctly detected
- ✅ Hazard type is identified (tsunami/high_waves)
- ✅ Location is extracted (Chennai/Mumbai)
- ✅ Urgency score is high (>0.7)

#### **Scenario 2: Non-Hazard Content Filtering**

Test with non-hazard content:

```bash
curl -X POST "http://localhost:8000/api/v1/process/single" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Beautiful sunset at Marina Beach today. Perfect weather for photography.",
    "source": "social_media",
    "timestamp": "2024-01-01T10:00:00Z"
  }'
```

**Expected Results:**
- ✅ `is_hazard` should be `false`
- ✅ `hazard_type` should be `"none"`
- ✅ Location still extracted (Marina Beach)

#### **Scenario 3: Misinformation Detection**

Test with suspicious content:

```bash
curl -X POST "http://localhost:8000/api/v1/process/single" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Scientists confirm aliens caused the tsunami! Share to warn everyone!",
    "source": "social_media",
    "timestamp": "2024-01-01T10:00:00Z"
  }'
```

**Expected Results:**
- ✅ Misinformation flags should be raised
- ✅ Confidence should indicate suspicious content

#### **Scenario 4: Batch Processing**

Test batch processing with multiple reports:

```bash
curl -X POST "http://localhost:8000/api/v1/process/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "reports": [
      {
        "text": "Tsunami warning for Chennai coast",
        "source": "twitter",
        "timestamp": "2024-01-01T10:00:00Z"
      },
      {
        "text": "High waves at Mumbai beach, very dangerous",
        "source": "user_report", 
        "timestamp": "2024-01-01T10:05:00Z"
      },
      {
        "text": "Beautiful weather in Goa today",
        "source": "social_media",
        "timestamp": "2024-01-01T10:10:00Z"
      }
    ]
  }'
```

**Expected Results:**
- ✅ All 3 reports processed
- ✅ 2 hazard reports identified
- ✅ 1 non-hazard report filtered
- ✅ Locations extracted correctly

### ✅ **Phase 7: Data Export Testing**

Test data export functionality:

```bash
# Test GeoJSON export
curl "http://localhost:8000/api/v1/export/geojson?hours=24"

# Test CSV export
curl "http://localhost:8000/api/v1/export/csv?hours=24"

# Test statistics
curl "http://localhost:8000/api/v1/statistics"
```

### ✅ **Phase 8: Feedback System Testing**

Test the operator feedback system:

```bash
curl -X POST "http://localhost:8000/api/v1/feedback" \
  -H "Content-Type: application/json" \
  -d '{
    "report_id": "test_report_001",
    "feedback_type": "correction",
    "original_prediction": {
      "is_hazard": true,
      "hazard_type": "tsunami"
    },
    "correct_prediction": {
      "is_hazard": false,
      "hazard_type": "none"
    },
    "operator_id": "test_operator",
    "comments": "False alarm - just high tide"
  }'
```

## 🚨 **Critical Issues to Check**

### **1. Model Loading Issues**
- ✅ All ML models load without errors
- ✅ Memory usage is reasonable (<4GB)
- ✅ Loading time is acceptable (<60 seconds)

### **2. Database Connectivity**
- ✅ MongoDB connection established
- ✅ Data can be written and read
- ✅ Collections are properly indexed

### **3. Error Handling**
- ✅ Graceful handling of malformed input
- ✅ Proper error messages for debugging
- ✅ System continues running after errors

### **4. Performance Benchmarks**
- ✅ Single report processing: <2 seconds
- ✅ Batch processing (10 reports): <5 seconds
- ✅ API response time: <1 second for health checks
- ✅ Memory stable under load

### **5. Security Validation**
- ✅ No SQL injection vulnerabilities
- ✅ Input validation prevents XSS
- ✅ API rate limiting works
- ✅ No sensitive data in logs

## 📊 **Performance Benchmarks**

### **Expected Performance Metrics:**

| Metric | Target | Acceptable | Requires Investigation |
|--------|--------|------------|----------------------|
| Single Report Processing | <1s | <2s | >2s |
| Batch Processing (10 items) | <3s | <5s | >5s |
| API Response Time | <500ms | <1s | >1s |
| Memory Usage | <2GB | <4GB | >4GB |
| Accuracy (Hazard Detection) | >95% | >90% | <90% |
| Location Extraction Accuracy | >90% | >85% | <85% |

## 🔍 **Common Issues & Solutions**

### **Issue: Models fail to load**
**Solution:** 
- Check internet connectivity for downloading models
- Verify sufficient disk space (>5GB)
- Ensure Python dependencies are installed

### **Issue: Database connection fails**
**Solution:**
- Verify MongoDB is running: `docker ps`
- Check connection string in .env file
- Ensure port 27017 is not blocked

### **Issue: Poor accuracy results**
**Solution:**
- Check if input text is properly cleaned
- Verify language detection is working
- Review model confidence scores

### **Issue: Slow performance**
**Solution:**
- Check system resources (CPU, RAM)
- Consider GPU acceleration for models
- Implement caching for repeated requests

## 🚀 **Ready for Integration Checklist**

Before declaring Model1 ready for team integration:

- [ ] ✅ All unit tests pass (>95% success rate)
- [ ] ✅ All API endpoints respond correctly
- [ ] ✅ Multilingual processing works for major Indian languages
- [ ] ✅ Performance meets benchmarks
- [ ] ✅ Error handling is robust
- [ ] ✅ Database operations are stable
- [ ] ✅ Docker containers start and run properly
- [ ] ✅ Documentation is complete and accurate
- [ ] ✅ Security validation passed
- [ ] ✅ Load testing completed successfully

## 📞 **Support & Troubleshooting**

If you encounter issues during testing:

1. **Check logs:** `docker-compose logs model-a-ml`
2. **Verify configuration:** Review .env file settings
3. **Monitor resources:** Check CPU, memory, and disk usage
4. **Test connectivity:** Ensure all services can communicate
5. **Review dependencies:** Verify all packages are installed correctly

---

**Once all tests pass, Model1 is ready for team integration! 🎉**