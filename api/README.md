# 🚢 Titanic ML API

Production-ready REST API for serving Titanic survival prediction models built with FastAPI.

## Features

### 🚀 Core Functionality
- **Real-time Predictions**: Single passenger survival predictions
- **Batch Processing**: Process up to 1000 passengers at once
- **Multiple Models**: Support for different ML algorithms
- **Data Validation**: Comprehensive input validation and preprocessing
- **Interactive Documentation**: Auto-generated API docs with Swagger UI

### 🛡️ Production Features
- **Rate Limiting**: Prevents API abuse
- **Authentication**: HTTP Bearer token support (configurable)
- **Error Handling**: Comprehensive error responses
- **Logging**: Detailed request and prediction logging
- **Health Checks**: Monitor API status and performance
- **CORS Support**: Cross-origin resource sharing
- **Background Tasks**: Async logging and monitoring

### 📊 Monitoring & Analytics
- **Usage Statistics**: API usage metrics and trends  
- **Model Performance**: Track prediction accuracy and speed
- **System Health**: Uptime, memory usage, and diagnostics

## Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/yourusername/Titanic-Dataset.git
cd Titanic-Dataset/api

# Install dependencies
pip install -r requirements.txt

# Or install with base requirements
pip install -r ../requirements/base.txt
pip install fastapi uvicorn slowapi
```

### 2. Start the API

```bash
# Development server
python app.py

# Or with uvicorn directly
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Access the API

- **API Base URL**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

## API Endpoints

### Core Endpoints

#### `GET /` - API Information
Returns basic API information and available models.

#### `GET /health` - Health Check  
System health status, uptime, and performance metrics.

#### `GET /models` - List Models
List all available ML models with details.

#### `POST /predict` - Single Prediction
Predict survival for one passenger.

**Request Body:**
```json
{
  "pclass": 1,
  "sex": "female", 
  "age": 25,
  "sibsp": 0,
  "parch": 0,
  "fare": 100.0,
  "embarked": "S"
}
```

**Response:**
```json
{
  "survived": true,
  "survival_probability": 0.87,
  "confidence": "High (87.0%)",
  "model_used": "Random Forest v1.0",
  "prediction_id": "pred_123456",
  "timestamp": "2024-01-15T10:30:45.123456",
  "passenger_profile": "First-class female passenger, age 25"
}
```

#### `POST /predict/batch` - Batch Predictions
Process multiple passengers (up to 1000 per request).

**Request Body:**
```json
{
  "passengers": [
    {
      "pclass": 1,
      "sex": "female",
      "age": 25,
      "sibsp": 0, 
      "parch": 0,
      "fare": 100.0,
      "embarked": "S"
    },
    {
      "pclass": 3,
      "sex": "male",
      "age": 30,
      "sibsp": 1,
      "parch": 2,
      "fare": 15.0,
      "embarked": "S"
    }
  ],
  "model_name": "default"
}
```

#### `GET /stats` - API Statistics
Usage statistics, performance metrics, and model information.

### Authentication (Optional)

The API supports HTTP Bearer token authentication. To enable:

1. Set authentication requirements in production
2. Include Authorization header: `Authorization: Bearer <your-token>`

## Data Validation

### Input Parameters

All passenger data is validated:

- **pclass**: Integer 1-3 (passenger class)
- **sex**: String "male" or "female"  
- **age**: Float 0-120 (age in years)
- **sibsp**: Integer 0-20 (siblings/spouses aboard)
- **parch**: Integer 0-20 (parents/children aboard)
- **fare**: Float 0-1000 (passenger fare)
- **embarked**: String "S", "C", or "Q" (embarkation port)

### Error Responses

The API returns detailed error messages for invalid inputs:

```json
{
  "error": "Validation Error",
  "message": "Request data validation failed", 
  "details": [
    {
      "loc": ["age"],
      "msg": "ensure this value is less than or equal to 120",
      "type": "value_error.number.not_le"
    }
  ]
}
```

## Rate Limiting

API includes rate limiting to prevent abuse:

- **Single Predictions**: 100 requests/minute per IP
- **Batch Predictions**: 10 requests/minute per IP  

Rate limit headers are included in responses:
- `X-RateLimit-Limit`: Request limit
- `X-RateLimit-Remaining`: Remaining requests
- `X-RateLimit-Reset`: Reset time

## Models

### Supported Models

1. **Random Forest** (default)
   - Algorithm: RandomForestClassifier
   - Accuracy: ~84%
   - Fast inference, good interpretability

2. **Neural Network** (if TensorFlow available)
   - Algorithm: Deep Neural Network
   - Accuracy: ~82%
   - Advanced feature learning

### Model Selection

Specify model in requests:
```bash
curl -X POST "http://localhost:8000/predict?model_name=neural_network" \
     -H "Content-Type: application/json" \
     -d '{"pclass": 1, "sex": "female", "age": 25, ...}'
```

## Usage Examples

### Python Client

```python
import requests

# Single prediction
response = requests.post(
    "http://localhost:8000/predict",
    json={
        "pclass": 1,
        "sex": "female",
        "age": 25, 
        "sibsp": 0,
        "parch": 0,
        "fare": 100.0,
        "embarked": "S"
    }
)

result = response.json()
print(f"Survived: {result['survived']}")
print(f"Probability: {result['survival_probability']:.2%}")
```

### cURL Examples

```bash
# Health check
curl http://localhost:8000/health

# Single prediction
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "pclass": 3,
       "sex": "male", 
       "age": 22,
       "sibsp": 1,
       "parch": 0,
       "fare": 7.25,
       "embarked": "S"
     }'

# List models
curl http://localhost:8000/models

# API statistics
curl http://localhost:8000/stats
```

### JavaScript/Fetch

```javascript
// Single prediction
const prediction = await fetch('http://localhost:8000/predict', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    pclass: 1,
    sex: 'female',
    age: 25,
    sibsp: 0,
    parch: 0, 
    fare: 100.0,
    embarked: 'S'
  })
});

const result = await prediction.json();
console.log(`Survival probability: ${result.survival_probability}`);
```

## Configuration

### Environment Variables

```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# Rate Limiting  
RATE_LIMIT_ENABLED=true
RATE_LIMIT_PER_MINUTE=100

# Authentication
AUTH_ENABLED=false
SECRET_KEY=your-secret-key

# Logging
LOG_LEVEL=INFO
LOG_FILE=api.log

# Model Configuration
DEFAULT_MODEL=random_forest
MODEL_PATH=../models/
```

### Production Deployment

```bash
# With Gunicorn
pip install gunicorn
gunicorn app:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000

# With Docker
docker build -t titanic-api .
docker run -p 8000:8000 titanic-api

# With systemd service
sudo systemctl start titanic-api
sudo systemctl enable titanic-api
```

## Performance

### Benchmarks

- **Single Prediction**: ~20-50ms response time
- **Batch Prediction**: ~5-10ms per passenger
- **Memory Usage**: ~200MB baseline + model memory
- **Throughput**: ~1000 predictions/second (with proper hardware)

### Optimization Tips

1. **Use Batch Predictions** for multiple passengers
2. **Enable Model Caching** in production
3. **Configure Rate Limiting** appropriately
4. **Use Async Processing** for large batches
5. **Monitor Memory Usage** with large models

## Monitoring

### Health Checks

The `/health` endpoint provides:
- API status and uptime
- Models loaded status  
- Total predictions processed
- System information

### Logging

All requests and predictions are logged:
```
2024-01-15 10:30:45,123 - INFO - Prediction logged: pred_123456
2024-01-15 10:30:45,124 - INFO - Model: random_forest, Time: 23.4ms
```

### Metrics (Future Enhancement)

Integration with monitoring systems:
- Prometheus metrics endpoint
- Grafana dashboards
- Custom alerting rules

## Security

### Best Practices

1. **Enable Authentication** in production
2. **Use HTTPS** for all communications
3. **Configure CORS** appropriately  
4. **Set Rate Limits** to prevent abuse
5. **Validate All Inputs** thoroughly
6. **Log Security Events** for monitoring

### Production Checklist

- [ ] Enable authentication
- [ ] Configure HTTPS/TLS
- [ ] Set appropriate CORS policies
- [ ] Configure rate limiting
- [ ] Set up monitoring and alerting
- [ ] Enable request logging
- [ ] Use environment variables for secrets
- [ ] Set up proper error handling

## Testing

```bash
# Install test dependencies
pip install pytest pytest-asyncio httpx

# Run API tests
pytest tests/test_api.py -v

# Load testing
pip install locust
locust -f tests/load_test.py --host=http://localhost:8000
```

## Troubleshooting

### Common Issues

1. **Models Not Loading**
   - Check model files exist in `/models/` directory
   - Verify file permissions
   - Check model compatibility

2. **Rate Limit Exceeded**
   - Reduce request frequency
   - Use batch predictions for multiple passengers
   - Contact admin for limit increases

3. **Validation Errors**
   - Check input data format
   - Verify all required fields are provided
   - Ensure values are within valid ranges

### Debug Mode

Enable debug logging:
```python
import logging
logging.getLogger().setLevel(logging.DEBUG)
```

## Contributing

1. Fork the repository
2. Create a feature branch  
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## API Documentation

For complete API documentation, visit http://localhost:8000/docs after starting the server.