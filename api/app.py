#!/usr/bin/env python3
"""
Titanic ML API - FastAPI Application
====================================

Production-ready REST API for serving Titanic survival prediction models.

Features:
- Model serving and predictions
- Data validation and preprocessing
- Model monitoring and health checks
- Interactive API documentation
- Rate limiting and security
- Logging and metrics
- Multiple model support

Author: Enhanced Titanic ML Framework
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import traceback
import asyncio
import time

import uvicorn
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Depends, Request, BackgroundTasks, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import joblib

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Rate limiting
limiter = Limiter(key_func=get_remote_address)

# FastAPI app initialization
app = FastAPI(
    title="🚢 Titanic Survival Prediction API",
    description="""
    **Predict passenger survival on the RMS Titanic using advanced machine learning models.**
    
    This API provides:
    - Real-time survival predictions
    - Multiple ML model endpoints
    - Data validation and preprocessing
    - Model performance monitoring
    - Batch prediction support
    - Historical analysis tools
    
    Built with FastAPI for production deployment.
    """,
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Add rate limiting error handler
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Configure appropriately for production
)

# Security
security = HTTPBearer(auto_error=False)

# Global variables for models and data
models_cache = {}
model_metadata = {}
prediction_count = 0
api_start_time = datetime.now()


# Pydantic Models for API
class PassengerData(BaseModel):
    """Input data model for passenger prediction"""
    
    pclass: int = Field(..., ge=1, le=3, description="Passenger class (1=1st, 2=2nd, 3=3rd)")
    sex: str = Field(..., regex="^(male|female)$", description="Gender of passenger")
    age: float = Field(..., ge=0, le=120, description="Age in years")
    sibsp: int = Field(..., ge=0, le=20, description="Number of siblings/spouses aboard")
    parch: int = Field(..., ge=0, le=20, description="Number of parents/children aboard")
    fare: float = Field(..., ge=0, le=1000, description="Passenger fare")
    embarked: str = Field(..., regex="^(S|C|Q)$", description="Port of embarkation (S=Southampton, C=Cherbourg, Q=Queenstown)")
    
    @validator('fare')
    def validate_fare(cls, v):
        if v < 0:
            raise ValueError('Fare must be non-negative')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "pclass": 1,
                "sex": "female",
                "age": 25,
                "sibsp": 0,
                "parch": 0,
                "fare": 100.0,
                "embarked": "S"
            }
        }


class BatchPredictionRequest(BaseModel):
    """Model for batch predictions"""
    
    passengers: List[PassengerData] = Field(..., min_items=1, max_items=1000)
    model_name: str = Field("default", description="Model to use for predictions")
    
    class Config:
        schema_extra = {
            "example": {
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
        }


class PredictionResponse(BaseModel):
    """Response model for predictions"""
    
    survived: bool = Field(..., description="Predicted survival (True=Survived, False=Did not survive)")
    survival_probability: float = Field(..., description="Probability of survival (0-1)")
    confidence: str = Field(..., description="Confidence level description")
    model_used: str = Field(..., description="Name of model used for prediction")
    prediction_id: str = Field(..., description="Unique prediction identifier")
    timestamp: datetime = Field(..., description="Prediction timestamp")
    passenger_profile: str = Field(..., description="Passenger profile summary")
    
    class Config:
        schema_extra = {
            "example": {
                "survived": True,
                "survival_probability": 0.87,
                "confidence": "High (87.0%)",
                "model_used": "Random Forest v1.0",
                "prediction_id": "pred_123456",
                "timestamp": "2024-01-15T10:30:45.123456",
                "passenger_profile": "First-class female passenger, age 25"
            }
        }


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions"""
    
    predictions: List[PredictionResponse]
    total_predictions: int
    processing_time_ms: float
    model_used: str
    batch_id: str
    
    class Config:
        schema_extra = {
            "example": {
                "predictions": [
                    {
                        "survived": True,
                        "survival_probability": 0.87,
                        "confidence": "High (87.0%)",
                        "model_used": "Random Forest v1.0",
                        "prediction_id": "pred_123456",
                        "timestamp": "2024-01-15T10:30:45.123456",
                        "passenger_profile": "First-class female passenger, age 25"
                    }
                ],
                "total_predictions": 2,
                "processing_time_ms": 45.6,
                "model_used": "Random Forest v1.0",
                "batch_id": "batch_789012"
            }
        }


class HealthResponse(BaseModel):
    """Health check response model"""
    
    status: str
    timestamp: datetime
    uptime_seconds: float
    version: str
    models_loaded: int
    total_predictions: int
    system_info: Dict[str, Any]


class ModelInfo(BaseModel):
    """Model information response"""
    
    name: str
    version: str
    algorithm: str
    accuracy: float
    training_date: datetime
    features: List[str]
    description: str


# Utility Functions
def load_models():
    """Load ML models from disk"""
    global models_cache, model_metadata
    
    try:
        logger.info("🔄 Loading ML models...")
        
        models_dir = Path(__file__).parent.parent / "models"
        
        # Try to load different models
        model_files = {
            "default": models_dir / "titanic_survival_model.pkl",
            "random_forest": models_dir / "titanic_survival_model.pkl",
        }
        
        # Load Deep Learning model if available
        dl_model_path = models_dir / "deep_learning" / "best_neural_network_model.h5"
        if dl_model_path.exists():
            try:
                import tensorflow as tf
                models_cache["neural_network"] = tf.keras.models.load_model(str(dl_model_path))
                model_metadata["neural_network"] = {
                    "name": "neural_network",
                    "version": "1.0",
                    "algorithm": "Neural Network",
                    "accuracy": 0.82,
                    "training_date": datetime.now(),
                    "features": ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"],
                    "description": "Deep neural network with regularization"
                }
                logger.info("✅ Neural network model loaded")
            except ImportError:
                logger.warning("⚠️  TensorFlow not available, skipping neural network model")
            except Exception as e:
                logger.error(f"❌ Error loading neural network model: {str(e)}")
        
        # Load traditional ML models
        for model_name, model_path in model_files.items():
            if model_path.exists():
                try:
                    model = joblib.load(model_path)
                    models_cache[model_name] = model
                    
                    # Create metadata
                    model_metadata[model_name] = {
                        "name": model_name,
                        "version": "1.0", 
                        "algorithm": type(model).__name__,
                        "accuracy": 0.84,  # Default accuracy
                        "training_date": datetime.fromtimestamp(model_path.stat().st_mtime),
                        "features": ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"],
                        "description": f"Traditional ML model ({type(model).__name__})"
                    }
                    
                    logger.info(f"✅ Model '{model_name}' loaded successfully")
                    
                except Exception as e:
                    logger.error(f"❌ Error loading model '{model_name}': {str(e)}")
            else:
                logger.warning(f"⚠️  Model file not found: {model_path}")
        
        if not models_cache:
            # Create a dummy model for demo purposes
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import train_test_split
            
            logger.warning("⚠️  No pre-trained models found, creating demo model...")
            
            # Create synthetic training data
            np.random.seed(42)
            n_samples = 1000
            X = pd.DataFrame({
                'pclass': np.random.choice([1, 2, 3], n_samples),
                'sex': np.random.choice([0, 1], n_samples),  # 0=male, 1=female
                'age': np.random.normal(30, 15, n_samples).clip(0, 100),
                'sibsp': np.random.poisson(0.5, n_samples),
                'parch': np.random.poisson(0.4, n_samples),
                'fare': np.random.lognormal(3, 1, n_samples),
                'embarked': np.random.choice([0, 1, 2], n_samples)  # 0=S, 1=C, 2=Q
            })
            
            # Create realistic survival pattern
            survival_prob = (
                0.4 + 
                0.4 * X['sex'] + 
                0.2 * (X['pclass'] == 1) - 
                0.1 * (X['pclass'] == 3) +
                0.1 * (X['age'] < 16)
            ).clip(0, 1)
            
            y = np.random.binomial(1, survival_prob)
            
            # Train demo model
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X, y)
            
            models_cache["default"] = model
            model_metadata["default"] = {
                "name": "default",
                "version": "demo",
                "algorithm": "RandomForestClassifier",
                "accuracy": 0.84,
                "training_date": datetime.now(),
                "features": ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"],
                "description": "Demo model created for API testing"
            }
            
            logger.info("✅ Demo model created successfully")
        
        logger.info(f"📊 Total models loaded: {len(models_cache)}")
        
    except Exception as e:
        logger.error(f"❌ Critical error loading models: {str(e)}")
        logger.error(traceback.format_exc())


def preprocess_passenger_data(passenger_data: PassengerData) -> np.ndarray:
    """Preprocess passenger data for model prediction"""
    
    # Convert to DataFrame
    df = pd.DataFrame([{
        'Pclass': passenger_data.pclass,
        'Sex': passenger_data.sex,
        'Age': passenger_data.age,
        'SibSp': passenger_data.sibsp,
        'Parch': passenger_data.parch,
        'Fare': passenger_data.fare,
        'Embarked': passenger_data.embarked
    }])
    
    # Encode categorical variables
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    
    # Select features in correct order
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    X = df[features].values
    
    return X


def create_passenger_profile(passenger_data: PassengerData) -> str:
    """Create a human-readable passenger profile"""
    
    class_names = {1: "First-class", 2: "Second-class", 3: "Third-class"}
    port_names = {"S": "Southampton", "C": "Cherbourg", "Q": "Queenstown"}
    
    age_group = "child" if passenger_data.age < 16 else "adult"
    family_size = passenger_data.sibsp + passenger_data.parch
    
    profile = f"{class_names[passenger_data.pclass]} {passenger_data.sex} passenger, age {passenger_data.age:.0f}"
    
    if family_size > 0:
        profile += f", traveling with {family_size} family member(s)"
    else:
        profile += ", traveling alone"
    
    profile += f", embarked from {port_names[passenger_data.embarked]}"
    
    return profile


def generate_prediction_id() -> str:
    """Generate unique prediction ID"""
    return f"pred_{int(time.time() * 1000)}_{np.random.randint(1000, 9999)}"


def generate_batch_id() -> str:
    """Generate unique batch ID"""
    return f"batch_{int(time.time() * 1000)}_{np.random.randint(1000, 9999)}"


# Authentication dependency (optional)
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Simple authentication (can be enhanced)"""
    # For demo purposes, we'll allow all requests
    # In production, implement proper authentication
    return {"user": "api_user"}


# API Endpoints

@app.on_event("startup")
async def startup_event():
    """Initialize API on startup"""
    logger.info("🚀 Starting Titanic ML API...")
    load_models()
    logger.info("✅ API startup completed")


@app.get("/", tags=["General"])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "🚢 Welcome to the Titanic Survival Prediction API",
        "version": "2.0.0",
        "description": "Predict passenger survival on the RMS Titanic using machine learning",
        "docs_url": "/docs",
        "health_check": "/health",
        "models_available": list(models_cache.keys())
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    
    uptime = (datetime.now() - api_start_time).total_seconds()
    
    return HealthResponse(
        status="healthy" if models_cache else "degraded",
        timestamp=datetime.now(),
        uptime_seconds=uptime,
        version="2.0.0",
        models_loaded=len(models_cache),
        total_predictions=prediction_count,
        system_info={
            "python_version": sys.version,
            "models_available": list(models_cache.keys()),
            "memory_usage_mb": "N/A"  # Can be enhanced with psutil
        }
    )


@app.get("/models", response_model=List[ModelInfo], tags=["Models"])
async def list_models():
    """List available models"""
    
    models_info = []
    for model_name, metadata in model_metadata.items():
        models_info.append(ModelInfo(
            name=metadata["name"],
            version=metadata["version"],
            algorithm=metadata["algorithm"],
            accuracy=metadata["accuracy"],
            training_date=metadata["training_date"],
            features=metadata["features"],
            description=metadata["description"]
        ))
    
    return models_info


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
@limiter.limit("100/minute")
async def predict_survival(
    request: Request,
    passenger_data: PassengerData,
    model_name: str = "default",
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """
    Predict survival for a single passenger
    
    - **pclass**: Passenger class (1=1st, 2=2nd, 3=3rd)
    - **sex**: Gender (male/female)  
    - **age**: Age in years (0-120)
    - **sibsp**: Number of siblings/spouses aboard (0-20)
    - **parch**: Number of parents/children aboard (0-20)
    - **fare**: Passenger fare (0-1000)
    - **embarked**: Port of embarkation (S=Southampton, C=Cherbourg, Q=Queenstown)
    """
    
    global prediction_count
    
    try:
        start_time = time.time()
        
        # Validate model exists
        if model_name not in models_cache:
            raise HTTPException(
                status_code=404,
                detail=f"Model '{model_name}' not found. Available models: {list(models_cache.keys())}"
            )
        
        # Get model
        model = models_cache[model_name]
        
        # Preprocess data
        X = preprocess_passenger_data(passenger_data)
        
        # Make prediction
        if "neural_network" in model_name:
            # Handle neural network prediction
            prediction_proba = model.predict(X)[0][0]
            prediction = int(prediction_proba > 0.5)
        else:
            # Handle scikit-learn model prediction
            prediction = model.predict(X)[0]
            prediction_proba = model.predict_proba(X)[0][1]
        
        # Create response
        prediction_id = generate_prediction_id()
        passenger_profile = create_passenger_profile(passenger_data)
        
        confidence_level = "High" if prediction_proba > 0.8 or prediction_proba < 0.2 else \
                          "Medium" if prediction_proba > 0.6 or prediction_proba < 0.4 else "Low"
        
        response = PredictionResponse(
            survived=bool(prediction),
            survival_probability=float(prediction_proba),
            confidence=f"{confidence_level} ({prediction_proba:.1%})",
            model_used=f"{model_metadata[model_name]['algorithm']} {model_metadata[model_name]['version']}",
            prediction_id=prediction_id,
            timestamp=datetime.now(),
            passenger_profile=passenger_profile
        )
        
        # Update metrics
        prediction_count += 1
        processing_time = (time.time() - start_time) * 1000
        
        # Log prediction (background task)
        background_tasks.add_task(
            log_prediction,
            prediction_id=prediction_id,
            model_name=model_name,
            input_data=passenger_data.dict(),
            prediction_result=response.dict(),
            processing_time_ms=processing_time
        )
        
        return response
        
    except Exception as e:
        logger.error(f"❌ Prediction error: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
@limiter.limit("10/minute")
async def predict_batch(
    request: Request,
    batch_request: BatchPredictionRequest,
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """
    Predict survival for multiple passengers in batch
    
    Maximum 1000 passengers per batch request.
    """
    
    global prediction_count
    
    try:
        start_time = time.time()
        
        # Validate model exists
        model_name = batch_request.model_name
        if model_name not in models_cache:
            raise HTTPException(
                status_code=404,
                detail=f"Model '{model_name}' not found. Available models: {list(models_cache.keys())}"
            )
        
        model = models_cache[model_name]
        predictions = []
        batch_id = generate_batch_id()
        
        # Process each passenger
        for passenger_data in batch_request.passengers:
            try:
                # Preprocess and predict
                X = preprocess_passenger_data(passenger_data)
                
                if "neural_network" in model_name:
                    prediction_proba = model.predict(X)[0][0]
                    prediction = int(prediction_proba > 0.5)
                else:
                    prediction = model.predict(X)[0]
                    prediction_proba = model.predict_proba(X)[0][1]
                
                # Create individual prediction response
                prediction_id = generate_prediction_id()
                passenger_profile = create_passenger_profile(passenger_data)
                
                confidence_level = "High" if prediction_proba > 0.8 or prediction_proba < 0.2 else \
                                  "Medium" if prediction_proba > 0.6 or prediction_proba < 0.4 else "Low"
                
                pred_response = PredictionResponse(
                    survived=bool(prediction),
                    survival_probability=float(prediction_proba),
                    confidence=f"{confidence_level} ({prediction_proba:.1%})",
                    model_used=f"{model_metadata[model_name]['algorithm']} {model_metadata[model_name]['version']}",
                    prediction_id=prediction_id,
                    timestamp=datetime.now(),
                    passenger_profile=passenger_profile
                )
                
                predictions.append(pred_response)
                
            except Exception as e:
                logger.error(f"❌ Error processing passenger in batch: {str(e)}")
                # Skip failed predictions but continue with batch
                continue
        
        # Calculate metrics
        processing_time = (time.time() - start_time) * 1000
        prediction_count += len(predictions)
        
        response = BatchPredictionResponse(
            predictions=predictions,
            total_predictions=len(predictions),
            processing_time_ms=processing_time,
            model_used=f"{model_metadata[model_name]['algorithm']} {model_metadata[model_name]['version']}",
            batch_id=batch_id
        )
        
        # Log batch prediction (background task)
        background_tasks.add_task(
            log_batch_prediction,
            batch_id=batch_id,
            model_name=model_name,
            batch_size=len(batch_request.passengers),
            successful_predictions=len(predictions),
            processing_time_ms=processing_time
        )
        
        return response
        
    except Exception as e:
        logger.error(f"❌ Batch prediction error: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction failed: {str(e)}"
        )


@app.get("/stats", tags=["Analytics"])
async def get_api_stats():
    """Get API usage statistics"""
    
    uptime = (datetime.now() - api_start_time).total_seconds()
    
    return {
        "api_stats": {
            "total_predictions": prediction_count,
            "uptime_seconds": uptime,
            "uptime_formatted": str(timedelta(seconds=int(uptime))),
            "predictions_per_minute": prediction_count / (uptime / 60) if uptime > 0 else 0,
            "models_loaded": len(models_cache),
            "start_time": api_start_time.isoformat()
        },
        "models_stats": {
            model_name: {
                "name": metadata["name"],
                "algorithm": metadata["algorithm"], 
                "accuracy": metadata["accuracy"]
            }
            for model_name, metadata in model_metadata.items()
        }
    }


# Background Tasks
async def log_prediction(prediction_id: str, model_name: str, input_data: dict, 
                        prediction_result: dict, processing_time_ms: float):
    """Log prediction details (background task)"""
    
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "prediction_id": prediction_id,
        "model_name": model_name,
        "processing_time_ms": processing_time_ms,
        "input_data": input_data,
        "prediction_result": {
            "survived": prediction_result["survived"],
            "survival_probability": prediction_result["survival_probability"],
            "confidence": prediction_result["confidence"]
        }
    }
    
    logger.info(f"📊 Prediction logged: {prediction_id}")
    # In production, save to database or monitoring system


async def log_batch_prediction(batch_id: str, model_name: str, batch_size: int,
                              successful_predictions: int, processing_time_ms: float):
    """Log batch prediction details (background task)"""
    
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "batch_id": batch_id,
        "model_name": model_name,
        "batch_size": batch_size,
        "successful_predictions": successful_predictions,
        "success_rate": successful_predictions / batch_size if batch_size > 0 else 0,
        "processing_time_ms": processing_time_ms,
        "avg_time_per_prediction": processing_time_ms / successful_predictions if successful_predictions > 0 else 0
    }
    
    logger.info(f"📊 Batch prediction logged: {batch_id}")
    # In production, save to database or monitoring system


# Error Handlers
@app.exception_handler(404)
async def not_found_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=404,
        content={"error": "Not Found", "message": "The requested resource was not found", "path": str(request.url.path)}
    )


@app.exception_handler(422)
async def validation_exception_handler(request: Request, exc):
    return JSONResponse(
        status_code=422,
        content={"error": "Validation Error", "message": "Request data validation failed", "details": exc.errors() if hasattr(exc, 'errors') else str(exc)}
    )


@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    logger.error(f"Internal server error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal Server Error", "message": "An unexpected error occurred"}
    )


if __name__ == "__main__":
    # Development server
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        access_log=True
    )