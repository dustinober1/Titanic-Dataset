#!/usr/bin/env python3
"""
Pytest Configuration and Shared Fixtures
========================================

This configuration file provides shared fixtures and settings for all tests
in the Titanic ML project. It ensures consistent test environments and 
provides commonly used test data and utilities.

Author: Enhanced Titanic ML Testing Framework
"""

import pytest
import numpy as np
import pandas as pd
import os
import tempfile
import shutil
from pathlib import Path
import warnings

# Suppress warnings during testing
warnings.filterwarnings('ignore')

# Set numpy random seed for reproducibility
np.random.seed(42)


@pytest.fixture(scope="session")
def test_data_dir():
    """Create a temporary directory for test data"""
    temp_dir = tempfile.mkdtemp(prefix="titanic_test_")
    yield temp_dir
    # Cleanup after all tests
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture(scope="session") 
def sample_titanic_dataset():
    """Create a comprehensive sample Titanic dataset for testing"""
    np.random.seed(42)
    n_samples = 200
    
    # Generate realistic passenger data
    passenger_ids = range(1, n_samples + 1)
    
    # Realistic survival distribution based on historical patterns
    pclass = np.random.choice([1, 2, 3], n_samples, p=[0.24, 0.21, 0.55])
    sex = np.random.choice(['male', 'female'], n_samples, p=[0.65, 0.35])
    age = np.random.gamma(2, 15).clip(0.1, 80)  # Realistic age distribution
    
    # Family structure
    sibsp = np.random.poisson(0.5, n_samples).clip(0, 8)
    parch = np.random.poisson(0.4, n_samples).clip(0, 6)
    
    # Fare based on class with some randomness
    fare_base = np.where(pclass == 1, 80, np.where(pclass == 2, 20, 10))
    fare = (fare_base * np.random.lognormal(0, 0.5)).clip(0, 500)
    
    # Embarked port
    embarked = np.random.choice(['S', 'C', 'Q'], n_samples, p=[0.72, 0.19, 0.09])
    
    # Survival based on historical patterns
    survival_prob = (
        0.15 +  # Base survival rate
        0.5 * (sex == 'female') +      # Female advantage
        0.2 * (pclass == 1) +          # First class advantage  
        0.1 * (pclass == 2) +          # Second class advantage
        0.15 * (age < 16) +            # Children advantage
        -0.1 * (age > 60)              # Elderly disadvantage
    )
    survived = np.random.binomial(1, np.clip(survival_prob, 0, 1), n_samples)
    
    # Generate realistic names
    male_names = [f"Smith, Mr. John_{i}" for i in range(n_samples)]
    female_names = [f"Johnson, Mrs. Mary_{i}" for i in range(n_samples)]
    names = [female_names[i] if sex[i] == 'female' else male_names[i] 
             for i in range(n_samples)]
    
    # Generate ticket numbers
    tickets = [f"TICKET_{i:05d}" for i in passenger_ids]
    
    # Generate cabin data (mostly missing, as in real data)
    cabins = [f"C{np.random.randint(1, 200)}" if np.random.random() < 0.3 else np.nan 
              for _ in range(n_samples)]
    
    return pd.DataFrame({
        'PassengerId': passenger_ids,
        'Survived': survived,
        'Pclass': pclass,
        'Name': names,
        'Sex': sex,
        'Age': age,
        'SibSp': sibsp,
        'Parch': parch,
        'Ticket': tickets,
        'Fare': fare,
        'Cabin': cabins,
        'Embarked': embarked
    })


@pytest.fixture
def minimal_titanic_data():
    """Create minimal valid Titanic data for quick tests"""
    return pd.DataFrame({
        'PassengerId': [1, 2, 3, 4, 5],
        'Survived': [0, 1, 1, 1, 0],
        'Pclass': [3, 1, 3, 1, 3],
        'Name': ['Smith, Mr. John', 'Johnson, Mrs. Mary', 'Brown, Miss. Sarah', 
                'Davis, Mr. James', 'Wilson, Mrs. Emma'],
        'Sex': ['male', 'female', 'female', 'male', 'female'],
        'Age': [22.0, 38.0, 26.0, 35.0, 29.0],
        'SibSp': [1, 1, 0, 1, 0],
        'Parch': [0, 0, 0, 0, 1],
        'Ticket': ['A001', 'B002', 'C003', 'D004', 'E005'],
        'Fare': [7.25, 71.28, 7.92, 53.10, 8.05],
        'Cabin': [np.nan, 'C85', np.nan, 'C123', 'G6'],
        'Embarked': ['S', 'C', 'S', 'S', 'Q']
    })


@pytest.fixture
def corrupted_data():
    """Create data with various corruption issues for testing error handling"""
    return pd.DataFrame({
        'PassengerId': [1, 2, 3, None, 5],  # Missing ID
        'Survived': [0, 1, -1, 1, 2],       # Invalid values
        'Pclass': [3, 1, 0, 1, 4],          # Invalid class
        'Name': ['Smith, Mr. John', '', 'Brown, Miss. Sarah', None, 'Wilson, Mrs. Emma'],
        'Sex': ['male', 'female', 'unknown', 'female', ''],  # Invalid gender
        'Age': [22.0, -5.0, 26.0, 150.0, np.nan],          # Invalid ages
        'SibSp': [1, -1, 0, 1, 20],         # Invalid family size
        'Parch': [0, 0, -1, 0, 15],         # Invalid family size
        'Ticket': ['A001', 'B002', '', 'D004', None],
        'Fare': [7.25, -10.0, 7.92, 1000.0, np.nan],       # Invalid fares
        'Cabin': [np.nan, 'C85', np.nan, 'INVALID', 'G6'],
        'Embarked': ['S', 'X', 'S', '', None]               # Invalid ports
    })


@pytest.fixture
def expected_data_schema():
    """Define the expected data schema for validation"""
    return {
        'required_columns': [
            'PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 
            'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'
        ],
        'column_types': {
            'PassengerId': ['int64', 'int32'],
            'Survived': ['int64', 'int32'], 
            'Pclass': ['int64', 'int32'],
            'Name': ['object'],
            'Sex': ['object'],
            'Age': ['float64', 'float32'],
            'SibSp': ['int64', 'int32'],
            'Parch': ['int64', 'int32'],
            'Ticket': ['object'],
            'Fare': ['float64', 'float32'],
            'Cabin': ['object'],
            'Embarked': ['object']
        },
        'value_constraints': {
            'Survived': [0, 1],
            'Pclass': [1, 2, 3],
            'Sex': ['male', 'female'],
            'Age': {'min': 0, 'max': 120},
            'SibSp': {'min': 0, 'max': 20},
            'Parch': {'min': 0, 'max': 20},
            'Fare': {'min': 0, 'max': 1000},
            'Embarked': ['S', 'C', 'Q']
        }
    }


@pytest.fixture
def trained_model_artifacts(tmp_path):
    """Create mock trained model artifacts for testing"""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    import joblib
    
    # Create simple mock model
    model = RandomForestClassifier(n_estimators=5, random_state=42)
    
    # Create mock training data
    X_mock = np.random.randn(100, 5)
    y_mock = np.random.randint(0, 2, 100)
    
    # Fit model
    model.fit(X_mock, y_mock)
    
    # Create scaler
    scaler = StandardScaler()
    scaler.fit(X_mock)
    
    # Create label encoders
    le_dict = {
        'Sex': LabelEncoder().fit(['male', 'female']),
        'Embarked': LabelEncoder().fit(['S', 'C', 'Q'])
    }
    
    # Save artifacts
    model_path = tmp_path / "test_model.pkl"
    scaler_path = tmp_path / "test_scaler.pkl"
    encoders_path = tmp_path / "test_encoders.pkl"
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(le_dict, encoders_path)
    
    return {
        'model': model,
        'scaler': scaler,
        'encoders': le_dict,
        'paths': {
            'model': model_path,
            'scaler': scaler_path,
            'encoders': encoders_path
        }
    }


@pytest.fixture
def performance_benchmarks():
    """Define performance benchmarks for model validation"""
    return {
        'minimum_accuracy': 0.70,
        'minimum_auc': 0.70,
        'maximum_training_time': 300,  # seconds
        'maximum_prediction_time': 0.1,  # seconds per prediction
        'maximum_model_size': 50,  # MB
        'minimum_cv_score': 0.65
    }


@pytest.fixture(scope="function")
def temp_model_dir(tmp_path):
    """Create temporary directory for model artifacts during tests"""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    return model_dir


@pytest.fixture
def mock_config():
    """Provide mock configuration for testing"""
    return {
        'data': {
            'raw_data_path': 'data/raw/Titanic-Dataset.csv',
            'processed_data_path': 'data/processed/',
            'validation_split': 0.2,
            'random_seed': 42
        },
        'model': {
            'algorithms': ['RandomForest', 'LogisticRegression', 'SVM'],
            'hyperparameter_tuning': True,
            'cross_validation_folds': 5,
            'scoring_metric': 'accuracy'
        },
        'testing': {
            'performance_threshold': 0.70,
            'outlier_threshold': 3.0,
            'missing_value_threshold': 0.25
        },
        'logging': {
            'level': 'INFO',
            'format': '%(asctime)s - %(levelname)s - %(message)s'
        }
    }


# Pytest configuration options
def pytest_configure(config):
    """Configure pytest settings"""
    # Add custom markers
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "regression: marks tests as regression tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically"""
    for item in items:
        # Mark slow tests
        if "slow" in item.nodeid:
            item.add_marker(pytest.mark.slow)
        
        # Mark integration tests
        if "integration" in item.nodeid.lower() or "end_to_end" in item.nodeid.lower():
            item.add_marker(pytest.mark.integration)
        
        # Mark unit tests (default)
        if not any(marker.name in ["integration", "slow"] for marker in item.iter_markers()):
            item.add_marker(pytest.mark.unit)


# Custom assertions
def assert_dataframe_equal(df1, df2, check_dtype=True, check_names=True):
    """Custom assertion for DataFrame equality with better error messages"""
    try:
        pd.testing.assert_frame_equal(df1, df2, check_dtype=check_dtype, check_names=check_names)
    except AssertionError as e:
        # Enhance error message with more details
        msg = f"DataFrames are not equal:\n{str(e)}\n"
        msg += f"DF1 shape: {df1.shape}, DF2 shape: {df2.shape}\n"
        if df1.shape == df2.shape:
            diff_mask = df1 != df2
            if diff_mask.any().any():
                msg += f"Differences found in columns: {diff_mask.any()[diff_mask.any()].index.tolist()}"
        raise AssertionError(msg)


def assert_model_performance(accuracy, auc, min_accuracy=0.7, min_auc=0.7):
    """Custom assertion for model performance validation"""
    assert accuracy >= min_accuracy, f"Model accuracy {accuracy:.4f} below minimum {min_accuracy}"
    assert auc >= min_auc, f"Model AUC {auc:.4f} below minimum {min_auc}"
    assert 0 <= accuracy <= 1, f"Invalid accuracy value: {accuracy}"
    assert 0 <= auc <= 1, f"Invalid AUC value: {auc}"


# Test utilities
class TestDataGenerator:
    """Utility class for generating test data with specific characteristics"""
    
    @staticmethod
    def create_data_with_missing_values(n_samples=100, missing_rate=0.1):
        """Create data with specified missing value rate"""
        np.random.seed(42)
        data = pd.DataFrame({
            'feature1': np.random.randn(n_samples),
            'feature2': np.random.randn(n_samples),
            'target': np.random.randint(0, 2, n_samples)
        })
        
        # Introduce missing values
        for col in data.columns:
            missing_mask = np.random.random(n_samples) < missing_rate
            data.loc[missing_mask, col] = np.nan
        
        return data
    
    @staticmethod
    def create_data_with_outliers(n_samples=100, outlier_rate=0.05):
        """Create data with specified outlier rate"""
        np.random.seed(42)
        n_outliers = int(n_samples * outlier_rate)
        
        # Normal data
        normal_data = np.random.randn(n_samples - n_outliers)
        
        # Outlier data
        outliers = np.random.choice([-5, 5], n_outliers) * np.random.randn(n_outliers)
        
        return pd.DataFrame({
            'feature': np.concatenate([normal_data, outliers]),
            'target': np.random.randint(0, 2, n_samples)
        })


# Make utilities available at module level
__all__ = [
    'assert_dataframe_equal',
    'assert_model_performance', 
    'TestDataGenerator'
]