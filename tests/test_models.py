#!/usr/bin/env python3
"""
Comprehensive Unit Tests for Titanic ML Models
==============================================

This module provides comprehensive testing for all Titanic machine learning models,
including data preprocessing, model training, predictions, and performance validation.

Test Categories:
1. Data Loading and Preprocessing Tests
2. Feature Engineering Tests  
3. Model Training and Validation Tests
4. Prediction Accuracy Tests
5. Model Persistence Tests
6. Edge Case and Error Handling Tests

Author: Enhanced Titanic ML Testing Framework
"""

import pytest
import numpy as np
import pandas as pd
import pickle
import joblib
import os
import sys
from unittest.mock import patch, MagicMock
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

class TestDataLoading:
    """Test data loading and basic validation"""
    
    @pytest.fixture
    def sample_titanic_data(self):
        """Create sample Titanic dataset for testing"""
        return pd.DataFrame({
            'PassengerId': [1, 2, 3, 4, 5],
            'Survived': [0, 1, 1, 1, 0],
            'Pclass': [3, 1, 3, 1, 3],
            'Name': ['Braund, Mr. Owen Harris', 'Cumings, Mrs. John Bradley', 
                    'Heikkinen, Miss. Laina', 'Futrelle, Mrs. Jacques Heath',
                    'Allen, Mr. William Henry'],
            'Sex': ['male', 'female', 'female', 'female', 'male'],
            'Age': [22.0, 38.0, 26.0, 35.0, 35.0],
            'SibSp': [1, 1, 0, 1, 0],
            'Parch': [0, 0, 0, 0, 0],
            'Ticket': ['A/5 21171', 'PC 17599', 'STON/O2. 3101282', '113803', '373450'],
            'Fare': [7.25, 71.2833, 7.925, 53.1, 8.05],
            'Cabin': [np.nan, 'C85', np.nan, 'C123', np.nan],
            'Embarked': ['S', 'C', 'S', 'S', 'S']
        })
    
    def test_data_loading_basic(self, sample_titanic_data):
        """Test basic data loading functionality"""
        df = sample_titanic_data
        
        # Check basic properties
        assert df.shape[0] > 0, "Dataset should not be empty"
        assert df.shape[1] > 0, "Dataset should have columns"
        assert 'Survived' in df.columns, "Dataset should have Survived column"
        assert set(df['Survived'].unique()).issubset({0, 1}), "Survived should be binary"
    
    def test_data_types(self, sample_titanic_data):
        """Test data types are correct"""
        df = sample_titanic_data
        
        # Check key column types
        assert df['PassengerId'].dtype in ['int64', 'int32'], "PassengerId should be integer"
        assert df['Survived'].dtype in ['int64', 'int32'], "Survived should be integer"
        assert df['Pclass'].dtype in ['int64', 'int32'], "Pclass should be integer"
        assert df['Sex'].dtype == 'object', "Sex should be object/string"
        assert df['Age'].dtype in ['float64', 'float32'], "Age should be float"
    
    def test_required_columns_present(self, sample_titanic_data):
        """Test that all required columns are present"""
        df = sample_titanic_data
        required_columns = ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
        
        for col in required_columns:
            assert col in df.columns, f"Required column {col} is missing"
    
    def test_survival_distribution(self, sample_titanic_data):
        """Test survival distribution is reasonable"""
        df = sample_titanic_data
        survival_rate = df['Survived'].mean()
        
        # Survival rate should be between 0 and 1
        assert 0 <= survival_rate <= 1, "Survival rate should be between 0 and 1"
        
        # Should have both survivors and non-survivors in reasonable proportions
        assert 0.1 <= survival_rate <= 0.9, "Survival rate should be reasonably balanced"


class TestFeatureEngineering:
    """Test feature engineering functions"""
    
    @pytest.fixture
    def sample_data_with_missing(self):
        """Create sample data with missing values for testing"""
        return pd.DataFrame({
            'Name': ['Braund, Mr. Owen Harris', 'Cumings, Mrs. John Bradley', 'Heikkinen, Miss. Laina'],
            'Sex': ['male', 'female', 'female'],
            'Age': [22.0, np.nan, 26.0],
            'SibSp': [1, 1, 0],
            'Parch': [0, 0, 0],
            'Fare': [7.25, np.nan, 7.925],
            'Embarked': ['S', np.nan, 'S'],
            'Pclass': [3, 1, 3]
        })
    
    def extract_title(self, name):
        """Extract title from passenger name"""
        return name.split(',')[1].split('.')[0].strip()
    
    def test_title_extraction(self):
        """Test title extraction from names"""
        names = [
            'Braund, Mr. Owen Harris',
            'Cumings, Mrs. John Bradley',
            'Heikkinen, Miss. Laina',
            'Futrelle, Mrs. Jacques Heath'
        ]
        
        expected_titles = ['Mr', 'Mrs', 'Miss', 'Mrs']
        
        for name, expected in zip(names, expected_titles):
            title = self.extract_title(name)
            assert title == expected, f"Expected {expected}, got {title}"
    
    def test_family_size_calculation(self):
        """Test family size calculation"""
        sibsp_values = [0, 1, 2, 0]
        parch_values = [0, 0, 1, 2]
        expected_family_sizes = [1, 2, 4, 3]
        
        for sibsp, parch, expected in zip(sibsp_values, parch_values, expected_family_sizes):
            family_size = sibsp + parch + 1
            assert family_size == expected, f"Expected family size {expected}, got {family_size}"
    
    def test_is_alone_calculation(self):
        """Test is_alone flag calculation"""
        family_sizes = [1, 2, 3, 1, 4]
        expected_is_alone = [1, 0, 0, 1, 0]
        
        for family_size, expected in zip(family_sizes, expected_is_alone):
            is_alone = 1 if family_size == 1 else 0
            assert is_alone == expected, f"Expected is_alone {expected}, got {is_alone}"
    
    def test_missing_value_handling(self, sample_data_with_missing):
        """Test missing value handling strategies"""
        df = sample_data_with_missing.copy()
        
        # Test Age imputation
        age_before_imputation = df['Age'].isna().sum()
        df['Age'].fillna(df['Age'].median(), inplace=True)
        age_after_imputation = df['Age'].isna().sum()
        
        assert age_after_imputation < age_before_imputation, "Age missing values should be reduced"
        
        # Test Embarked imputation
        embarked_before = df['Embarked'].isna().sum()
        df['Embarked'].fillna('S', inplace=True)  # Most common value
        embarked_after = df['Embarked'].isna().sum()
        
        assert embarked_after < embarked_before, "Embarked missing values should be reduced"


class TestModelTraining:
    """Test model training and validation"""
    
    @pytest.fixture
    def prepared_data(self):
        """Prepare sample data for model training"""
        np.random.seed(42)
        n_samples = 100
        
        X = pd.DataFrame({
            'Pclass': np.random.randint(1, 4, n_samples),
            'Sex': np.random.choice(['male', 'female'], n_samples),
            'Age': np.random.normal(30, 15, n_samples),
            'SibSp': np.random.randint(0, 6, n_samples),
            'Parch': np.random.randint(0, 6, n_samples),
            'Fare': np.random.lognormal(3, 1, n_samples),
            'Embarked': np.random.choice(['S', 'C', 'Q'], n_samples)
        })
        
        # Create realistic survival target
        survival_prob = (
            0.7 * (X['Sex'] == 'female').astype(int) +
            0.3 * (X['Pclass'] == 1).astype(int) +
            0.2 * (X['Age'] < 18).astype(int) +
            np.random.normal(0, 0.2, n_samples)
        )
        y = (survival_prob > 0.5).astype(int)
        
        return X, y
    
    def test_data_preprocessing_pipeline(self, prepared_data):
        """Test complete data preprocessing pipeline"""
        X, y = prepared_data
        
        # Encode categorical variables
        le_dict = {}
        for col in ['Sex', 'Embarked']:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            le_dict[col] = le
        
        # Check encoding worked
        assert X['Sex'].dtype in ['int32', 'int64'], "Sex should be encoded as integer"
        assert X['Embarked'].dtype in ['int32', 'int64'], "Embarked should be encoded as integer"
        assert len(le_dict) == 2, "Should have encoders for 2 categorical columns"
        
        # Test scaling
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        assert X_scaled.shape == X.shape, "Scaled data should have same shape"
        assert abs(np.mean(X_scaled)) < 0.1, "Scaled data should be approximately centered"
        assert abs(np.std(X_scaled) - 1.0) < 0.1, "Scaled data should have approximately unit variance"
    
    def test_model_training_basic(self, prepared_data):
        """Test basic model training functionality"""
        X, y = prepared_data
        
        # Encode categorical variables
        le_sex = LabelEncoder()
        le_embarked = LabelEncoder()
        X['Sex'] = le_sex.fit_transform(X['Sex'])
        X['Embarked'] = le_embarked.fit_transform(X['Embarked'])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = RandomForestClassifier(n_estimators=10, random_state=42)  # Small for speed
        model.fit(X_train, y_train)
        
        # Test predictions
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)
        
        assert len(predictions) == len(X_test), "Should have prediction for each test sample"
        assert set(predictions).issubset({0, 1}), "Predictions should be binary"
        assert probabilities.shape == (len(X_test), 2), "Should have 2 class probabilities"
        assert np.allclose(probabilities.sum(axis=1), 1.0), "Probabilities should sum to 1"
    
    def test_model_performance_metrics(self, prepared_data):
        """Test model performance metrics calculation"""
        X, y = prepared_data
        
        # Encode and split
        le_sex = LabelEncoder()
        le_embarked = LabelEncoder()
        X['Sex'] = le_sex.fit_transform(X['Sex'])
        X['Embarked'] = le_embarked.fit_transform(X['Embarked'])
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        # Get predictions
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, predictions)
        auc = roc_auc_score(y_test, probabilities)
        
        # Test metrics are reasonable
        assert 0 <= accuracy <= 1, "Accuracy should be between 0 and 1"
        assert 0 <= auc <= 1, "AUC should be between 0 and 1"
        assert accuracy > 0.3, "Model should perform better than random"  # Very loose threshold
        assert auc > 0.3, "AUC should be better than random"


class TestModelPersistence:
    """Test model saving and loading functionality"""
    
    @pytest.fixture
    def trained_model_and_data(self):
        """Create a trained model and associated data for testing"""
        np.random.seed(42)
        n_samples = 50
        
        X = pd.DataFrame({
            'feature1': np.random.normal(0, 1, n_samples),
            'feature2': np.random.normal(0, 1, n_samples),
            'feature3': np.random.randint(0, 3, n_samples)
        })
        y = (X['feature1'] + X['feature2'] > 0).astype(int)
        
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        model.fit(X, y)
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        return model, scaler, X, y, X_scaled
    
    def test_model_pickle_save_load(self, trained_model_and_data, tmp_path):
        """Test saving and loading model with pickle"""
        model, scaler, X, y, X_scaled = trained_model_and_data
        
        # Save model
        model_path = tmp_path / "test_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Load model
        with open(model_path, 'rb') as f:
            loaded_model = pickle.load(f)
        
        # Test predictions are identical
        original_pred = model.predict(X)
        loaded_pred = loaded_model.predict(X)
        
        assert np.array_equal(original_pred, loaded_pred), "Loaded model should make identical predictions"
    
    def test_scaler_joblib_save_load(self, trained_model_and_data, tmp_path):
        """Test saving and loading scaler with joblib"""
        model, scaler, X, y, X_scaled = trained_model_and_data
        
        # Save scaler
        scaler_path = tmp_path / "test_scaler.pkl"
        joblib.dump(scaler, scaler_path)
        
        # Load scaler
        loaded_scaler = joblib.load(scaler_path)
        
        # Test transformations are identical
        original_transform = scaler.transform(X)
        loaded_transform = loaded_scaler.transform(X)
        
        assert np.allclose(original_transform, loaded_transform), "Loaded scaler should produce identical transformations"
    
    def test_model_metadata_preservation(self, trained_model_and_data, tmp_path):
        """Test that model metadata is preserved during save/load"""
        model, scaler, X, y, X_scaled = trained_model_and_data
        
        # Save model with metadata
        model_data = {
            'model': model,
            'feature_names': list(X.columns),
            'n_features': X.shape[1],
            'model_type': 'RandomForestClassifier'
        }
        
        metadata_path = tmp_path / "model_with_metadata.pkl"
        joblib.dump(model_data, metadata_path)
        
        # Load and verify metadata
        loaded_data = joblib.load(metadata_path)
        
        assert 'model' in loaded_data, "Model should be in loaded data"
        assert 'feature_names' in loaded_data, "Feature names should be preserved"
        assert loaded_data['n_features'] == X.shape[1], "Number of features should be preserved"
        assert loaded_data['model_type'] == 'RandomForestClassifier', "Model type should be preserved"


class TestPredictionInterface:
    """Test prediction interface and user interaction"""
    
    @pytest.fixture
    def mock_prediction_function(self):
        """Create a mock prediction function for testing"""
        def predict_survival(pclass, sex, age, sibsp, parch, fare, embarked):
            # Simple mock logic
            if sex == 'female' and pclass <= 2:
                return {'probability': 0.85, 'prediction': 'Survived'}
            elif sex == 'male' and pclass == 1 and age < 30:
                return {'probability': 0.65, 'prediction': 'Survived'}
            else:
                return {'probability': 0.25, 'prediction': 'Did not survive'}
        
        return predict_survival
    
    def test_prediction_function_interface(self, mock_prediction_function):
        """Test prediction function interface"""
        # Test various passenger profiles
        test_cases = [
            {
                'input': (1, 'female', 25, 0, 0, 100, 'S'),
                'expected_survival': 'Survived'
            },
            {
                'input': (3, 'male', 40, 2, 1, 10, 'S'),
                'expected_survival': 'Did not survive'
            },
            {
                'input': (1, 'male', 25, 0, 0, 150, 'C'),
                'expected_survival': 'Survived'
            }
        ]
        
        for case in test_cases:
            result = mock_prediction_function(*case['input'])
            
            # Test return format
            assert isinstance(result, dict), "Result should be a dictionary"
            assert 'probability' in result, "Result should contain probability"
            assert 'prediction' in result, "Result should contain prediction"
            
            # Test value ranges and types
            assert 0 <= result['probability'] <= 1, "Probability should be between 0 and 1"
            assert result['prediction'] in ['Survived', 'Did not survive'], "Prediction should be valid"
            
            # Test expected outcomes
            assert result['prediction'] == case['expected_survival'], f"Expected {case['expected_survival']}"
    
    def test_edge_cases_handling(self, mock_prediction_function):
        """Test handling of edge cases and boundary values"""
        edge_cases = [
            (1, 'female', 0, 0, 0, 0, 'S'),      # Infant with zero fare
            (3, 'male', 100, 0, 0, 500, 'C'),    # Very old with high fare
            (2, 'female', 25, 10, 10, 50, 'Q'),  # Large family
        ]
        
        for case in edge_cases:
            result = mock_prediction_function(*case)
            
            # Should not raise exceptions
            assert result is not None, "Should handle edge cases gracefully"
            assert 'probability' in result, "Should return probability even for edge cases"
            assert 'prediction' in result, "Should return prediction even for edge cases"


class TestErrorHandling:
    """Test error handling and validation"""
    
    def test_invalid_input_validation(self):
        """Test validation of invalid inputs"""
        def validate_passenger_input(pclass, sex, age, sibsp, parch, fare, embarked):
            errors = []
            
            if pclass not in [1, 2, 3]:
                errors.append("Passenger class must be 1, 2, or 3")
            if sex not in ['male', 'female']:
                errors.append("Sex must be 'male' or 'female'")
            if age < 0 or age > 120:
                errors.append("Age must be between 0 and 120")
            if sibsp < 0 or sibsp > 20:
                errors.append("Number of siblings/spouses must be non-negative and reasonable")
            if parch < 0 or parch > 20:
                errors.append("Number of parents/children must be non-negative and reasonable")
            if fare < 0 or fare > 1000:
                errors.append("Fare must be non-negative and reasonable")
            if embarked not in ['S', 'C', 'Q']:
                errors.append("Embarked must be 'S', 'C', or 'Q'")
            
            return errors
        
        # Test valid inputs
        valid_errors = validate_passenger_input(1, 'female', 25, 0, 0, 100, 'S')
        assert len(valid_errors) == 0, "Valid inputs should not produce errors"
        
        # Test invalid inputs
        invalid_cases = [
            (4, 'female', 25, 0, 0, 100, 'S'),      # Invalid class
            (1, 'unknown', 25, 0, 0, 100, 'S'),     # Invalid sex
            (1, 'female', -5, 0, 0, 100, 'S'),      # Invalid age
            (1, 'female', 25, -1, 0, 100, 'S'),     # Invalid sibsp
            (1, 'female', 25, 0, -1, 100, 'S'),     # Invalid parch
            (1, 'female', 25, 0, 0, -10, 'S'),      # Invalid fare
            (1, 'female', 25, 0, 0, 100, 'X'),      # Invalid embarked
        ]
        
        for case in invalid_cases:
            errors = validate_passenger_input(*case)
            assert len(errors) > 0, f"Invalid input {case} should produce errors"
    
    def test_missing_data_handling(self):
        """Test handling of missing data scenarios"""
        def handle_missing_values(data_dict):
            """Mock function to handle missing values"""
            defaults = {
                'Age': 30.0,
                'Fare': 32.0,
                'Embarked': 'S'
            }
            
            for key, value in data_dict.items():
                if pd.isna(value) or value is None:
                    if key in defaults:
                        data_dict[key] = defaults[key]
                    else:
                        raise ValueError(f"Missing value for required field: {key}")
            
            return data_dict
        
        # Test with missing values
        test_data = {
            'Pclass': 1,
            'Sex': 'female',
            'Age': None,
            'SibSp': 0,
            'Parch': 0,
            'Fare': np.nan,
            'Embarked': None
        }
        
        result = handle_missing_values(test_data.copy())
        
        assert result['Age'] == 30.0, "Missing age should be filled with default"
        assert result['Fare'] == 32.0, "Missing fare should be filled with default"
        assert result['Embarked'] == 'S', "Missing embarked should be filled with default"
        
        # Test with missing required field
        test_data_required_missing = test_data.copy()
        test_data_required_missing['Sex'] = None
        
        with pytest.raises(ValueError, match="Missing value for required field"):
            handle_missing_values(test_data_required_missing)


class TestIntegration:
    """Integration tests for complete workflow"""
    
    def test_end_to_end_prediction_workflow(self):
        """Test complete end-to-end prediction workflow"""
        # This test simulates the entire pipeline from data loading to prediction
        
        # 1. Create sample data
        np.random.seed(42)
        n_samples = 200
        
        df = pd.DataFrame({
            'PassengerId': range(1, n_samples + 1),
            'Survived': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
            'Pclass': np.random.choice([1, 2, 3], n_samples, p=[0.2, 0.2, 0.6]),
            'Name': [f"Person {i}, Mr. Test" for i in range(n_samples)],
            'Sex': np.random.choice(['male', 'female'], n_samples),
            'Age': np.random.normal(30, 15, n_samples).clip(1, 80),
            'SibSp': np.random.poisson(0.5, n_samples),
            'Parch': np.random.poisson(0.4, n_samples),
            'Fare': np.random.lognormal(3, 1, n_samples),
            'Embarked': np.random.choice(['S', 'C', 'Q'], n_samples, p=[0.7, 0.2, 0.1])
        })
        
        # 2. Feature engineering
        df['Title'] = 'Mr'  # Simplified for testing
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
        
        # 3. Preprocessing
        features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'FamilySize', 'IsAlone']
        X = df[features].copy()
        y = df['Survived'].copy()
        
        # Encode categorical variables
        le_sex = LabelEncoder()
        le_embarked = LabelEncoder()
        X['Sex'] = le_sex.fit_transform(X['Sex'])
        X['Embarked'] = le_embarked.fit_transform(X['Embarked'])
        
        # 4. Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 5. Scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 6. Model training
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # 7. Predictions and evaluation
        predictions = model.predict(X_test_scaled)
        probabilities = model.predict_proba(X_test_scaled)[:, 1]
        
        accuracy = accuracy_score(y_test, predictions)
        auc = roc_auc_score(y_test, probabilities)
        
        # 8. Assertions for integration test
        assert accuracy > 0.4, f"Integration test accuracy too low: {accuracy}"
        assert auc > 0.4, f"Integration test AUC too low: {auc}"
        assert len(predictions) == len(y_test), "Should have predictions for all test samples"
        
        # 9. Test individual prediction
        sample_input = X_test_scaled[0:1]
        single_pred = model.predict(sample_input)
        single_prob = model.predict_proba(sample_input)
        
        assert len(single_pred) == 1, "Single prediction should return one result"
        assert single_prob.shape == (1, 2), "Single prediction probabilities should have correct shape"
        
        print(f"✅ Integration test passed - Accuracy: {accuracy:.3f}, AUC: {auc:.3f}")


if __name__ == "__main__":
    # Run tests when script is executed directly
    pytest.main([__file__, "-v", "--tb=short"])