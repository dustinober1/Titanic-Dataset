#!/usr/bin/env python3
"""
Data Validation and Quality Assurance Tests
==========================================

This module provides comprehensive data validation tests to ensure data quality,
integrity, and consistency throughout the Titanic ML pipeline.

Validation Categories:
1. Data Schema Validation
2. Data Quality Checks
3. Statistical Validation
4. Business Logic Validation
5. Data Drift Detection
6. Outlier Detection and Handling

Author: Enhanced Titanic ML Testing Framework
"""

import pytest
import numpy as np
import pandas as pd
import warnings
from scipy import stats
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import os

warnings.filterwarnings('ignore')


class TestDataSchemaValidation:
    """Test data schema and structure validation"""
    
    @pytest.fixture
    def expected_schema(self):
        """Define expected data schema"""
        return {
            'columns': [
                'PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 
                'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'
            ],
            'dtypes': {
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
            'required_columns': [
                'PassengerId', 'Survived', 'Pclass', 'Sex', 
                'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'
            ]
        }
    
    @pytest.fixture
    def sample_valid_data(self):
        """Create sample data that passes validation"""
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
    
    def test_required_columns_present(self, sample_valid_data, expected_schema):
        """Test that all required columns are present"""
        df = sample_valid_data
        required_cols = expected_schema['required_columns']
        
        for col in required_cols:
            assert col in df.columns, f"Required column '{col}' is missing from dataset"
    
    def test_data_types_correct(self, sample_valid_data, expected_schema):
        """Test that column data types are correct"""
        df = sample_valid_data
        dtype_specs = expected_schema['dtypes']
        
        for col, expected_dtypes in dtype_specs.items():
            if col in df.columns:
                actual_dtype = str(df[col].dtype)
                assert actual_dtype in expected_dtypes, \
                    f"Column '{col}' has dtype '{actual_dtype}', expected one of {expected_dtypes}"
    
    def test_passenger_id_uniqueness(self, sample_valid_data):
        """Test that PassengerId values are unique"""
        df = sample_valid_data
        
        if 'PassengerId' in df.columns:
            unique_ids = df['PassengerId'].nunique()
            total_rows = len(df)
            
            assert unique_ids == total_rows, \
                f"PassengerId should be unique. Found {unique_ids} unique values for {total_rows} rows"
    
    def test_binary_columns_valid(self, sample_valid_data):
        """Test that binary columns contain only valid values"""
        df = sample_valid_data
        
        if 'Survived' in df.columns:
            valid_values = {0, 1}
            actual_values = set(df['Survived'].dropna().unique())
            
            assert actual_values.issubset(valid_values), \
                f"Survived column should only contain {valid_values}, found {actual_values}"
    
    def test_categorical_columns_valid(self, sample_valid_data):
        """Test that categorical columns contain only valid values"""
        df = sample_valid_data
        
        # Test Sex column
        if 'Sex' in df.columns:
            valid_sex_values = {'male', 'female'}
            actual_sex_values = set(df['Sex'].dropna().unique())
            
            assert actual_sex_values.issubset(valid_sex_values), \
                f"Sex column should only contain {valid_sex_values}, found {actual_sex_values}"
        
        # Test Pclass column
        if 'Pclass' in df.columns:
            valid_pclass_values = {1, 2, 3}
            actual_pclass_values = set(df['Pclass'].dropna().unique())
            
            assert actual_pclass_values.issubset(valid_pclass_values), \
                f"Pclass column should only contain {valid_pclass_values}, found {actual_pclass_values}"
        
        # Test Embarked column
        if 'Embarked' in df.columns:
            valid_embarked_values = {'S', 'C', 'Q'}
            actual_embarked_values = set(df['Embarked'].dropna().unique())
            
            assert actual_embarked_values.issubset(valid_embarked_values), \
                f"Embarked column should only contain {valid_embarked_values}, found {actual_embarked_values}"


class TestDataQualityChecks:
    """Test data quality and integrity"""
    
    @pytest.fixture
    def data_with_quality_issues(self):
        """Create data with various quality issues for testing"""
        return pd.DataFrame({
            'PassengerId': [1, 2, 3, 4, 5, 6, 7],
            'Survived': [0, 1, 1, -1, 0, 1, 2],  # Invalid values
            'Pclass': [3, 1, 3, 1, 3, 0, 4],     # Invalid values
            'Sex': ['male', 'female', 'female', 'unknown', 'male', 'female', ''],
            'Age': [22.0, 38.0, -5.0, 35.0, 200.0, 38.0, np.nan],  # Negative and extreme values
            'SibSp': [1, 1, 0, -1, 0, 20, 1],    # Negative and extreme values
            'Parch': [0, 0, 0, 0, -1, 15, 0],    # Negative and extreme values
            'Fare': [7.25, 71.28, -10.0, 53.1, 0.0, 1000.0, np.nan],  # Negative and extreme values
            'Embarked': ['S', 'C', 'S', 'X', 'S', '', 'Q']
        })
    
    def test_missing_value_percentage(self, data_with_quality_issues):
        """Test missing value percentages are within acceptable limits"""
        df = data_with_quality_issues
        
        # Define acceptable missing value thresholds
        missing_thresholds = {
            'PassengerId': 0.0,   # Should never be missing
            'Survived': 0.0,      # Should never be missing
            'Pclass': 0.0,        # Should never be missing
            'Sex': 0.05,          # Max 5% missing
            'Age': 0.25,          # Max 25% missing (common in historical data)
            'SibSp': 0.05,        # Max 5% missing
            'Parch': 0.05,        # Max 5% missing
            'Fare': 0.1,          # Max 10% missing
            'Embarked': 0.05      # Max 5% missing
        }
        
        for col, threshold in missing_thresholds.items():
            if col in df.columns:
                missing_pct = df[col].isna().mean()
                assert missing_pct <= threshold, \
                    f"Column '{col}' has {missing_pct:.2%} missing values, exceeds threshold of {threshold:.1%}"
    
    def test_duplicate_rows(self, data_with_quality_issues):
        """Test for duplicate rows"""
        df = data_with_quality_issues
        
        # Check for complete duplicates
        duplicate_count = df.duplicated().sum()
        duplicate_pct = duplicate_count / len(df)
        
        # Generally, we expect very few duplicates in passenger data
        assert duplicate_pct < 0.05, \
            f"Found {duplicate_count} duplicate rows ({duplicate_pct:.2%}), which exceeds 5% threshold"
    
    def test_data_consistency_rules(self, data_with_quality_issues):
        """Test business logic consistency rules"""
        df = data_with_quality_issues
        
        # Rule 1: Age should be non-negative and reasonable
        if 'Age' in df.columns:
            invalid_ages = df[(df['Age'] < 0) | (df['Age'] > 120)]['Age'].dropna()
            assert len(invalid_ages) == 0, \
                f"Found {len(invalid_ages)} records with invalid ages: {invalid_ages.tolist()}"
        
        # Rule 2: Fare should be non-negative
        if 'Fare' in df.columns:
            negative_fares = df[df['Fare'] < 0]['Fare'].dropna()
            assert len(negative_fares) == 0, \
                f"Found {len(negative_fares)} records with negative fares: {negative_fares.tolist()}"
        
        # Rule 3: SibSp and Parch should be non-negative
        for col in ['SibSp', 'Parch']:
            if col in df.columns:
                negative_values = df[df[col] < 0][col].dropna()
                assert len(negative_values) == 0, \
                    f"Found {len(negative_values)} records with negative {col}: {negative_values.tolist()}"
    
    def test_extreme_value_detection(self, data_with_quality_issues):
        """Test for extreme values that might indicate data quality issues"""
        df = data_with_quality_issues
        
        # Test for extreme family sizes
        if 'SibSp' in df.columns and 'Parch' in df.columns:
            df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
            extreme_family_sizes = df[df['FamilySize'] > 15]['FamilySize']
            
            # Flag as warning rather than error, as some large families existed
            if len(extreme_family_sizes) > 0:
                print(f"⚠️  Warning: Found {len(extreme_family_sizes)} records with very large families")
        
        # Test for extreme fares
        if 'Fare' in df.columns:
            fare_q99 = df['Fare'].quantile(0.99)
            extreme_fares = df[df['Fare'] > fare_q99 * 3]['Fare'].dropna()
            
            if len(extreme_fares) > 0:
                print(f"⚠️  Warning: Found {len(extreme_fares)} records with extremely high fares")


class TestStatisticalValidation:
    """Test statistical properties and distributions"""
    
    @pytest.fixture
    def reference_statistics(self):
        """Define expected statistical properties based on historical Titanic data"""
        return {
            'survival_rate': {'min': 0.30, 'max': 0.45},  # Historical ~38%
            'age_statistics': {
                'mean': {'min': 25, 'max': 35},
                'std': {'min': 10, 'max': 20},
                'median': {'min': 25, 'max': 35}
            },
            'fare_statistics': {
                'median': {'min': 10, 'max': 20},
                'mean': {'min': 25, 'max': 40}
            },
            'class_distribution': {
                'pclass_1': {'min': 0.15, 'max': 0.30},  # ~24%
                'pclass_2': {'min': 0.15, 'max': 0.25},  # ~21%
                'pclass_3': {'min': 0.50, 'max': 0.65}   # ~55%
            },
            'gender_distribution': {
                'male': {'min': 0.60, 'max': 0.70},      # ~65%
                'female': {'min': 0.30, 'max': 0.40}     # ~35%
            }
        }
    
    def create_realistic_sample_data(self, n=1000):
        """Create sample data with realistic statistical properties"""
        np.random.seed(42)
        
        # Create passenger classes with realistic distribution
        pclass = np.random.choice([1, 2, 3], n, p=[0.24, 0.21, 0.55])
        
        # Create gender with realistic distribution
        sex = np.random.choice(['male', 'female'], n, p=[0.65, 0.35])
        
        # Create realistic ages
        age = np.random.gamma(2, 15)  # Gamma distribution for realistic age distribution
        age = np.clip(age, 0.1, 80)
        
        # Create realistic fares based on class
        fare_base = np.where(pclass == 1, 80, np.where(pclass == 2, 20, 10))
        fare = fare_base * np.random.lognormal(0, 0.5)
        
        # Create survival with realistic patterns
        survival_prob = (
            0.4 +  # Base survival rate
            0.35 * (sex == 'female') +  # Female survival advantage
            0.15 * (pclass == 1) +      # First class advantage
            0.08 * (pclass == 2) +      # Second class advantage
            0.1 * (age < 16)            # Children advantage
        )
        survived = np.random.binomial(1, np.clip(survival_prob, 0, 1), n)
        
        return pd.DataFrame({
            'PassengerId': range(1, n + 1),
            'Survived': survived,
            'Pclass': pclass,
            'Sex': sex,
            'Age': age,
            'SibSp': np.random.poisson(0.5, n),
            'Parch': np.random.poisson(0.4, n),
            'Fare': fare,
            'Embarked': np.random.choice(['S', 'C', 'Q'], n, p=[0.72, 0.19, 0.09])
        })
    
    def test_survival_rate_statistical_validity(self, reference_statistics):
        """Test that survival rate is within expected historical range"""
        df = self.create_realistic_sample_data()
        
        survival_rate = df['Survived'].mean()
        expected_range = reference_statistics['survival_rate']
        
        assert expected_range['min'] <= survival_rate <= expected_range['max'], \
            f"Survival rate {survival_rate:.3f} is outside expected range {expected_range}"
    
    def test_age_distribution_validity(self, reference_statistics):
        """Test that age distribution matches expected patterns"""
        df = self.create_realistic_sample_data()
        
        age_stats = reference_statistics['age_statistics']
        age_data = df['Age'].dropna()
        
        # Test mean
        age_mean = age_data.mean()
        assert age_stats['mean']['min'] <= age_mean <= age_stats['mean']['max'], \
            f"Age mean {age_mean:.2f} is outside expected range {age_stats['mean']}"
        
        # Test median
        age_median = age_data.median()
        assert age_stats['median']['min'] <= age_median <= age_stats['median']['max'], \
            f"Age median {age_median:.2f} is outside expected range {age_stats['median']}"
    
    def test_class_distribution_validity(self, reference_statistics):
        """Test that passenger class distribution is realistic"""
        df = self.create_realistic_sample_data()
        
        class_dist = df['Pclass'].value_counts(normalize=True)
        expected_dist = reference_statistics['class_distribution']
        
        for pclass in [1, 2, 3]:
            actual_prop = class_dist.get(pclass, 0)
            expected_range = expected_dist[f'pclass_{pclass}']
            
            assert expected_range['min'] <= actual_prop <= expected_range['max'], \
                f"Class {pclass} proportion {actual_prop:.3f} is outside expected range {expected_range}"
    
    def test_correlation_patterns(self):
        """Test expected correlation patterns in the data"""
        df = self.create_realistic_sample_data()
        
        # Encode categorical variables for correlation
        df_encoded = df.copy()
        df_encoded['Sex'] = (df_encoded['Sex'] == 'female').astype(int)
        
        numeric_cols = ['Survived', 'Pclass', 'Sex', 'Age', 'Fare']
        corr_matrix = df_encoded[numeric_cols].corr()
        
        # Expected strong correlations
        expected_correlations = [
            ('Survived', 'Sex', 0.3),       # Females more likely to survive
            ('Survived', 'Pclass', -0.2),   # Higher class more likely to survive
            ('Pclass', 'Fare', -0.5),       # Higher class pays more
        ]
        
        for var1, var2, expected_min_corr in expected_correlations:
            actual_corr = abs(corr_matrix.loc[var1, var2])
            
            assert actual_corr >= abs(expected_min_corr), \
                f"Correlation between {var1} and {var2} is {actual_corr:.3f}, " \
                f"expected at least {abs(expected_min_corr):.3f}"
    
    def test_normality_assumptions(self):
        """Test normality assumptions for key variables"""
        df = self.create_realistic_sample_data(n=500)  # Smaller sample for normality tests
        
        # Test log-transformed fare for approximate normality
        log_fare = np.log(df['Fare'] + 1)  # Add 1 to handle zero fares
        _, p_value = stats.normaltest(log_fare)
        
        # We expect log-fare to be approximately normal (p > 0.01)
        if p_value <= 0.01:
            print(f"⚠️  Warning: Log-transformed fare may not be normally distributed (p={p_value:.4f})")


class TestDataDriftDetection:
    """Test for data drift and distribution changes"""
    
    def create_baseline_data(self, n=1000):
        """Create baseline dataset for drift comparison"""
        np.random.seed(42)
        return pd.DataFrame({
            'Age': np.random.normal(30, 15, n),
            'Fare': np.random.lognormal(3, 1, n),
            'Pclass': np.random.choice([1, 2, 3], n, p=[0.24, 0.21, 0.55]),
            'Sex': np.random.choice(['male', 'female'], n, p=[0.65, 0.35])
        })
    
    def create_drifted_data(self, n=1000):
        """Create dataset with simulated data drift"""
        np.random.seed(123)  # Different seed
        return pd.DataFrame({
            'Age': np.random.normal(35, 18, n),  # Shifted mean and variance
            'Fare': np.random.lognormal(3.5, 1.2, n),  # Higher fares
            'Pclass': np.random.choice([1, 2, 3], n, p=[0.3, 0.25, 0.45]),  # More first class
            'Sex': np.random.choice(['male', 'female'], n, p=[0.6, 0.4])  # More females
        })
    
    def test_kolmogorov_smirnov_drift_detection(self):
        """Test for distribution drift using Kolmogorov-Smirnov test"""
        baseline_data = self.create_baseline_data()
        drifted_data = self.create_drifted_data()
        
        # Test for drift in continuous variables
        continuous_vars = ['Age', 'Fare']
        drift_threshold = 0.01  # p-value threshold for detecting drift
        
        for var in continuous_vars:
            ks_statistic, p_value = stats.ks_2samp(
                baseline_data[var], drifted_data[var]
            )
            
            # For this test, we expect to detect drift (p < threshold)
            if p_value < drift_threshold:
                print(f"✅ Drift detected in {var}: KS statistic={ks_statistic:.4f}, p={p_value:.6f}")
            else:
                print(f"⚠️  No significant drift detected in {var}: p={p_value:.4f}")
    
    def test_chi_square_categorical_drift(self):
        """Test for drift in categorical variables using chi-square test"""
        baseline_data = self.create_baseline_data()
        drifted_data = self.create_drifted_data()
        
        categorical_vars = ['Pclass', 'Sex']
        
        for var in categorical_vars:
            # Create contingency table
            baseline_counts = baseline_data[var].value_counts().sort_index()
            drifted_counts = drifted_data[var].value_counts().sort_index()
            
            # Ensure same categories
            all_categories = set(baseline_counts.index) | set(drifted_counts.index)
            baseline_counts = baseline_counts.reindex(all_categories, fill_value=0)
            drifted_counts = drifted_counts.reindex(all_categories, fill_value=0)
            
            # Chi-square test
            contingency_table = np.array([baseline_counts.values, drifted_counts.values])
            chi2, p_value, _, _ = stats.chi2_contingency(contingency_table)
            
            print(f"Chi-square test for {var}: χ²={chi2:.4f}, p={p_value:.6f}")
    
    def test_statistical_moments_drift(self):
        """Test for drift in statistical moments (mean, variance, skewness)"""
        baseline_data = self.create_baseline_data()
        current_data = self.create_drifted_data()
        
        continuous_vars = ['Age', 'Fare']
        
        for var in continuous_vars:
            baseline_stats = {
                'mean': baseline_data[var].mean(),
                'std': baseline_data[var].std(),
                'skew': stats.skew(baseline_data[var])
            }
            
            current_stats = {
                'mean': current_data[var].mean(),
                'std': current_data[var].std(),
                'skew': stats.skew(current_data[var])
            }
            
            # Calculate relative changes
            for stat in ['mean', 'std']:
                relative_change = abs(current_stats[stat] - baseline_stats[stat]) / baseline_stats[stat]
                
                # Flag significant changes (>20% relative change)
                if relative_change > 0.2:
                    print(f"⚠️  Significant {stat} change in {var}: {relative_change:.2%}")


class TestOutlierDetection:
    """Test outlier detection and handling"""
    
    @pytest.fixture
    def data_with_outliers(self):
        """Create data with known outliers"""
        np.random.seed(42)
        n = 100
        
        # Normal data
        normal_data = np.random.normal(30, 10, n-5)
        
        # Add outliers
        outliers = np.array([100, 150, -20, 200, 0])
        
        combined_data = np.concatenate([normal_data, outliers])
        
        return pd.DataFrame({
            'Age': combined_data,
            'Fare': np.random.lognormal(3, 1, n) + np.random.choice([0, 500], n, p=[0.95, 0.05]),
            'SibSp': np.random.choice([0, 1, 2, 20], n, p=[0.7, 0.2, 0.05, 0.05])  # Extreme family size
        })
    
    def test_iqr_outlier_detection(self, data_with_outliers):
        """Test IQR-based outlier detection"""
        df = data_with_outliers
        
        for col in ['Age', 'Fare']:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
            outlier_pct = len(outliers) / len(df)
            
            # We expect some outliers in our test data
            assert outlier_pct > 0.02, f"Expected outliers in {col}, but found {outlier_pct:.2%}"
            
            # But not too many (>20% would indicate data quality issues)
            assert outlier_pct < 0.20, f"Too many outliers in {col}: {outlier_pct:.2%}"
            
            print(f"IQR outliers in {col}: {len(outliers)} ({outlier_pct:.1%})")
    
    def test_zscore_outlier_detection(self, data_with_outliers):
        """Test Z-score based outlier detection"""
        df = data_with_outliers
        
        for col in ['Age', 'Fare']:
            z_scores = np.abs(stats.zscore(df[col]))
            outliers = df[z_scores > 3][col]
            outlier_pct = len(outliers) / len(df)
            
            print(f"Z-score outliers in {col} (|z| > 3): {len(outliers)} ({outlier_pct:.1%})")
            
            # Check that outliers are flagged
            if col == 'Age':
                # We added extreme age values
                extreme_ages = df[df[col].isin([100, 150, -20, 200])][col]
                extreme_z_scores = np.abs(stats.zscore(df[col]))[df[col].isin([100, 150, -20, 200])]
                
                assert all(extreme_z_scores > 3), "Extreme age outliers should be detected"
    
    def test_modified_zscore_outlier_detection(self, data_with_outliers):
        """Test Modified Z-score (MAD-based) outlier detection"""
        df = data_with_outliers
        
        def modified_z_score(data):
            """Calculate modified Z-score using MAD"""
            median = np.median(data)
            mad = np.median(np.abs(data - median))
            modified_z_scores = 0.6745 * (data - median) / mad
            return np.abs(modified_z_scores)
        
        for col in ['Age', 'Fare']:
            modified_z_scores = modified_z_score(df[col])
            outliers = df[modified_z_scores > 3.5][col]
            outlier_pct = len(outliers) / len(df)
            
            print(f"Modified Z-score outliers in {col}: {len(outliers)} ({outlier_pct:.1%})")
            
            # Modified Z-score should be more robust to outliers than regular Z-score
            assert outlier_pct >= 0.02, f"Should detect some outliers in {col}"


if __name__ == "__main__":
    # Create a simple test runner for demonstration
    import sys
    
    print("🔍 Running Titanic Data Validation Tests")
    print("=" * 50)
    
    # Run a few key tests manually for demonstration
    test_schema = TestDataSchemaValidation()
    test_quality = TestDataQualityChecks()
    test_stats = TestStatisticalValidation()
    test_drift = TestDataDriftDetection()
    test_outliers = TestOutlierDetection()
    
    try:
        # Create sample data
        sample_data = test_schema.sample_valid_data()
        expected_schema = test_schema.expected_schema()
        
        # Run some tests
        print("✅ Schema validation tests")
        test_schema.test_required_columns_present(sample_data, expected_schema)
        test_schema.test_data_types_correct(sample_data, expected_schema)
        
        print("✅ Statistical validation tests")
        test_stats.test_survival_rate_statistical_validity(test_stats.reference_statistics())
        
        print("✅ Drift detection tests")
        test_drift.test_kolmogorov_smirnov_drift_detection()
        
        print("✅ Outlier detection tests")
        outlier_data = test_outliers.data_with_outliers()
        test_outliers.test_iqr_outlier_detection(outlier_data)
        
        print("\\n🎉 All validation tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        sys.exit(1)
    
    print("\\n💡 To run full test suite: pytest test_data_validation.py -v")