#!/usr/bin/env python3
"""
Schema Validation Module
========================

Provides comprehensive schema validation for Titanic dataset including:
- Column presence validation
- Data type validation  
- Value constraint validation
- Schema evolution tracking

Author: Enhanced Titanic ML Framework
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Container for validation results"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    summary: Dict[str, Any]
    timestamp: datetime


@dataclass  
class ColumnSchema:
    """Schema definition for a single column"""
    name: str
    dtype: Union[str, List[str]]
    nullable: bool = True
    unique: bool = False
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    allowed_values: Optional[List[Any]] = None
    pattern: Optional[str] = None
    description: str = ""


class SchemaValidator:
    """
    Comprehensive schema validation for Titanic dataset
    
    This validator ensures data conforms to expected schema and 
    business rules for the Titanic survival prediction dataset.
    """
    
    def __init__(self):
        """Initialize validator with Titanic dataset schema"""
        self.schema = self._define_titanic_schema()
        self.validation_history: List[ValidationResult] = []
    
    def _define_titanic_schema(self) -> Dict[str, ColumnSchema]:
        """Define the expected schema for Titanic dataset"""
        
        return {
            'PassengerId': ColumnSchema(
                name='PassengerId',
                dtype=['int64', 'int32'],
                nullable=False,
                unique=True,
                min_value=1,
                description="Unique identifier for each passenger"
            ),
            
            'Survived': ColumnSchema(
                name='Survived', 
                dtype=['int64', 'int32'],
                nullable=False,
                allowed_values=[0, 1],
                description="Survival indicator (0=No, 1=Yes)"
            ),
            
            'Pclass': ColumnSchema(
                name='Pclass',
                dtype=['int64', 'int32'],
                nullable=False,
                allowed_values=[1, 2, 3],
                description="Passenger class (1=1st, 2=2nd, 3=3rd)"
            ),
            
            'Name': ColumnSchema(
                name='Name',
                dtype=['object'],
                nullable=False,
                description="Passenger name"
            ),
            
            'Sex': ColumnSchema(
                name='Sex',
                dtype=['object'],
                nullable=False,
                allowed_values=['male', 'female'],
                description="Gender of passenger"
            ),
            
            'Age': ColumnSchema(
                name='Age',
                dtype=['float64', 'float32'],
                nullable=True,
                min_value=0,
                max_value=120,
                description="Age in years"
            ),
            
            'SibSp': ColumnSchema(
                name='SibSp',
                dtype=['int64', 'int32'],
                nullable=False,
                min_value=0,
                max_value=20,
                description="Number of siblings/spouses aboard"
            ),
            
            'Parch': ColumnSchema(
                name='Parch', 
                dtype=['int64', 'int32'],
                nullable=False,
                min_value=0,
                max_value=20,
                description="Number of parents/children aboard"
            ),
            
            'Ticket': ColumnSchema(
                name='Ticket',
                dtype=['object'],
                nullable=False,
                description="Ticket number"
            ),
            
            'Fare': ColumnSchema(
                name='Fare',
                dtype=['float64', 'float32'], 
                nullable=True,
                min_value=0,
                max_value=1000,
                description="Passenger fare"
            ),
            
            'Cabin': ColumnSchema(
                name='Cabin',
                dtype=['object'],
                nullable=True,
                description="Cabin number"
            ),
            
            'Embarked': ColumnSchema(
                name='Embarked',
                dtype=['object'],
                nullable=True,
                allowed_values=['S', 'C', 'Q'],
                description="Port of embarkation (S=Southampton, C=Cherbourg, Q=Queenstown)"
            )
        }
    
    def validate_dataframe(self, df: pd.DataFrame, strict: bool = False) -> ValidationResult:
        """
        Validate a DataFrame against the Titanic schema
        
        Args:
            df: DataFrame to validate
            strict: If True, warnings become errors
            
        Returns:
            ValidationResult with detailed validation information
        """
        errors = []
        warnings = []
        
        logger.info(f"🔍 Starting schema validation for DataFrame with {len(df)} rows, {len(df.columns)} columns")
        
        # Check required columns
        missing_cols = self._validate_required_columns(df)
        if missing_cols:
            errors.extend(missing_cols)
        
        # Check extra columns
        extra_cols = self._validate_extra_columns(df)
        if extra_cols:
            if strict:
                errors.extend(extra_cols)
            else:
                warnings.extend(extra_cols)
        
        # Validate each column that exists
        for col_name, col_schema in self.schema.items():
            if col_name in df.columns:
                col_errors, col_warnings = self._validate_column(df, col_name, col_schema, strict)
                errors.extend(col_errors)
                warnings.extend(col_warnings)
        
        # Business logic validation
        business_errors, business_warnings = self._validate_business_rules(df, strict)
        errors.extend(business_errors)
        warnings.extend(business_warnings)
        
        # Create summary
        summary = self._create_validation_summary(df, errors, warnings)
        
        # Create result
        result = ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            summary=summary,
            timestamp=datetime.now()
        )
        
        # Store in history
        self.validation_history.append(result)
        
        # Log results
        self._log_validation_results(result)
        
        return result
    
    def _validate_required_columns(self, df: pd.DataFrame) -> List[str]:
        """Validate that all required columns are present"""
        errors = []
        required_columns = list(self.schema.keys())
        
        for col in required_columns:
            if col not in df.columns:
                errors.append(f"Required column '{col}' is missing from dataset")
        
        return errors
    
    def _validate_extra_columns(self, df: pd.DataFrame) -> List[str]:
        """Check for unexpected columns"""
        warnings = []
        expected_columns = set(self.schema.keys())
        actual_columns = set(df.columns)
        
        extra_columns = actual_columns - expected_columns
        
        for col in extra_columns:
            warnings.append(f"Unexpected column '{col}' found in dataset")
        
        return warnings
    
    def _validate_column(self, df: pd.DataFrame, col_name: str, 
                        col_schema: ColumnSchema, strict: bool = False) -> Tuple[List[str], List[str]]:
        """Validate a specific column against its schema"""
        errors = []
        warnings = []
        
        column_data = df[col_name]
        
        # Data type validation
        dtype_errors = self._validate_dtype(column_data, col_schema)
        errors.extend(dtype_errors)
        
        # Null value validation
        null_errors = self._validate_nulls(column_data, col_schema)
        if strict:
            errors.extend(null_errors)
        else:
            warnings.extend(null_errors)
        
        # Uniqueness validation
        unique_errors = self._validate_uniqueness(column_data, col_schema)
        errors.extend(unique_errors)
        
        # Value range validation
        range_errors = self._validate_value_range(column_data, col_schema)
        errors.extend(range_errors)
        
        # Allowed values validation
        allowed_errors = self._validate_allowed_values(column_data, col_schema)
        errors.extend(allowed_errors)
        
        return errors, warnings
    
    def _validate_dtype(self, column_data: pd.Series, col_schema: ColumnSchema) -> List[str]:
        """Validate column data type"""
        errors = []
        
        actual_dtype = str(column_data.dtype)
        expected_dtypes = col_schema.dtype if isinstance(col_schema.dtype, list) else [col_schema.dtype]
        
        if actual_dtype not in expected_dtypes:
            errors.append(
                f"Column '{col_schema.name}' has dtype '{actual_dtype}', "
                f"expected one of {expected_dtypes}"
            )
        
        return errors
    
    def _validate_nulls(self, column_data: pd.Series, col_schema: ColumnSchema) -> List[str]:
        """Validate null value constraints"""
        errors = []
        
        null_count = column_data.isnull().sum()
        
        if not col_schema.nullable and null_count > 0:
            errors.append(
                f"Column '{col_schema.name}' contains {null_count} null values "
                f"but should not be nullable"
            )
        
        # Check for excessive null values (>50% for nullable columns)
        if col_schema.nullable and null_count > len(column_data) * 0.5:
            errors.append(
                f"Column '{col_schema.name}' has {null_count}/{len(column_data)} "
                f"({null_count/len(column_data):.1%}) null values, which may indicate data quality issues"
            )
        
        return errors
    
    def _validate_uniqueness(self, column_data: pd.Series, col_schema: ColumnSchema) -> List[str]:
        """Validate uniqueness constraints"""
        errors = []
        
        if col_schema.unique:
            duplicate_count = column_data.duplicated().sum()
            if duplicate_count > 0:
                errors.append(
                    f"Column '{col_schema.name}' should be unique but contains "
                    f"{duplicate_count} duplicate values"
                )
        
        return errors
    
    def _validate_value_range(self, column_data: pd.Series, col_schema: ColumnSchema) -> List[str]:
        """Validate value range constraints"""
        errors = []
        
        if col_schema.min_value is not None or col_schema.max_value is not None:
            numeric_data = column_data.dropna()
            
            if len(numeric_data) > 0:
                if col_schema.min_value is not None:
                    below_min = numeric_data < col_schema.min_value
                    if below_min.any():
                        count = below_min.sum()
                        min_val = numeric_data[below_min].min()
                        errors.append(
                            f"Column '{col_schema.name}' has {count} values below minimum "
                            f"{col_schema.min_value} (lowest: {min_val})"
                        )
                
                if col_schema.max_value is not None:
                    above_max = numeric_data > col_schema.max_value
                    if above_max.any():
                        count = above_max.sum()
                        max_val = numeric_data[above_max].max()
                        errors.append(
                            f"Column '{col_schema.name}' has {count} values above maximum "
                            f"{col_schema.max_value} (highest: {max_val})"
                        )
        
        return errors
    
    def _validate_allowed_values(self, column_data: pd.Series, col_schema: ColumnSchema) -> List[str]:
        """Validate allowed values constraints"""
        errors = []
        
        if col_schema.allowed_values is not None:
            non_null_data = column_data.dropna()
            allowed_set = set(col_schema.allowed_values)
            actual_values = set(non_null_data.unique())
            
            invalid_values = actual_values - allowed_set
            
            if invalid_values:
                errors.append(
                    f"Column '{col_schema.name}' contains invalid values: {list(invalid_values)}. "
                    f"Allowed values: {col_schema.allowed_values}"
                )
        
        return errors
    
    def _validate_business_rules(self, df: pd.DataFrame, strict: bool = False) -> Tuple[List[str], List[str]]:
        """Validate business-specific rules for Titanic dataset"""
        errors = []
        warnings = []
        
        # Rule: Family size consistency
        if 'SibSp' in df.columns and 'Parch' in df.columns:
            family_size = df['SibSp'] + df['Parch'] + 1
            large_families = family_size > 15
            
            if large_families.any():
                count = large_families.sum()
                max_size = family_size.max()
                warning_msg = (
                    f"Found {count} passengers with unusually large families "
                    f"(largest: {max_size} people). This may indicate data quality issues."
                )
                if strict:
                    errors.append(warning_msg)
                else:
                    warnings.append(warning_msg)
        
        # Rule: Age and fare relationship reasonableness
        if 'Age' in df.columns and 'Fare' in df.columns and 'Pclass' in df.columns:
            # Children in first class with very low fares might be data issues
            children_first_class = df[(df['Age'] < 5) & (df['Pclass'] == 1) & (df['Fare'] < 10)]
            
            if len(children_first_class) > 0:
                warnings.append(
                    f"Found {len(children_first_class)} children in first class with very low fares. "
                    "This may indicate data quality issues or special circumstances."
                )
        
        # Rule: Fare consistency with passenger class
        if 'Fare' in df.columns and 'Pclass' in df.columns:
            # Check for extreme fare inconsistencies
            high_fare_third_class = df[(df['Pclass'] == 3) & (df['Fare'] > 100)]
            low_fare_first_class = df[(df['Pclass'] == 1) & (df['Fare'] < 20)]
            
            if len(high_fare_third_class) > 0:
                warnings.append(
                    f"Found {len(high_fare_third_class)} third-class passengers with very high fares (>$100)"
                )
            
            if len(low_fare_first_class) > 0:
                warnings.append(
                    f"Found {len(low_fare_first_class)} first-class passengers with very low fares (<$20)"
                )
        
        return errors, warnings
    
    def _create_validation_summary(self, df: pd.DataFrame, errors: List[str], warnings: List[str]) -> Dict[str, Any]:
        """Create validation summary statistics"""
        
        summary = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'error_count': len(errors),
            'warning_count': len(warnings),
            'validation_passed': len(errors) == 0,
            'columns_validated': len([col for col in self.schema.keys() if col in df.columns]),
            'missing_columns': [col for col in self.schema.keys() if col not in df.columns],
            'extra_columns': [col for col in df.columns if col not in self.schema.keys()],
            'null_counts': df.isnull().sum().to_dict(),
            'data_types': df.dtypes.astype(str).to_dict()
        }
        
        # Add column-specific statistics
        for col in df.columns:
            if col in self.schema:
                col_stats = {}
                
                if df[col].dtype in ['int64', 'float64']:
                    col_stats.update({
                        'min': float(df[col].min()) if not df[col].isnull().all() else None,
                        'max': float(df[col].max()) if not df[col].isnull().all() else None,
                        'mean': float(df[col].mean()) if not df[col].isnull().all() else None
                    })
                
                col_stats.update({
                    'unique_count': df[col].nunique(),
                    'null_count': int(df[col].isnull().sum()),
                    'null_percentage': float(df[col].isnull().mean() * 100)
                })
                
                summary[f'{col}_stats'] = col_stats
        
        return summary
    
    def _log_validation_results(self, result: ValidationResult):
        """Log validation results"""
        
        if result.is_valid:
            logger.info("✅ Schema validation PASSED")
        else:
            logger.error("❌ Schema validation FAILED")
        
        logger.info(f"📊 Validation Summary:")
        logger.info(f"   - Rows: {result.summary['total_rows']}")
        logger.info(f"   - Columns: {result.summary['total_columns']}")
        logger.info(f"   - Errors: {result.summary['error_count']}")
        logger.info(f"   - Warnings: {result.summary['warning_count']}")
        
        # Log errors
        for error in result.errors:
            logger.error(f"❌ {error}")
        
        # Log warnings  
        for warning in result.warnings:
            logger.warning(f"⚠️  {warning}")
    
    def get_schema_info(self) -> Dict[str, Any]:
        """Get information about the current schema"""
        
        schema_info = {
            'total_columns': len(self.schema),
            'required_columns': [col for col, schema in self.schema.items() if not schema.nullable],
            'optional_columns': [col for col, schema in self.schema.items() if schema.nullable],
            'unique_columns': [col for col, schema in self.schema.items() if schema.unique],
            'constrained_columns': [col for col, schema in self.schema.items() 
                                  if schema.allowed_values or schema.min_value is not None or schema.max_value is not None],
            'column_details': {}
        }
        
        for col_name, col_schema in self.schema.items():
            schema_info['column_details'][col_name] = {
                'dtype': col_schema.dtype,
                'nullable': col_schema.nullable,
                'unique': col_schema.unique,
                'min_value': col_schema.min_value,
                'max_value': col_schema.max_value,
                'allowed_values': col_schema.allowed_values,
                'description': col_schema.description
            }
        
        return schema_info
    
    def export_validation_report(self, result: ValidationResult, filepath: str):
        """Export validation results to a report file"""
        
        report_content = f"""
# Titanic Dataset Validation Report
Generated: {result.timestamp}

## Summary
- **Validation Status**: {'✅ PASSED' if result.is_valid else '❌ FAILED'}
- **Total Rows**: {result.summary['total_rows']}
- **Total Columns**: {result.summary['total_columns']}  
- **Errors**: {result.summary['error_count']}
- **Warnings**: {result.summary['warning_count']}

## Errors ({len(result.errors)})
"""
        
        for i, error in enumerate(result.errors, 1):
            report_content += f"{i}. {error}\n"
        
        report_content += f"\n## Warnings ({len(result.warnings)})\n"
        
        for i, warning in enumerate(result.warnings, 1):
            report_content += f"{i}. {warning}\n"
        
        report_content += "\n## Column Statistics\n"
        
        for col, stats in result.summary.items():
            if col.endswith('_stats'):
                col_name = col.replace('_stats', '')
                report_content += f"\n### {col_name}\n"
                for key, value in stats.items():
                    report_content += f"- {key}: {value}\n"
        
        with open(filepath, 'w') as f:
            f.write(report_content)
        
        logger.info(f"📄 Validation report saved to: {filepath}")


if __name__ == "__main__":
    # Demo usage
    import sys
    import os
    
    # Add src to path
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    
    # Create validator
    validator = SchemaValidator()
    
    # Show schema info
    schema_info = validator.get_schema_info()
    print("📋 Titanic Dataset Schema Information:")
    print(f"   Total columns: {schema_info['total_columns']}")
    print(f"   Required columns: {len(schema_info['required_columns'])}")
    print(f"   Optional columns: {len(schema_info['optional_columns'])}")
    
    # Create sample valid data
    sample_data = pd.DataFrame({
        'PassengerId': [1, 2, 3],
        'Survived': [0, 1, 1],
        'Pclass': [3, 1, 3],
        'Name': ['Smith, Mr. John', 'Johnson, Mrs. Mary', 'Brown, Miss. Sarah'],
        'Sex': ['male', 'female', 'female'],
        'Age': [22.0, 38.0, 26.0],
        'SibSp': [1, 1, 0],
        'Parch': [0, 0, 0],
        'Ticket': ['A001', 'B002', 'C003'],
        'Fare': [7.25, 71.28, 7.92],
        'Cabin': [np.nan, 'C85', np.nan],
        'Embarked': ['S', 'C', 'S']
    })
    
    print("\n🔍 Validating sample data...")
    result = validator.validate_dataframe(sample_data)
    
    print(f"\n📊 Validation Result: {'✅ PASSED' if result.is_valid else '❌ FAILED'}")