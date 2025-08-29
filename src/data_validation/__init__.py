"""
Data Validation Module
======================

This module provides comprehensive data validation and quality monitoring
capabilities for the Titanic ML project.

Components:
- Schema validation
- Data quality checks  
- Statistical validation
- Data drift detection
- Outlier detection
- Data profiling and reporting

Author: Enhanced Titanic ML Framework
"""

from .validator import DataValidator
from .quality_monitor import DataQualityMonitor
from .drift_detector import DataDriftDetector
from .outlier_detector import OutlierDetector
from .schema_validator import SchemaValidator

__all__ = [
    'DataValidator',
    'DataQualityMonitor', 
    'DataDriftDetector',
    'OutlierDetector',
    'SchemaValidator'
]

__version__ = '1.0.0'