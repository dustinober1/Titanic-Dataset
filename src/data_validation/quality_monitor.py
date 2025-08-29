#!/usr/bin/env python3
"""
Data Quality Monitoring System
==============================

Comprehensive data quality monitoring for Titanic ML pipeline including:
- Real-time quality metrics tracking
- Automated quality alerts
- Quality trend analysis
- Data quality dashboards
- Quality rule engine

Author: Enhanced Titanic ML Framework
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import json
import warnings
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class QualityMetrics:
    """Container for data quality metrics"""
    timestamp: datetime
    dataset_name: str
    total_rows: int
    total_columns: int
    completeness_score: float
    accuracy_score: float
    consistency_score: float
    validity_score: float
    overall_quality_score: float
    issues_detected: int
    critical_issues: int
    warnings_count: int
    metrics_details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QualityRule:
    """Definition of a data quality rule"""
    name: str
    description: str
    rule_type: str  # 'completeness', 'accuracy', 'consistency', 'validity'
    severity: str   # 'critical', 'major', 'minor', 'info'
    threshold: float
    check_function: Callable
    enabled: bool = True
    tags: List[str] = field(default_factory=list)


@dataclass
class QualityIssue:
    """Container for quality issues"""
    rule_name: str
    severity: str
    description: str
    affected_records: int
    affected_columns: List[str]
    metric_value: float
    threshold: float
    timestamp: datetime
    details: Dict[str, Any] = field(default_factory=dict)


class DataQualityMonitor:
    """
    Comprehensive data quality monitoring system
    
    Tracks multiple dimensions of data quality:
    - Completeness: Missing/null values
    - Accuracy: Data correctness and precision  
    - Consistency: Data uniformity and standards
    - Validity: Data format and business rule compliance
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize quality monitor with configuration"""
        self.config = config or self._default_config()
        self.rules = self._initialize_quality_rules()
        self.quality_history: List[QualityMetrics] = []
        self.alerts_enabled = self.config.get('alerts_enabled', True)
        self.baseline_metrics: Optional[QualityMetrics] = None
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for quality monitoring"""
        return {
            'completeness_threshold': 0.95,  # 95% completeness required
            'accuracy_threshold': 0.90,      # 90% accuracy required  
            'consistency_threshold': 0.85,   # 85% consistency required
            'validity_threshold': 0.95,      # 95% validity required
            'overall_threshold': 0.90,       # 90% overall quality required
            'alerts_enabled': True,
            'trend_window_days': 7,
            'quality_report_frequency': 'daily',
            'critical_issue_threshold': 5,
            'auto_remediation': False
        }
    
    def _initialize_quality_rules(self) -> List[QualityRule]:
        """Initialize predefined quality rules for Titanic dataset"""
        
        rules = []
        
        # Completeness Rules
        rules.append(QualityRule(
            name='passenger_id_completeness',
            description='PassengerId should be 100% complete',
            rule_type='completeness',
            severity='critical',
            threshold=1.0,
            check_function=lambda df: self._check_column_completeness(df, 'PassengerId'),
            tags=['required', 'identifier']
        ))
        
        rules.append(QualityRule(
            name='survival_completeness',
            description='Survived field should be 100% complete',
            rule_type='completeness',
            severity='critical', 
            threshold=1.0,
            check_function=lambda df: self._check_column_completeness(df, 'Survived'),
            tags=['required', 'target']
        ))
        
        rules.append(QualityRule(
            name='age_completeness',
            description='Age should be at least 75% complete',
            rule_type='completeness',
            severity='major',
            threshold=0.75,
            check_function=lambda df: self._check_column_completeness(df, 'Age'),
            tags=['demographic']
        ))
        
        # Accuracy Rules
        rules.append(QualityRule(
            name='age_range_accuracy',
            description='Age values should be between 0 and 120',
            rule_type='accuracy',
            severity='critical',
            threshold=0.99,
            check_function=lambda df: self._check_age_range(df),
            tags=['range_check', 'demographic']
        ))
        
        rules.append(QualityRule(
            name='fare_accuracy',
            description='Fare values should be non-negative and reasonable',
            rule_type='accuracy',
            severity='major',
            threshold=0.95,
            check_function=lambda df: self._check_fare_accuracy(df),
            tags=['monetary', 'range_check']
        ))
        
        # Consistency Rules
        rules.append(QualityRule(
            name='passenger_class_consistency',
            description='Passenger class should be 1, 2, or 3',
            rule_type='consistency',
            severity='critical',
            threshold=1.0,
            check_function=lambda df: self._check_pclass_consistency(df),
            tags=['categorical', 'business_rule']
        ))
        
        rules.append(QualityRule(
            name='gender_consistency',
            description='Sex should be male or female',
            rule_type='consistency',
            severity='critical',
            threshold=1.0,
            check_function=lambda df: self._check_gender_consistency(df),
            tags=['categorical', 'demographic']
        ))
        
        rules.append(QualityRule(
            name='embarked_consistency',
            description='Embarked should be S, C, or Q',
            rule_type='consistency',
            severity='major',
            threshold=0.98,
            check_function=lambda df: self._check_embarked_consistency(df),
            tags=['categorical', 'historical']
        ))
        
        # Validity Rules
        rules.append(QualityRule(
            name='passenger_id_uniqueness',
            description='PassengerId should be unique',
            rule_type='validity',
            severity='critical',
            threshold=1.0,
            check_function=lambda df: self._check_passenger_id_uniqueness(df),
            tags=['uniqueness', 'identifier']
        ))
        
        rules.append(QualityRule(
            name='family_size_validity',
            description='Family size (SibSp + Parch) should be reasonable',
            rule_type='validity',
            severity='minor',
            threshold=0.95,
            check_function=lambda df: self._check_family_size_validity(df),
            tags=['business_logic', 'relationship']
        ))
        
        rules.append(QualityRule(
            name='name_format_validity',
            description='Name should follow expected format',
            rule_type='validity',
            severity='minor',
            threshold=0.90,
            check_function=lambda df: self._check_name_format(df),
            tags=['format', 'text']
        ))
        
        return rules
    
    def assess_quality(self, df: pd.DataFrame, dataset_name: str = "titanic") -> QualityMetrics:
        """
        Perform comprehensive quality assessment
        
        Args:
            df: DataFrame to assess
            dataset_name: Name of the dataset
            
        Returns:
            QualityMetrics with detailed assessment results
        """
        logger.info(f"🔍 Starting quality assessment for {dataset_name} dataset ({len(df)} rows)")
        
        # Run all quality rules
        issues = []
        rule_results = {}
        
        for rule in self.rules:
            if rule.enabled:
                try:
                    result = rule.check_function(df)
                    rule_results[rule.name] = result
                    
                    if result < rule.threshold:
                        issue = QualityIssue(
                            rule_name=rule.name,
                            severity=rule.severity,
                            description=rule.description,
                            affected_records=self._calculate_affected_records(df, rule, result),
                            affected_columns=self._get_affected_columns(rule),
                            metric_value=result,
                            threshold=rule.threshold,
                            timestamp=datetime.now(),
                            details={'rule_type': rule.rule_type, 'tags': rule.tags}
                        )\n                        issues.append(issue)\n                        \nexcept Exception as e:\n                    logger.error(f\"Error running quality rule '{rule.name}': {str(e)}\")\n                    rule_results[rule.name] = 0.0\n        \n        # Calculate dimension scores\n        completeness_score = self._calculate_dimension_score(rule_results, 'completeness')\n        accuracy_score = self._calculate_dimension_score(rule_results, 'accuracy')\n        consistency_score = self._calculate_dimension_score(rule_results, 'consistency')\n        validity_score = self._calculate_dimension_score(rule_results, 'validity')\n        \n        # Calculate overall quality score\n        overall_score = np.mean([completeness_score, accuracy_score, consistency_score, validity_score])\n        \n        # Count issues by severity\n        critical_issues = sum(1 for issue in issues if issue.severity == 'critical')\n        warnings_count = sum(1 for issue in issues if issue.severity in ['minor', 'info'])\n        \n        # Create metrics object\n        metrics = QualityMetrics(\n            timestamp=datetime.now(),\n            dataset_name=dataset_name,\n            total_rows=len(df),\n            total_columns=len(df.columns),\n            completeness_score=completeness_score,\n            accuracy_score=accuracy_score,\n            consistency_score=consistency_score,\n            validity_score=validity_score,\n            overall_quality_score=overall_score,\n            issues_detected=len(issues),\n            critical_issues=critical_issues,\n            warnings_count=warnings_count,\n            metrics_details={\n                'rule_results': rule_results,\n                'issues': [self._issue_to_dict(issue) for issue in issues],\n                'basic_stats': self._calculate_basic_stats(df)\n            }\n        )\n        \n        # Store in history\n        self.quality_history.append(metrics)\n        \n        # Set baseline if not exists\n        if self.baseline_metrics is None:\n            self.baseline_metrics = metrics\n            logger.info(\"📊 Baseline metrics established\")\n        \n        # Generate alerts if enabled\n        if self.alerts_enabled:\n            self._generate_alerts(metrics, issues)\n        \n        # Log results\n        self._log_quality_results(metrics)\n        \n        return metrics\n    \n    def _check_column_completeness(self, df: pd.DataFrame, column: str) -> float:\n        \"\"\"Check completeness of a specific column\"\"\"\n        if column not in df.columns:\n            return 0.0\n        return 1.0 - (df[column].isnull().sum() / len(df))\n    \n    def _check_age_range(self, df: pd.DataFrame) -> float:\n        \"\"\"Check if age values are in valid range\"\"\"\n        if 'Age' not in df.columns:\n            return 0.0\n        \n        age_data = df['Age'].dropna()\n        if len(age_data) == 0:\n            return 1.0\n        \n        valid_ages = ((age_data >= 0) & (age_data <= 120))\n        return valid_ages.mean()\n    \n    def _check_fare_accuracy(self, df: pd.DataFrame) -> float:\n        \"\"\"Check if fare values are reasonable\"\"\"\n        if 'Fare' not in df.columns:\n            return 0.0\n        \n        fare_data = df['Fare'].dropna()\n        if len(fare_data) == 0:\n            return 1.0\n        \n        # Check for non-negative and reasonable values (< $1000)\n        valid_fares = ((fare_data >= 0) & (fare_data <= 1000))\n        return valid_fares.mean()\n    \n    def _check_pclass_consistency(self, df: pd.DataFrame) -> float:\n        \"\"\"Check passenger class consistency\"\"\"\n        if 'Pclass' not in df.columns:\n            return 0.0\n        \n        pclass_data = df['Pclass'].dropna()\n        if len(pclass_data) == 0:\n            return 1.0\n        \n        valid_classes = pclass_data.isin([1, 2, 3])\n        return valid_classes.mean()\n    \n    def _check_gender_consistency(self, df: pd.DataFrame) -> float:\n        \"\"\"Check gender field consistency\"\"\"\n        if 'Sex' not in df.columns:\n            return 0.0\n        \n        sex_data = df['Sex'].dropna()\n        if len(sex_data) == 0:\n            return 1.0\n        \n        valid_genders = sex_data.isin(['male', 'female'])\n        return valid_genders.mean()\n    \n    def _check_embarked_consistency(self, df: pd.DataFrame) -> float:\n        \"\"\"Check embarked port consistency\"\"\"\n        if 'Embarked' not in df.columns:\n            return 0.0\n        \n        embarked_data = df['Embarked'].dropna()\n        if len(embarked_data) == 0:\n            return 1.0\n        \n        valid_ports = embarked_data.isin(['S', 'C', 'Q'])\n        return valid_ports.mean()\n    \n    def _check_passenger_id_uniqueness(self, df: pd.DataFrame) -> float:\n        \"\"\"Check passenger ID uniqueness\"\"\"\n        if 'PassengerId' not in df.columns:\n            return 0.0\n        \n        passenger_ids = df['PassengerId'].dropna()\n        if len(passenger_ids) == 0:\n            return 1.0\n        \n        return 1.0 - (passenger_ids.duplicated().sum() / len(passenger_ids))\n    \n    def _check_family_size_validity(self, df: pd.DataFrame) -> float:\n        \"\"\"Check family size reasonableness\"\"\"\n        if 'SibSp' not in df.columns or 'Parch' not in df.columns:\n            return 0.0\n        \n        family_size = df['SibSp'] + df['Parch'] + 1\n        reasonable_size = family_size <= 15  # Reasonable family size limit\n        return reasonable_size.mean()\n    \n    def _check_name_format(self, df: pd.DataFrame) -> float:\n        \"\"\"Check name format validity\"\"\"\n        if 'Name' not in df.columns:\n            return 0.0\n        \n        names = df['Name'].dropna()\n        if len(names) == 0:\n            return 1.0\n        \n        # Check if names contain comma and period (typical format)\n        valid_format = names.str.contains(r'.*, .+\\..+', na=False)\n        return valid_format.mean()\n    \n    def _calculate_dimension_score(self, rule_results: Dict[str, float], dimension: str) -> float:\n        \"\"\"Calculate score for a quality dimension\"\"\"\n        dimension_rules = [rule for rule in self.rules if rule.rule_type == dimension and rule.enabled]\n        \n        if not dimension_rules:\n            return 1.0\n        \n        scores = [rule_results.get(rule.name, 0.0) for rule in dimension_rules]\n        return np.mean(scores)\n    \n    def _calculate_affected_records(self, df: pd.DataFrame, rule: QualityRule, result: float) -> int:\n        \"\"\"Calculate number of records affected by a quality issue\"\"\"\n        # Simplified calculation - can be enhanced based on specific rule\n        return int(len(df) * (1.0 - result))\n    \n    def _get_affected_columns(self, rule: QualityRule) -> List[str]:\n        \"\"\"Get columns affected by a quality rule\"\"\"\n        # Simple mapping - can be enhanced\n        column_mapping = {\n            'passenger_id_completeness': ['PassengerId'],\n            'survival_completeness': ['Survived'],\n            'age_completeness': ['Age'],\n            'age_range_accuracy': ['Age'],\n            'fare_accuracy': ['Fare'],\n            'passenger_class_consistency': ['Pclass'],\n            'gender_consistency': ['Sex'],\n            'embarked_consistency': ['Embarked'],\n            'passenger_id_uniqueness': ['PassengerId'],\n            'family_size_validity': ['SibSp', 'Parch'],\n            'name_format_validity': ['Name']\n        }\n        return column_mapping.get(rule.name, [])\n    \n    def _calculate_basic_stats(self, df: pd.DataFrame) -> Dict[str, Any]:\n        \"\"\"Calculate basic dataset statistics\"\"\"\n        return {\n            'row_count': len(df),\n            'column_count': len(df.columns),\n            'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024**2),\n            'null_counts': df.isnull().sum().to_dict(),\n            'dtypes': df.dtypes.astype(str).to_dict(),\n            'duplicate_rows': df.duplicated().sum()\n        }\n    \n    def _issue_to_dict(self, issue: QualityIssue) -> Dict[str, Any]:\n        \"\"\"Convert QualityIssue to dictionary\"\"\"\n        return {\n            'rule_name': issue.rule_name,\n            'severity': issue.severity,\n            'description': issue.description,\n            'affected_records': issue.affected_records,\n            'affected_columns': issue.affected_columns,\n            'metric_value': issue.metric_value,\n            'threshold': issue.threshold,\n            'timestamp': issue.timestamp.isoformat(),\n            'details': issue.details\n        }\n    \n    def _generate_alerts(self, metrics: QualityMetrics, issues: List[QualityIssue]):\n        \"\"\"Generate quality alerts\"\"\"\n        \n        # Critical quality alert\n        if metrics.overall_quality_score < self.config['overall_threshold']:\n            logger.error(\n                f\"🚨 CRITICAL: Overall quality score {metrics.overall_quality_score:.2%} \"\n                f\"below threshold {self.config['overall_threshold']:.1%}\"\n            )\n        \n        # Critical issues alert\n        if metrics.critical_issues >= self.config['critical_issue_threshold']:\n            logger.error(\n                f\"🚨 CRITICAL: {metrics.critical_issues} critical issues detected, \"\n                f\"exceeds threshold of {self.config['critical_issue_threshold']}\"\n            )\n        \n        # Dimension-specific alerts\n        if metrics.completeness_score < self.config['completeness_threshold']:\n            logger.warning(\n                f\"⚠️  Data completeness {metrics.completeness_score:.2%} below threshold\"\n            )\n        \n        if metrics.accuracy_score < self.config['accuracy_threshold']:\n            logger.warning(\n                f\"⚠️  Data accuracy {metrics.accuracy_score:.2%} below threshold\"\n            )\n    \n    def _log_quality_results(self, metrics: QualityMetrics):\n        \"\"\"Log quality assessment results\"\"\"\n        \n        logger.info(f\"📊 Quality Assessment Results for {metrics.dataset_name}:\")\n        logger.info(f\"   Overall Score: {metrics.overall_quality_score:.2%}\")\n        logger.info(f\"   Completeness: {metrics.completeness_score:.2%}\")\n        logger.info(f\"   Accuracy: {metrics.accuracy_score:.2%}\")\n        logger.info(f\"   Consistency: {metrics.consistency_score:.2%}\")\n        logger.info(f\"   Validity: {metrics.validity_score:.2%}\")\n        logger.info(f\"   Issues: {metrics.issues_detected} ({metrics.critical_issues} critical)\")\n    \n    def get_quality_trend(self, days: int = 7) -> Dict[str, List[float]]:\n        \"\"\"Get quality trends over specified time period\"\"\"\n        \n        cutoff_date = datetime.now() - timedelta(days=days)\n        recent_metrics = [\n            m for m in self.quality_history \n            if m.timestamp >= cutoff_date\n        ]\n        \n        if not recent_metrics:\n            return {}\n        \n        trends = {\n            'timestamps': [m.timestamp.isoformat() for m in recent_metrics],\n            'overall_score': [m.overall_quality_score for m in recent_metrics],\n            'completeness': [m.completeness_score for m in recent_metrics],\n            'accuracy': [m.accuracy_score for m in recent_metrics],\n            'consistency': [m.consistency_score for m in recent_metrics],\n            'validity': [m.validity_score for m in recent_metrics],\n            'issues_count': [m.issues_detected for m in recent_metrics],\n            'critical_issues': [m.critical_issues for m in recent_metrics]\n        }\n        \n        return trends\n    \n    def export_quality_report(self, metrics: QualityMetrics, filepath: str):\n        \"\"\"Export detailed quality report\"\"\"\n        \n        report = {\n            'report_metadata': {\n                'generated_at': datetime.now().isoformat(),\n                'dataset_name': metrics.dataset_name,\n                'assessment_timestamp': metrics.timestamp.isoformat()\n            },\n            'quality_summary': {\n                'overall_score': metrics.overall_quality_score,\n                'completeness_score': metrics.completeness_score,\n                'accuracy_score': metrics.accuracy_score,\n                'consistency_score': metrics.consistency_score,\n                'validity_score': metrics.validity_score,\n                'total_rows': metrics.total_rows,\n                'total_columns': metrics.total_columns,\n                'issues_detected': metrics.issues_detected,\n                'critical_issues': metrics.critical_issues,\n                'warnings_count': metrics.warnings_count\n            },\n            'detailed_results': metrics.metrics_details,\n            'quality_thresholds': self.config,\n            'rules_applied': [\n                {\n                    'name': rule.name,\n                    'description': rule.description,\n                    'type': rule.rule_type,\n                    'severity': rule.severity,\n                    'threshold': rule.threshold,\n                    'enabled': rule.enabled,\n                    'tags': rule.tags\n                }\n                for rule in self.rules\n            ]\n        }\n        \n        with open(filepath, 'w') as f:\n            json.dump(report, f, indent=2, default=str)\n        \n        logger.info(f\"📄 Quality report exported to: {filepath}\")\n    \n    def compare_with_baseline(self, current_metrics: QualityMetrics) -> Dict[str, Any]:\n        \"\"\"Compare current metrics with baseline\"\"\"\n        \n        if self.baseline_metrics is None:\n            return {'status': 'no_baseline', 'message': 'No baseline metrics available'}\n        \n        comparison = {\n            'baseline_timestamp': self.baseline_metrics.timestamp.isoformat(),\n            'current_timestamp': current_metrics.timestamp.isoformat(),\n            'score_changes': {\n                'overall': current_metrics.overall_quality_score - self.baseline_metrics.overall_quality_score,\n                'completeness': current_metrics.completeness_score - self.baseline_metrics.completeness_score,\n                'accuracy': current_metrics.accuracy_score - self.baseline_metrics.accuracy_score,\n                'consistency': current_metrics.consistency_score - self.baseline_metrics.consistency_score,\n                'validity': current_metrics.validity_score - self.baseline_metrics.validity_score\n            },\n            'issue_changes': {\n                'total_issues': current_metrics.issues_detected - self.baseline_metrics.issues_detected,\n                'critical_issues': current_metrics.critical_issues - self.baseline_metrics.critical_issues\n            },\n            'status': 'improved' if current_metrics.overall_quality_score > self.baseline_metrics.overall_quality_score else 'degraded'\n        }\n        \n        return comparison\n\n\nif __name__ == \"__main__\":\n    # Demo usage\n    import sys\n    import os\n    \n    # Add src to path\n    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))\n    \n    # Create quality monitor\n    monitor = DataQualityMonitor()\n    \n    # Create sample data\n    sample_data = pd.DataFrame({\n        'PassengerId': [1, 2, 3, 4, 5],\n        'Survived': [0, 1, 1, 1, 0],\n        'Pclass': [3, 1, 3, 1, 3],\n        'Name': ['Smith, Mr. John', 'Johnson, Mrs. Mary', 'Brown, Miss. Sarah', 'Davis, Mr. James', 'Wilson, Mrs. Emma'],\n        'Sex': ['male', 'female', 'female', 'male', 'female'],\n        'Age': [22.0, 38.0, 26.0, np.nan, 29.0],\n        'SibSp': [1, 1, 0, 1, 0],\n        'Parch': [0, 0, 0, 0, 1],\n        'Ticket': ['A001', 'B002', 'C003', 'D004', 'E005'],\n        'Fare': [7.25, 71.28, 7.92, 53.10, 8.05],\n        'Cabin': [np.nan, 'C85', np.nan, 'C123', 'G6'],\n        'Embarked': ['S', 'C', 'S', 'S', 'Q']\n    })\n    \n    print(\"🔍 Running quality assessment on sample data...\")\n    \n    # Assess quality\n    metrics = monitor.assess_quality(sample_data, \"sample_titanic\")\n    \n    print(f\"\\n📊 Quality Assessment Complete:\")\n    print(f\"   Overall Score: {metrics.overall_quality_score:.2%}\")\n    print(f\"   Issues Detected: {metrics.issues_detected}\")\n    print(f\"   Critical Issues: {metrics.critical_issues}\")"