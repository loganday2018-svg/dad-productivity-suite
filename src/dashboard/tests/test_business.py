"""
Unit tests for business logic module.
"""

import pytest
import pandas as pd
import numpy as np
from modules.business import CalendarHelper, AnomalyDetector, add_month_ordering


class TestCalendarHelper:
    """Tests for CalendarHelper class."""

    def test_month_to_number(self):
        """Test month name to number conversion."""
        assert CalendarHelper.month_to_number('Jan') == 1
        assert CalendarHelper.month_to_number('Dec') == 12
        assert CalendarHelper.month_to_number('FY') == 13
        assert CalendarHelper.month_to_number('Invalid') == 0

    def test_get_latest_month(self):
        """Test latest month detection."""
        data = pd.DataFrame({
            'entity': ['E1', 'E1', 'E1'],
            'account': ['A1', 'A1', 'A1'],
            'scenario': ['actual_cy', 'actual_cy', 'actual_cy'],
            'month': ['Jan', 'Feb', 'Mar'],
            'value': [100, 200, 300]
        })

        latest = CalendarHelper.get_latest_month(data)
        assert latest == 'Mar'

    def test_get_latest_month_with_nulls(self):
        """Test latest month detection with null values."""
        data = pd.DataFrame({
            'entity': ['E1', 'E1', 'E1', 'E1'],
            'account': ['A1', 'A1', 'A1', 'A1'],
            'scenario': ['actual_cy', 'actual_cy', 'actual_cy', 'actual_cy'],
            'month': ['Jan', 'Feb', 'Mar', 'Apr'],
            'value': [100, 200, 300, np.nan]
        })

        latest = CalendarHelper.get_latest_month(data)
        assert latest == 'Mar'

    def test_get_trailing_months(self):
        """Test trailing month generation."""
        trailing = CalendarHelper.get_trailing_months('Jun', 3)
        assert trailing == ['Apr', 'May', 'Jun']

        trailing = CalendarHelper.get_trailing_months('Mar', 5)
        assert trailing == ['Jan', 'Feb', 'Mar']


class TestAnomalyDetector:
    """Tests for AnomalyDetector class."""

    def test_calculate_zscore(self):
        """Test z-score calculation."""
        detector = AnomalyDetector(z_threshold=2.0)

        # Create series with clear outlier
        values = pd.Series([100, 110, 105, 95, 100, 105, 100, 110, 95, 100, 105, 500])

        z_scores = detector.calculate_zscore(values)

        # Last value should have high z-score
        assert abs(z_scores.iloc[-1]) > 2.0

        # Earlier values should be within normal range
        assert abs(z_scores.iloc[6]) < 2.0

    def test_detect_anomalies(self):
        """Test anomaly detection."""
        detector = AnomalyDetector(z_threshold=2.0, min_periods=6)

        # Create test data with anomaly
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        values = [100, 110, 105, 95, 100, 105, 100, 110, 95, 100, 105, 500]

        data = pd.DataFrame({
            'entity': ['E1'] * 12,
            'account': ['Sales'] * 12,
            'scenario': ['actual_cy'] * 12,
            'month': months,
            'value': values,
            'is_total': [False] * 12,
            'category': ['Revenue & Throughput'] * 12
        })

        anomalies = detector.detect_anomalies(data, exclude_totals=True)

        # Should detect the outlier in December
        assert len(anomalies) > 0
        assert 'Dec' in anomalies['month'].values

    def test_detect_anomalies_excludes_totals(self):
        """Test that totals are excluded from anomaly detection."""
        detector = AnomalyDetector(z_threshold=2.0)

        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        values = [100, 110, 105, 95, 100, 105, 100, 110, 95, 100, 105, 500]

        data = pd.DataFrame({
            'entity': ['E1'] * 12,
            'account': ['TOTAL Sales'] * 12,
            'scenario': ['actual_cy'] * 12,
            'month': months,
            'value': values,
            'is_total': [True] * 12,
            'category': ['Revenue & Throughput'] * 12
        })

        anomalies = detector.detect_anomalies(data, exclude_totals=True)

        # Should not detect any anomalies since all are totals
        assert len(anomalies) == 0

    def test_generate_explanation(self):
        """Test explanation generation."""
        detector = AnomalyDetector()

        anomaly = pd.Series({
            'account': 'Sales',
            'entity': 'RWW',
            'value': 500,
            'delta_vs_mean': 400,
            'trailing_mean': 100,
            'z_score': 3.5,
            'py_value': 120
        })

        explanation = detector.generate_explanation(anomaly)

        assert 'Sales' in explanation
        assert 'RWW' in explanation
        assert '500' in explanation
        assert '3.5' in explanation
        assert '120' in explanation


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_add_month_ordering(self):
        """Test month ordering addition."""
        data = pd.DataFrame({
            'month': ['Jan', 'Mar', 'Feb', 'FY']
        })

        result = add_month_ordering(data)

        assert 'month_num' in result.columns
        assert result.loc[result['month'] == 'Jan', 'month_num'].values[0] == 1
        assert result.loc[result['month'] == 'Mar', 'month_num'].values[0] == 3
        assert result.loc[result['month'] == 'FY', 'month_num'].values[0] == 13
