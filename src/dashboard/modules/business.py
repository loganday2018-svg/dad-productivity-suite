"""
Business logic for P&L analytics.

Includes calendar helpers, anomaly detection, and variance calculations.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Month ordering
MONTH_ORDER = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'FY']


class CalendarHelper:
    """Helper for calendar and period calculations."""

    @staticmethod
    def month_to_number(month: str) -> int:
        """
        Convert month name to number.

        Args:
            month: Month name (Jan, Feb, etc.)

        Returns:
            Month number (1-12, 13 for FY)
        """
        if month == 'FY':
            return 13
        try:
            return MONTH_ORDER.index(month) + 1
        except ValueError:
            return 0

    @staticmethod
    def get_latest_month(df: pd.DataFrame, scenario: str = 'actual_cy', min_nonzero_accounts: int = 10) -> str:
        """
        Find the latest month with meaningful data.

        A month is considered to have meaningful data if it has at least
        min_nonzero_accounts with non-zero values.

        Args:
            df: Tidy DataFrame with columns: entity, account, scenario, month, value
            scenario: Which scenario to check (default: actual_cy)
            min_nonzero_accounts: Minimum number of accounts with non-zero data

        Returns:
            Latest month name
        """
        # Filter to scenario, exclude totals to avoid double-counting
        scenario_data = df[(df['scenario'] == scenario) & (~df['is_total'])]

        # For each month, count accounts with non-zero values
        month_quality = {}
        for month in MONTH_ORDER[:-1]:  # Exclude 'FY'
            month_data = scenario_data[scenario_data['month'] == month]
            nonzero_count = (month_data['value'].abs() > 0.01).sum()  # Allow small rounding errors
            month_quality[month] = nonzero_count

        # Find months that meet threshold
        valid_months = [m for m, count in month_quality.items() if count >= min_nonzero_accounts]

        # Sort by month order
        sorted_months = sorted(
            valid_months,
            key=lambda x: CalendarHelper.month_to_number(x)
        )

        if not sorted_months:
            logger.warning(f"No months found with sufficient data for scenario {scenario}")
            # Fallback to old logic
            months_with_data = scenario_data[scenario_data['value'].notna()]['month'].unique()
            sorted_months = sorted(
                [m for m in months_with_data if m != 'FY'],
                key=lambda x: CalendarHelper.month_to_number(x)
            )
            if not sorted_months:
                return 'Jan'

        latest = sorted_months[-1]
        logger.info(f"Latest month detected: {latest} (with {month_quality.get(latest, 0)} non-zero accounts)")
        return latest

    @staticmethod
    def get_trailing_months(end_month: str, n: int = 12) -> List[str]:
        """
        Get list of N trailing months ending at end_month.

        Args:
            end_month: Ending month name
            n: Number of months to include

        Returns:
            List of month names in chronological order
        """
        end_idx = CalendarHelper.month_to_number(end_month) - 1

        if end_idx < 0 or end_idx >= 12:
            return MONTH_ORDER[:12]

        # Simple trailing window (doesn't wrap years)
        start_idx = max(0, end_idx - n + 1)
        return MONTH_ORDER[start_idx:end_idx + 1]


class AnomalyDetector:
    """Detect anomalies in financial data using statistical methods."""

    def __init__(self, z_threshold: float = 2.0, min_periods: int = 6):
        """
        Initialize detector.

        Args:
            z_threshold: Z-score threshold for flagging anomalies
            min_periods: Minimum data points required for detection
        """
        self.z_threshold = z_threshold
        self.min_periods = min_periods

    def calculate_zscore(self, series: pd.Series) -> pd.Series:
        """
        Calculate rolling z-scores.

        Args:
            series: Time series of values

        Returns:
            Series of z-scores
        """
        # Rolling statistics over trailing 12 months
        rolling_mean = series.rolling(window=12, min_periods=self.min_periods).mean()
        rolling_std = series.rolling(window=12, min_periods=self.min_periods).std()

        # Calculate z-score
        z_scores = (series - rolling_mean) / rolling_std

        return z_scores

    def detect_anomalies(
        self,
        df: pd.DataFrame,
        exclude_totals: bool = True
    ) -> pd.DataFrame:
        """
        Detect anomalies across all entities and accounts.

        Args:
            df: Tidy DataFrame with entity, account, scenario, month, value, is_total
            exclude_totals: Whether to exclude total rows from detection

        Returns:
            DataFrame with anomalies and metadata:
                entity, account, month, value, z_score, trailing_mean, trailing_std,
                py_value, delta_vs_mean, category
        """
        # Filter to actual CY data
        cy_data = df[df['scenario'] == 'actual_cy'].copy()

        # Exclude totals if requested
        if exclude_totals:
            cy_data = cy_data[~cy_data['is_total']]

        # Get PY data for comparison
        py_data = df[df['scenario'] == 'actual_py'][['entity', 'account', 'month', 'value']]
        py_data = py_data.rename(columns={'value': 'py_value'})

        # Sort by month order
        cy_data['month_num'] = cy_data['month'].apply(CalendarHelper.month_to_number)
        cy_data = cy_data.sort_values(['entity', 'account', 'month_num'])

        anomalies = []

        # Group by entity and account
        for (entity, account), group in cy_data.groupby(['entity', 'account']):
            if len(group) < self.min_periods:
                continue

            # Calculate z-scores
            group = group.copy()
            group['z_score'] = self.calculate_zscore(group['value'])

            # Calculate rolling statistics
            group['trailing_mean'] = group['value'].rolling(
                window=12, min_periods=self.min_periods
            ).mean()
            group['trailing_std'] = group['value'].rolling(
                window=12, min_periods=self.min_periods
            ).std()
            group['delta_vs_mean'] = group['value'] - group['trailing_mean']

            # Filter to anomalies
            anomaly_rows = group[group['z_score'].abs() >= self.z_threshold]

            for _, row in anomaly_rows.iterrows():
                # Get PY value for same month
                py_match = py_data[
                    (py_data['entity'] == entity) &
                    (py_data['account'] == account) &
                    (py_data['month'] == row['month'])
                ]
                py_value = py_match['py_value'].values[0] if len(py_match) > 0 else None

                anomalies.append({
                    'entity': entity,
                    'account': account,
                    'month': row['month'],
                    'value': row['value'],
                    'z_score': row['z_score'],
                    'trailing_mean': row['trailing_mean'],
                    'trailing_std': row['trailing_std'],
                    'delta_vs_mean': row['delta_vs_mean'],
                    'py_value': py_value,
                    'category': row['category']
                })

        anomaly_df = pd.DataFrame(anomalies)
        logger.info(f"Detected {len(anomaly_df)} anomalies")
        return anomaly_df

    def generate_explanation(self, anomaly: pd.Series) -> str:
        """
        Generate human-readable explanation for an anomaly.

        Args:
            anomaly: Series containing anomaly data

        Returns:
            Explanation string
        """
        account = anomaly['account']
        entity = anomaly['entity']
        value = anomaly['value']
        delta = anomaly['delta_vs_mean']
        avg = anomaly['trailing_mean']
        z = anomaly['z_score']
        py_value = anomaly.get('py_value', None)

        explanation = (
            f"{account} at {entity} = ${value:,.0f}; "
            f"${delta:+,.0f} vs 12-mo avg ${avg:,.0f}; "
            f"z-score {z:.2f}"
        )

        if py_value is not None:
            py_delta = value - py_value
            explanation += f"; prior year same month ${py_value:,.0f} ({py_delta:+,.0f})"

        return explanation


def add_month_ordering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add month_num column for sorting.

    Args:
        df: DataFrame with 'month' column

    Returns:
        DataFrame with added month_num column
    """
    df = df.copy()
    df['month_num'] = df['month'].apply(CalendarHelper.month_to_number)
    return df


def filter_to_latest_month(df: pd.DataFrame, latest_month: str) -> pd.DataFrame:
    """
    Filter DataFrame to latest month.

    Args:
        df: DataFrame with 'month' column
        latest_month: Month to filter to

    Returns:
        Filtered DataFrame
    """
    return df[df['month'] == latest_month].copy()


def calculate_yoy_delta(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate year-over-year deltas.

    Args:
        df: Tidy DataFrame with actual_cy and actual_py scenarios

    Returns:
        DataFrame with yoy_delta column
    """
    # Pivot to get CY and PY side by side
    pivot = df.pivot_table(
        index=['entity', 'account', 'month'],
        columns='scenario',
        values='value',
        aggfunc='first'
    ).reset_index()

    # Calculate delta
    if 'actual_cy' in pivot.columns and 'actual_py' in pivot.columns:
        pivot['yoy_delta'] = pivot['actual_cy'] - pivot['actual_py']
        pivot['yoy_pct'] = (pivot['yoy_delta'] / pivot['actual_py'].abs()) * 100

    return pivot
