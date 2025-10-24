"""
KPI aggregation and enterprise-wide calculations.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KPIAggregator:
    """Aggregate KPIs across entities."""

    def __init__(self, df: pd.DataFrame):
        """
        Initialize aggregator.

        Args:
            df: Tidy DataFrame with entity, account, scenario, month, value, category
        """
        self.df = df

    def aggregate_enterprise(self, month: str, scenario: str = 'actual_cy') -> pd.DataFrame:
        """
        Aggregate all entities to enterprise level.

        Args:
            month: Month to aggregate
            scenario: Scenario to use

        Returns:
            DataFrame with enterprise-level accounts
        """
        # Filter to month and scenario
        filtered = self.df[
            (self.df['month'] == month) &
            (self.df['scenario'] == scenario)
        ]

        # Sum across entities
        enterprise = filtered.groupby(['account', 'category', 'is_total'])['value'].sum().reset_index()
        enterprise['entity'] = 'Enterprise (All)'
        enterprise['month'] = month
        enterprise['scenario'] = scenario

        logger.info(f"Aggregated {len(enterprise)} enterprise accounts for {month}")
        return enterprise

    def calculate_kpis(self, entity: str, month: str, scenario: str = 'actual_cy') -> Dict[str, float]:
        """
        Calculate key financial metrics for an entity.

        Args:
            entity: Entity name or 'Enterprise (All)'
            month: Month to calculate
            scenario: Scenario to use

        Returns:
            Dict of KPI names to values
        """
        if entity == 'Enterprise (All)':
            data = self.aggregate_enterprise(month, scenario)
        else:
            data = self.df[
                (self.df['entity'] == entity) &
                (self.df['month'] == month) &
                (self.df['scenario'] == scenario)
            ]

        kpis = {}

        # Helper to get total by keyword - prioritize TOTAL rows
        def get_total(*keywords: str) -> Optional[float]:
            for keyword in keywords:
                # First try to find total rows with this keyword
                total_matches = data[
                    data['account'].str.contains(keyword, case=False, na=False) &
                    data['is_total']
                ]
                if len(total_matches) > 0:
                    return total_matches['value'].iloc[0]

                # Fallback to any row with keyword
                matches = data[data['account'].str.contains(keyword, case=False, na=False)]
                if len(matches) > 0:
                    return matches['value'].iloc[0]
            return None

        # Revenue & Throughput - prioritize "Total Net Sales" or "Net Sales"
        revenue = get_total('Total Net Sales', 'Net Sales', 'Total Sales', 'Sales', 'Revenue')
        if revenue:
            kpis['revenue'] = revenue

        # Gross Profit
        gross_profit = get_total('Gross Profit')
        if gross_profit and revenue:
            kpis['gross_profit'] = gross_profit
            kpis['gross_margin_pct'] = (gross_profit / revenue) * 100

        # EBITDA
        ebitda = get_total('EBITDA')
        if ebitda:
            kpis['ebitda'] = ebitda
            if revenue:
                kpis['ebitda_pct'] = (ebitda / revenue) * 100

        # Operating Income
        operating_income = get_total('Operating Income')
        if operating_income:
            kpis['operating_income'] = operating_income
            if revenue:
                kpis['operating_margin_pct'] = (operating_income / revenue) * 100

        return kpis

    def get_category_summary(
        self,
        entity: str,
        month: str,
        category: str,
        scenario: str = 'actual_cy'
    ) -> pd.DataFrame:
        """
        Get all accounts in a category for an entity/month.

        Args:
            entity: Entity name
            month: Month
            category: Category name (e.g., 'Operating Expenses')
            scenario: Scenario

        Returns:
            DataFrame of accounts in category
        """
        if entity == 'Enterprise (All)':
            data = self.aggregate_enterprise(month, scenario)
        else:
            data = self.df[
                (self.df['entity'] == entity) &
                (self.df['month'] == month) &
                (self.df['scenario'] == scenario)
            ]

        return data[data['category'] == category].copy()

    def calculate_rolling_metrics(
        self,
        entity: str,
        account: str,
        months: list,
        scenario: str = 'actual_cy'
    ) -> pd.DataFrame:
        """
        Calculate rolling metrics for an account.

        Args:
            entity: Entity name
            account: Account name
            months: List of months in order
            scenario: Scenario

        Returns:
            DataFrame with month, value, rolling_avg
        """
        data = self.df[
            (self.df['entity'] == entity) &
            (self.df['account'] == account) &
            (self.df['scenario'] == scenario) &
            (self.df['month'].isin(months))
        ].copy()

        # Sort by month order
        month_order_map = {m: i for i, m in enumerate(months)}
        data['month_order'] = data['month'].map(month_order_map)
        data = data.sort_values('month_order')

        # Calculate rolling average
        data['rolling_avg'] = data['value'].rolling(window=3, min_periods=1).mean()

        return data[['month', 'value', 'rolling_avg']]

    def get_pl_table(
        self,
        entity: str,
        months: list,
        scenario: str = 'actual_cy',
        include_totals: bool = True
    ) -> pd.DataFrame:
        """
        Generate P&L table for an entity with multiple months.

        Args:
            entity: Entity name
            months: List of months to include as columns
            scenario: Scenario
            include_totals: Whether to include total rows

        Returns:
            DataFrame with accounts as rows, months as columns
        """
        if entity == 'Enterprise (All)':
            # Need to aggregate for each month
            dfs = []
            for month in months:
                month_data = self.aggregate_enterprise(month, scenario)
                month_data['month'] = month
                dfs.append(month_data)
            data = pd.concat(dfs, ignore_index=True)
        else:
            data = self.df[
                (self.df['entity'] == entity) &
                (self.df['month'].isin(months)) &
                (self.df['scenario'] == scenario)
            ]

        if not include_totals:
            data = data[~data['is_total']]

        # Pivot to wide format
        table = data.pivot_table(
            index=['account', 'category', 'is_total'],
            columns='month',
            values='value',
            aggfunc='first'
        ).reset_index()

        # Reorder columns to match month order
        month_cols = [m for m in months if m in table.columns]
        other_cols = ['account', 'category', 'is_total']
        table = table[other_cols + month_cols]

        return table


def create_sparkline_data(series: pd.Series) -> list:
    """
    Create sparkline-friendly data from a series.

    Args:
        series: Pandas Series of values

    Returns:
        List of values suitable for plotting
    """
    return series.fillna(0).tolist()
