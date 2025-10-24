"""
Unit tests for parser module.
"""

import pytest
import pandas as pd
from modules.parser import PLParser


class TestPLParser:
    """Tests for PLParser class."""

    def test_detect_header_row(self):
        """Test header row detection."""
        # Create mock DataFrame with header at row 13
        rows = []
        for i in range(15):
            if i < 13:
                rows.append(['', '', ''])
            elif i == 13:
                rows.append(['', 'Actual', 'CY'])
            else:
                rows.append(['Account', 'Sales', 'Parts'])

        data = pd.DataFrame(rows)

        parser = PLParser('dummy.xlsx')
        header_idx = parser.detect_header_row(data)

        assert header_idx == 13  # Row with 'Actual'

    def test_is_total_row(self):
        """Test total row identification."""
        parser = PLParser('dummy.xlsx')

        assert parser.is_total_row('TOTAL Revenue')
        assert parser.is_total_row('Gross Profit')
        assert parser.is_total_row('EBITDA')
        assert not parser.is_total_row('Sales, Parts')

    def test_categorize_account(self):
        """Test account categorization."""
        parser = PLParser('dummy.xlsx')

        assert parser.categorize_account('Sales, Parts') == 'Revenue & Throughput'
        assert parser.categorize_account('COGS - Materials') == 'COGS'
        assert parser.categorize_account('Operating Expense') == 'Operating Expenses'
        assert parser.categorize_account('Other Item') == 'Other'

    def test_find_account_column(self):
        """Test account column detection."""
        data = pd.DataFrame({
            0: ['', '', '', 'Account Name', 'Sales', 'COGS', 'Opex'],
            1: ['', '', '', 'Actual', 100, 50, 30],
            2: ['', '', '', 'CY', 110, 55, 32]
        })

        parser = PLParser('dummy.xlsx')
        account_col = parser.find_account_column(data, header_idx=3)

        assert account_col == 0

    def test_extract_scenario_columns(self):
        """Test scenario column extraction."""
        header_row = pd.Series([
            '', 'Actual', 'CY', 'Jan', 'Feb', 'Mar', 'FY',
            'Actual', 'PY', 'Jan', 'Feb', 'Mar', 'FY'
        ])

        parser = PLParser('dummy.xlsx')
        scenarios = parser.extract_scenario_columns(header_row)

        assert 'actual_cy' in scenarios
        assert 'actual_py' in scenarios
        assert len(scenarios['actual_cy']) == 4  # Jan, Feb, Mar, FY
        assert len(scenarios['actual_py']) == 4
