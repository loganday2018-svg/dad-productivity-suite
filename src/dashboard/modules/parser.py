"""
Excel P&L parser for multi-entity workbooks.

Handles detection of header rows, scenario blocks (Actual CY/PY, Budget),
and conversion to tidy data format.
"""

import pandas as pd
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Entity tab allowlist
ENTITY_TABS = ['RWW', 'HG', 'DCs', 'Corp']

# Month names
MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'FY']

# Keywords for identifying totals
TOTAL_KEYWORDS = ['TOTAL', 'Gross', 'EBITDA', 'Operating Income', 'Net Income']

# Account category mapping
CATEGORY_MAP = {
    'Revenue & Throughput': ['Sales', 'Revenue', 'Throughput'],
    'COGS': ['COGS', 'Cost of Goods'],
    'Operating Expenses': ['Opex', 'Operating Expense', 'SG&A', 'Selling', 'General', 'Administrative'],
}


class PLParser:
    """Parser for P&L Excel workbooks."""

    def __init__(self, file_path: str):
        """
        Initialize parser.

        Args:
            file_path: Path to Excel workbook
        """
        self.file_path = Path(file_path)
        self.workbook = None
        self.sheet_names = []

    def load_workbook(self) -> None:
        """Load Excel workbook and get sheet names."""
        logger.info(f"Loading workbook: {self.file_path}")
        self.workbook = pd.ExcelFile(self.file_path, engine='openpyxl')
        self.sheet_names = self.workbook.sheet_names
        logger.info(f"Found {len(self.sheet_names)} sheets")

    def detect_header_row(self, df: pd.DataFrame) -> Optional[int]:
        """
        Detect header row containing scenario blocks.

        Searches for patterns like 'Actual' followed by 'CY' or month names.

        Args:
            df: DataFrame containing sheet data

        Returns:
            Row index of header, or None if not found
        """
        for idx in range(min(20, len(df))):  # Search first 20 rows
            row_str = ' '.join([str(val) for val in df.iloc[idx].values if pd.notna(val)])
            # Look for Actual CY pattern or month names
            if ('Actual' in row_str and 'CY' in row_str) or sum(month in row_str for month in MONTHS[:3]) >= 2:
                logger.info(f"Detected header row at index {idx}")
                return idx
        logger.warning("Could not detect header row")
        return None

    def extract_scenario_columns(self, header_row: pd.Series) -> Dict[str, List[int]]:
        """
        Extract column indices for each scenario block.

        For files with multiple years, returns only the LATEST year for CY
        and the SECOND LATEST for PY.

        Args:
            header_row: The header row as a Series

        Returns:
            Dict mapping scenario name to list of column indices
        """
        # First, find all year blocks
        year_blocks = {}  # year -> list of column indices

        for idx, val in enumerate(header_row):
            val_str = str(val).strip()

            # Check if this is a month column
            for month in MONTHS:
                if val_str.startswith(month):
                    # Extract year (e.g., "Jan 22" -> "22", "FY24" -> "24")
                    year_part = val_str.replace(month, '').strip()
                    if year_part:
                        if year_part not in year_blocks:
                            year_blocks[year_part] = []
                        year_blocks[year_part].append(idx)
                    break

        # Sort years to find latest
        sorted_years = sorted(year_blocks.keys())

        scenarios = {}
        if len(sorted_years) >= 1:
            # Latest year = Actual CY
            latest_year = sorted_years[-1]
            scenarios['actual_cy'] = year_blocks[latest_year]
            logger.info(f"Using year {latest_year} as Actual CY")

        if len(sorted_years) >= 2:
            # Second latest = Actual PY
            py_year = sorted_years[-2]
            scenarios['actual_py'] = year_blocks[py_year]
            logger.info(f"Using year {py_year} as Actual PY")

        logger.info(f"Extracted scenarios: {list(scenarios.keys())}")
        return scenarios

    def find_account_column(self, df: pd.DataFrame, header_idx: int) -> Optional[int]:
        """
        Find the column containing account labels.

        Args:
            df: DataFrame containing sheet data
            header_idx: Index of header row

        Returns:
            Column index containing account names
        """
        # Look for column with text values below header
        for col_idx in range(min(10, df.shape[1])):  # Check first 10 columns
            # Count non-null text values in next 10 rows
            sample = df.iloc[header_idx + 1:header_idx + 11, col_idx]
            text_count = sample.apply(lambda x: isinstance(x, str) and len(x.strip()) > 0).sum()

            if text_count >= 5:  # At least 5 text values
                logger.info(f"Account column detected at index {col_idx}")
                return col_idx

        logger.warning("Could not detect account column, using column 0")
        return 0

    def categorize_account(self, account: str) -> str:
        """
        Assign category to account based on name.

        Args:
            account: Account name

        Returns:
            Category name
        """
        account_lower = account.lower()

        for category, keywords in CATEGORY_MAP.items():
            if any(kw.lower() in account_lower for kw in keywords):
                return category

        return 'Other'

    def is_total_row(self, account: str) -> bool:
        """
        Check if account represents a total/subtotal.

        Args:
            account: Account name

        Returns:
            True if total row
        """
        return any(kw in account for kw in TOTAL_KEYWORDS)

    def parse_entity_tab(self, sheet_name: str) -> pd.DataFrame:
        """
        Parse a single entity tab into tidy format.

        Args:
            sheet_name: Name of sheet to parse

        Returns:
            Tidy DataFrame with columns: entity, account, scenario, month, value, is_total, category
        """
        logger.info(f"Parsing entity tab: {sheet_name}")

        # Read sheet
        df = pd.read_excel(self.workbook, sheet_name=sheet_name, header=None)

        # Detect header row
        header_idx = self.detect_header_row(df)
        if header_idx is None:
            logger.error(f"Could not parse {sheet_name}: header not found")
            return pd.DataFrame()

        # Extract scenarios and columns
        header_row = df.iloc[header_idx]
        scenarios = self.extract_scenario_columns(header_row)
        account_col = self.find_account_column(df, header_idx)

        if not scenarios:
            logger.error(f"Could not parse {sheet_name}: no scenarios found")
            return pd.DataFrame()

        # Build tidy data
        records = []

        # Process rows below header
        for row_idx in range(header_idx + 1, len(df)):
            account = df.iloc[row_idx, account_col]

            # Skip blank or # rows
            if pd.isna(account) or str(account).strip() in ['', '#']:
                continue

            account = str(account).strip()
            is_total = self.is_total_row(account)
            category = self.categorize_account(account)

            # Extract values for each scenario
            for scenario_name, col_indices in scenarios.items():
                for col_idx in col_indices:
                    # Get month name from header (e.g., "Jan 22" -> "Jan")
                    month_raw = str(header_row.iloc[col_idx]).strip()

                    # Extract just the month part
                    month = None
                    for m in MONTHS:
                        if month_raw.startswith(m):
                            month = m
                            break

                    if month is None:
                        continue

                    value = df.iloc[row_idx, col_idx]

                    # Skip non-numeric values
                    if pd.isna(value):
                        continue

                    try:
                        value = float(value)
                    except (ValueError, TypeError):
                        continue

                    records.append({
                        'entity': sheet_name,
                        'account': account,
                        'scenario': scenario_name,
                        'month': month,
                        'value': value,
                        'is_total': is_total,
                        'category': category
                    })

        result = pd.DataFrame(records)
        logger.info(f"Parsed {len(result)} records from {sheet_name}")
        return result

    def parse_all_entities(self) -> pd.DataFrame:
        """
        Parse all entity tabs and combine into single DataFrame.

        Returns:
            Combined tidy DataFrame
        """
        all_data = []

        for sheet in self.sheet_names:
            # Only parse entity tabs
            if sheet in ENTITY_TABS:
                entity_data = self.parse_entity_tab(sheet)
                if not entity_data.empty:
                    all_data.append(entity_data)

        if not all_data:
            logger.error("No entity data parsed")
            return pd.DataFrame()

        combined = pd.concat(all_data, ignore_index=True)
        logger.info(f"Combined data: {len(combined)} total records from {len(all_data)} entities")
        return combined

    def parse_summary_sheet(self, sheet_name: str) -> Optional[pd.DataFrame]:
        """
        Parse summary sheet for KPI rollups.

        Args:
            sheet_name: Name of summary sheet

        Returns:
            DataFrame with KPI data or None if parsing fails
        """
        try:
            logger.info(f"Parsing summary sheet: {sheet_name}")
            df = pd.read_excel(self.workbook, sheet_name=sheet_name, header=None)

            # Use same header detection logic
            header_idx = self.detect_header_row(df)
            if header_idx is None:
                return None

            # For now, return simplified parsing
            # Full implementation would extract specific KPIs
            return df

        except Exception as e:
            logger.warning(f"Could not parse summary sheet {sheet_name}: {e}")
            return None


def load_and_parse(file_path: str) -> pd.DataFrame:
    """
    Convenience function to load and parse workbook.

    Args:
        file_path: Path to Excel file

    Returns:
        Tidy DataFrame of all entity P&L data
    """
    parser = PLParser(file_path)
    parser.load_workbook()
    return parser.parse_all_entities()
