# P&L Dashboard

Interactive financial dashboard for multi-entity P&L analysis with anomaly detection.

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Data

Place your Excel workbook in `data/raw/` directory.

Run the data preparation script:

```bash
python scripts/run_prep.py data/raw/your-workbook.xlsx
```

This will:
- Parse all entity tabs
- Detect anomalies
- Generate processed datasets in `data/processed/`

### 3. Launch Dashboard

```bash
streamlit run app.py
```

The dashboard will open in your browser at `http://localhost:8501`

## Dashboard Features

### Sidebar Filters
- **Month Selection**: Choose reporting period (defaults to latest available)
- **Entity Selection**: View individual entities or enterprise-wide aggregation
- **Account Search**: Filter to specific accounts
- **Anomaly Severity**: Adjust sensitivity threshold

### Top KPI Cards
- Revenue & Throughput
- Gross Margin %
- EBITDA
- Operating Income

### Tabs

#### 1. Anomaly Radar
Identifies accounts with unusual variances based on trailing 12-month patterns:
- Variance metrics and z-scores
- Sparklines showing historical trends
- Drill-down to detailed analysis

#### 2. Trend Explorer
Visual analytics across time periods:
- Revenue and COGS trends
- Gross margin heatmaps by entity
- Operating expense breakdowns

#### 3. Entity Drilldown
Detailed P&L table for selected entity:
- Monthly performance columns
- Year-over-year comparisons
- Linked highlighting from anomaly selection

## Refreshing Data

When you receive a new monthly workbook:

1. Place the file in `data/raw/`
2. Run: `python scripts/run_prep.py data/raw/new-workbook.xlsx`
3. Reload the Streamlit dashboard

The system will auto-detect the latest month and update all calculations.

## Testing

Run the test suite:

```bash
pytest tests/ -v --cov=modules
```

## Project Structure

```
src/dashboard/
├── data/
│   ├── raw/           # Input Excel files
│   └── processed/     # Cached Parquet/CSV
├── scripts/
│   └── run_prep.py    # CLI data preparation
├── modules/
│   ├── parser.py      # Excel parsing logic
│   ├── business.py    # Calendar & anomaly detection
│   └── aggregator.py  # KPI calculations
├── tests/             # Unit tests
├── app.py             # Streamlit entry point
└── requirements.txt   # Dependencies
```

## Validation Checklist

After processing new data, verify:
- ✅ Latest month auto-detected correctly
- ✅ All entity tabs parsed (check logs for errors)
- ✅ KPI cards match summary sheet totals
- ✅ Anomalies align with known variances
- ✅ Charts render without errors

## Troubleshooting

**Dashboard won't start:**
- Ensure all dependencies installed: `pip install -r requirements.txt`
- Check Python version ≥ 3.9

**Data parsing errors:**
- Verify workbook structure matches expected format
- Check entity tab names against allowlist in `parser.py`
- Review logs in console for specific row/column issues

**Anomalies seem incorrect:**
- Adjust z-score threshold in sidebar
- Verify at least 6 months of historical data exist
- Check for outliers in raw data

## Support

For issues or questions, see the project repository.
