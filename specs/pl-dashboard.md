# P&L Dashboard Specification

## Goal
Build an interactive financial dashboard that parses multi-entity P&L Excel workbooks, detects anomalies, and provides drill-down analytics for monthly performance tracking.

## Requirements

### Data Structure
**Input**: Excel workbook with multiple sheets
- **Entity tabs**: RWW, HG, DCs, Corp (each with identical P&L structure)
- **Summary sheets**: MTD Summary, YTD Summary, Naish Bridge (consolidated KPIs)
- **Structure per entity tab**:
  - Rows 0-11: Metadata
  - Row ~14: Header row containing blocks (Actual CY, Actual PY, Budget)
  - Each block: 12 monthly columns (Jan-Dec) + FY column
  - Account labels: "Sales, Parts", "TOTAL COGS", "Gross Profit", etc.
  - Totals marked with keywords: "TOTAL", "Gross", "EBITDA"

### Core Functionality

#### 1. Parsing & Standardization
- Iterate entity tabs using allowlist; skip placeholder sheets
- Detect header row by searching for tokens: `Actual \nCY`, `Jan`, etc.
- Extract column indices for scenario blocks (Actual CY, Actual PY, Budget)
- Melt into tidy format: `entity | account | scenario | year | month | value`
- Strip blank/`#` account rows
- Tag totals via regex for filtering
- Inject categorical grouping (Revenue, COGS, Opex) using hierarchy cues
- Load summary sheets as optional modules (graceful failure if missing)

#### 2. Business Logic Layer
- **Calendar helpers**:
  - Identify `latest_month` (max month with non-null Actual CY)
  - Generate trailing N-month series per entity/account
- **Anomaly detection**:
  - Calculate z-scores over trailing 12 months of Actual CY
  - Require ≥6 data points minimum
  - Flag anomalies where `|z| ≥ 2`
  - Store: delta vs trailing mean, trailing std, Actual PY value
  - Generate explanation: `"{Account} at {Entity} = {value}; {delta} vs 12-mo avg {avg}; z-score {z}; prior year same month {py_value}"`
  - Exclude totals from anomaly detection
- **Enterprise aggregation**:
  - Sum entity Actual CY for enterprise-wide views
  - Compute rolling profit metrics: GM%, EBITDA%, etc.

#### 3. Data Products
- Cache harmonized data as Parquet/CSV in `data/processed/`
- Detect workbook updates via file hash/timestamp
- Generate aggregated datasets for KPI cards

#### 4. Streamlit Dashboard

**Layout**:
- **Sidebar filters**:
  - Month selector (default: latest)
  - Entity selection (Enterprise (All) + individual entities)
  - Account search toggle
  - Anomaly severity filter

- **Top KPI cards**:
  - Revenue & Throughput
  - Gross Margin %
  - EBITDA
  - Operating Income

- **Main tabs**:
  1. **Anomaly Radar**
     - Table of flagged accounts with variance metrics
     - Sparklines for trailing 12 months
     - Drill-down buttons

  2. **Trend Explorer**
     - Stacked area/line charts for revenue and COGS
     - Heatmaps for GM% by entity
     - Bar charts for operating expense categories

  3. **Entity Drilldown**
     - Dynamic P&L table for selected entity
     - Monthly columns
     - Toggle between Actual CY and YoY delta
     - Account highlighting from anomaly selection

**Features**:
- Cross-linking: anomaly selection updates charts and highlights in P&L table
- File upload widget for workbook updates
- Streamlit caching (`st.cache_data`) for performance

#### 5. Quality & Repeatability
- Unit tests (pytest) for:
  - Header detection
  - Melting logic
  - Percentile calculations
  - Anomaly thresholds
- CLI entry point: `scripts/run_prep.py` to regenerate processed data
- Review checklist for validation

## Technical Details

### Technology Stack
- **Python 3.9+**
- **pandas** - Data manipulation
- **openpyxl** - Excel parsing
- **streamlit** - Dashboard UI
- **plotly** - Interactive charts
- **pytest** - Testing
- **pyarrow** - Parquet caching

### Project Structure
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
├── tests/
│   ├── test_parser.py
│   ├── test_business.py
│   └── fixtures/      # Test data
├── app.py             # Streamlit entry point
├── requirements.txt
└── README.md
```

### Key Algorithms

**Header Detection**:
```python
# Search for header row containing scenario blocks
for idx, row in df.iterrows():
    if 'Actual' in str(row[col]) and 'CY' in str(row[col+1]):
        header_row = idx
        break
```

**Anomaly Detection**:
```python
# Z-score calculation over trailing 12 months
rolling_mean = series.rolling(12, min_periods=6).mean()
rolling_std = series.rolling(12, min_periods=6).std()
z_score = (value - rolling_mean) / rolling_std
anomaly = abs(z_score) >= 2
```

## Success Criteria
- ✅ Correctly parses all entity tabs without manual intervention
- ✅ Detects latest month automatically
- ✅ Anomalies match expected variance patterns
- ✅ KPIs align with summary sheet totals (±1% tolerance)
- ✅ Dashboard loads in <3 seconds with cached data
- ✅ File upload successfully triggers reprocessing
- ✅ All unit tests pass
- ✅ Cross-links between anomalies and charts work seamlessly

## Open Questions
1. Should Budget scenario be included in initial implementation?
2. What alert threshold for anomaly notifications (email/Slack)?
3. Export format preferences (PDF, Excel, CSV)?
