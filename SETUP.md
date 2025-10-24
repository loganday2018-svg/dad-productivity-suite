# Setup Guide

## Quick Start

### 1. Install Python Dependencies

Navigate to the dashboard directory:
```bash
cd src/dashboard
pip install -r requirements.txt
```

### 2. Prepare Your Data

Place your P&L Excel workbook in `src/dashboard/data/raw/`

Run the data preparation script:
```bash
python scripts/run_prep.py data/raw/your-workbook.xlsx
```

### 3. Launch the Dashboard

```bash
streamlit run app.py
```

The dashboard will open at `http://localhost:8501`

## Project Structure

```
dad-productivity-suite/
├── specs/                  # Codex-generated specifications
│   └── pl-dashboard.md    # P&L dashboard spec
├── src/
│   └── dashboard/         # P&L Dashboard implementation
│       ├── data/
│       │   ├── raw/       # Place Excel files here
│       │   └── processed/ # Auto-generated cached data
│       ├── modules/
│       │   ├── parser.py      # Excel parsing
│       │   ├── business.py    # Anomaly detection
│       │   └── aggregator.py  # KPI calculations
│       ├── scripts/
│       │   └── run_prep.py    # Data preparation CLI
│       ├── tests/         # Unit tests
│       ├── app.py         # Streamlit dashboard
│       └── requirements.txt
└── reviews/               # Code review feedback (from Codex)
```

## Workflow

1. **Planning** (Codex in Cursor): Create specs in `/specs`
2. **Implementation** (Claude Code): Build features in `/src`
3. **Review** (Codex): Add feedback to `/reviews`

## Running Tests

```bash
cd src/dashboard
pytest tests/ -v
```

## Troubleshooting

**Import errors:**
- Ensure you're in the `src/dashboard` directory
- Verify all dependencies installed: `pip install -r requirements.txt`

**No data in dashboard:**
- Run `python scripts/run_prep.py data/raw/your-file.xlsx` first
- Check console output for parsing errors

**Excel parsing fails:**
- Verify workbook has expected entity tabs (RWW, HG, DCs, Corp)
- Check that header rows contain "Actual CY" and month names
- Review parser logs for specific errors

## Next Steps

- Add more entity tabs to `ENTITY_TABS` in `modules/parser.py` if needed
- Adjust anomaly z-score threshold in dashboard sidebar
- Customize category mappings in `modules/parser.py`
