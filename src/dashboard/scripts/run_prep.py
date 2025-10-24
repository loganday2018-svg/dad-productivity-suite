"""
CLI script to prepare P&L data from Excel workbook.

Usage:
    python run_prep.py path/to/workbook.xlsx
"""

import sys
import argparse
from pathlib import Path
import logging
import hashlib
import json

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.parser import load_and_parse
from modules.business import AnomalyDetector, CalendarHelper, add_month_ordering
from modules.aggregator import KPIAggregator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_file_hash(file_path: Path) -> str:
    """Calculate hash of file for change detection."""
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        hasher.update(f.read())
    return hasher.hexdigest()


def save_metadata(output_dir: Path, file_hash: str, latest_month: str):
    """Save processing metadata."""
    metadata = {
        'file_hash': file_hash,
        'latest_month': latest_month,
        'processed_at': str(Path(__file__).parent.parent)
    }

    metadata_path = output_dir / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Saved metadata to {metadata_path}")


def main():
    """Main processing pipeline."""
    parser = argparse.ArgumentParser(description='Prepare P&L data for dashboard')
    parser.add_argument('input_file', help='Path to Excel workbook')
    parser.add_argument(
        '--output-dir',
        default='data/processed',
        help='Output directory for processed data'
    )
    parser.add_argument(
        '--z-threshold',
        type=float,
        default=2.0,
        help='Z-score threshold for anomaly detection'
    )

    args = parser.parse_args()

    input_path = Path(args.input_file)
    output_dir = Path(__file__).parent.parent / args.output_dir

    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Calculate file hash
    file_hash = get_file_hash(input_path)
    logger.info(f"Processing file: {input_path}")
    logger.info(f"File hash: {file_hash}")

    # Check if already processed
    metadata_path = output_dir / 'metadata.json'
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        if metadata.get('file_hash') == file_hash:
            logger.info("File unchanged, skipping processing")
            print(f"\n✓ Data already up to date (latest month: {metadata.get('latest_month')})")
            return

    logger.info("=" * 60)
    logger.info("STEP 1: Parsing Excel workbook")
    logger.info("=" * 60)

    # Parse workbook
    df = load_and_parse(str(input_path))

    if df.empty:
        logger.error("No data parsed from workbook")
        sys.exit(1)

    logger.info(f"Parsed {len(df)} records")

    # Add month ordering
    df = add_month_ordering(df)

    # Save raw parsed data
    raw_output = output_dir / 'pl_data.parquet'
    df.to_parquet(raw_output, index=False)
    logger.info(f"Saved parsed data to {raw_output}")

    logger.info("=" * 60)
    logger.info("STEP 2: Detecting latest month and anomalies")
    logger.info("=" * 60)

    # Detect latest month
    calendar = CalendarHelper()
    latest_month = calendar.get_latest_month(df)
    logger.info(f"Latest month: {latest_month}")

    # Detect anomalies
    detector = AnomalyDetector(z_threshold=args.z_threshold)
    anomalies = detector.detect_anomalies(df, exclude_totals=True)

    # Generate explanations
    if not anomalies.empty:
        anomalies['explanation'] = anomalies.apply(
            detector.generate_explanation,
            axis=1
        )

    # Save anomalies
    anomaly_output = output_dir / 'anomalies.parquet'
    anomalies.to_parquet(anomaly_output, index=False)
    logger.info(f"Saved {len(anomalies)} anomalies to {anomaly_output}")

    logger.info("=" * 60)
    logger.info("STEP 3: Calculating KPIs")
    logger.info("=" * 60)

    # Calculate enterprise KPIs
    aggregator = KPIAggregator(df)

    entities = df['entity'].unique().tolist() + ['Enterprise (All)']
    kpi_records = []

    for entity in entities:
        kpis = aggregator.calculate_kpis(entity, latest_month)
        if kpis:
            kpis['entity'] = entity
            kpis['month'] = latest_month
            kpi_records.append(kpis)
            logger.info(f"  {entity}: {len(kpis)} KPIs calculated")

    if kpi_records:
        import pandas as pd
        kpi_df = pd.DataFrame(kpi_records)
        kpi_output = output_dir / 'kpis.parquet'
        kpi_df.to_parquet(kpi_output, index=False)
        logger.info(f"Saved KPIs to {kpi_output}")

    # Save metadata
    save_metadata(output_dir, file_hash, latest_month)

    logger.info("=" * 60)
    logger.info("PROCESSING COMPLETE")
    logger.info("=" * 60)

    print("\n" + "=" * 60)
    print("✓ Data processing complete!")
    print("=" * 60)
    print(f"  Latest month:    {latest_month}")
    print(f"  Total records:   {len(df):,}")
    print(f"  Anomalies found: {len(anomalies)}")
    print(f"  Entities:        {len(df['entity'].unique())}")
    print("=" * 60)
    print(f"\nProcessed data saved to: {output_dir}")
    print("\nReady to launch dashboard:")
    print("  streamlit run app.py")
    print("=" * 60)


if __name__ == '__main__':
    main()
