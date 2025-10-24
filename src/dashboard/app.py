"""
Streamlit dashboard for P&L analysis.

Launch with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

# Add modules to path
sys.path.insert(0, str(Path(__file__).parent))

from modules.business import CalendarHelper, AnomalyDetector, add_month_ordering
from modules.aggregator import KPIAggregator, create_sparkline_data

# Page config
st.set_page_config(
    page_title="P&L Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Paths
DATA_DIR = Path(__file__).parent / 'data' / 'processed'


@st.cache_data
def load_data():
    """Load processed data from parquet files."""
    pl_data_path = DATA_DIR / 'pl_data.parquet'
    anomalies_path = DATA_DIR / 'anomalies.parquet'
    kpis_path = DATA_DIR / 'kpis.parquet'

    if not pl_data_path.exists():
        return None, None, None

    pl_data = pd.read_parquet(pl_data_path)
    anomalies = pd.read_parquet(anomalies_path) if anomalies_path.exists() else pd.DataFrame()
    kpis = pd.read_parquet(kpis_path) if kpis_path.exists() else pd.DataFrame()

    return pl_data, anomalies, kpis


def format_currency(value):
    """Format value as currency."""
    if pd.isna(value):
        return "N/A"
    if value >= 1_000_000:
        return f"${value/1_000_000:.1f}M"
    elif value >= 1_000:
        return f"${value/1_000:.0f}K"
    else:
        return f"${value:,.0f}"


def format_percentage(value):
    """Format value as percentage."""
    if pd.isna(value):
        return "N/A"
    return f"{value:.1f}%"


def main():
    """Main dashboard application."""

    st.title("📊 P&L Dashboard")

    # Load data
    pl_data, anomalies, kpis = load_data()

    if pl_data is None or pl_data.empty:
        st.error("No data found. Please run data preparation first:")
        st.code("python scripts/run_prep.py data/raw/your-workbook.xlsx")
        st.stop()

    # Initialize aggregator
    aggregator = KPIAggregator(pl_data)
    calendar = CalendarHelper()

    # Detect latest month
    latest_month = calendar.get_latest_month(pl_data)

    # Sidebar filters
    st.sidebar.header("Filters")

    # Month selection
    available_months = sorted(
        pl_data[pl_data['month'] != 'FY']['month'].unique(),
        key=lambda x: calendar.month_to_number(x)
    )

    # Set default to latest_month
    default_index = available_months.index(latest_month) if latest_month in available_months else len(available_months) - 1

    selected_month = st.sidebar.selectbox(
        "Month",
        options=available_months,
        index=default_index
    )

    # Entity selection
    entities = ['Enterprise (All)'] + sorted(pl_data['entity'].unique())
    selected_entity = st.sidebar.selectbox("Entity", options=entities)

    # Account search
    account_search = st.sidebar.text_input("Search Accounts", "")

    # Anomaly severity filter
    if not anomalies.empty:
        z_threshold = st.sidebar.slider(
            "Anomaly Sensitivity (Z-score)",
            min_value=1.0,
            max_value=3.0,
            value=2.0,
            step=0.1
        )
        filtered_anomalies = anomalies[anomalies['z_score'].abs() >= z_threshold]
    else:
        filtered_anomalies = pd.DataFrame()
        z_threshold = 2.0

    # Top KPI Cards
    st.header(f"Key Metrics - {selected_entity} ({selected_month})")

    entity_kpis = aggregator.calculate_kpis(selected_entity, selected_month)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        revenue = entity_kpis.get('revenue', 0)
        st.metric(
            "Revenue & Throughput",
            format_currency(revenue)
        )

    with col2:
        gm_pct = entity_kpis.get('gross_margin_pct', 0)
        st.metric(
            "Gross Margin %",
            format_percentage(gm_pct)
        )

    with col3:
        ebitda = entity_kpis.get('ebitda', 0)
        st.metric(
            "EBITDA",
            format_currency(ebitda)
        )

    with col4:
        op_income = entity_kpis.get('operating_income', 0)
        st.metric(
            "Operating Income",
            format_currency(op_income)
        )

    # Tabs
    tab1, tab2, tab3 = st.tabs(["🔍 Anomaly Radar", "📈 Trend Explorer", "🔬 Entity Drilldown"])

    # Tab 1: Anomaly Radar
    with tab1:
        st.header("Anomaly Radar")

        if filtered_anomalies.empty:
            st.info("No anomalies detected with current threshold.")
        else:
            # Filter by entity if not Enterprise
            if selected_entity != 'Enterprise (All)':
                entity_anomalies = filtered_anomalies[
                    filtered_anomalies['entity'] == selected_entity
                ]
            else:
                entity_anomalies = filtered_anomalies

            st.write(f"**{len(entity_anomalies)} anomalies detected**")

            # Display table
            display_cols = ['entity', 'account', 'month', 'value', 'z_score',
                           'delta_vs_mean', 'trailing_mean', 'py_value']

            display_df = entity_anomalies[display_cols].copy()
            display_df['value'] = display_df['value'].apply(lambda x: f"${x:,.0f}")
            display_df['z_score'] = display_df['z_score'].apply(lambda x: f"{x:.2f}")
            display_df['delta_vs_mean'] = display_df['delta_vs_mean'].apply(lambda x: f"${x:+,.0f}")
            display_df['trailing_mean'] = display_df['trailing_mean'].apply(lambda x: f"${x:,.0f}")
            display_df['py_value'] = display_df['py_value'].apply(
                lambda x: f"${x:,.0f}" if pd.notna(x) else "N/A"
            )

            st.dataframe(display_df, width='stretch')

            # Detailed explanations
            if len(entity_anomalies) > 0:
                st.subheader("Detailed Analysis")
                for idx, row in entity_anomalies.head(10).iterrows():
                    with st.expander(f"{row['account']} - {row['entity']}"):
                        if 'explanation' in row:
                            st.write(row['explanation'])
                        else:
                            st.write(f"Value: ${row['value']:,.0f}")
                            st.write(f"Z-score: {row['z_score']:.2f}")
                            st.write(f"12-month average: ${row['trailing_mean']:,.0f}")

    # Tab 2: Trend Explorer
    with tab2:
        st.header("Trend Explorer")

        # Get trailing 12 months
        trailing_months = calendar.get_trailing_months(selected_month, 12)

        # Revenue trend chart
        st.subheader("Revenue Trend")

        revenue_accounts = pl_data[
            (pl_data['category'] == 'Revenue & Throughput') &
            (pl_data['scenario'] == 'actual_cy') &
            (~pl_data['is_total'])
        ]

        if selected_entity != 'Enterprise (All)':
            revenue_accounts = revenue_accounts[revenue_accounts['entity'] == selected_entity]

        if not revenue_accounts.empty:
            revenue_pivot = revenue_accounts.pivot_table(
                index='month',
                columns='account',
                values='value',
                aggfunc='sum'
            ).reindex(trailing_months)

            fig = px.line(
                revenue_pivot,
                title="Revenue Components Over Time",
                labels={'value': 'Amount ($)', 'month': 'Month'}
            )
            st.plotly_chart(fig, width='stretch')

        # COGS trend
        st.subheader("Cost of Goods Sold Trend")

        cogs_accounts = pl_data[
            (pl_data['category'] == 'COGS') &
            (pl_data['scenario'] == 'actual_cy') &
            (~pl_data['is_total'])
        ]

        if selected_entity != 'Enterprise (All)':
            cogs_accounts = cogs_accounts[cogs_accounts['entity'] == selected_entity]

        if not cogs_accounts.empty:
            cogs_pivot = cogs_accounts.pivot_table(
                index='month',
                columns='account',
                values='value',
                aggfunc='sum'
            ).reindex(trailing_months)

            fig = px.bar(
                cogs_pivot,
                title="COGS Components Over Time",
                labels={'value': 'Amount ($)', 'month': 'Month'}
            )
            st.plotly_chart(fig, width='stretch')

        # Gross Margin Heatmap by Entity
        if selected_entity == 'Enterprise (All)':
            st.subheader("Gross Margin % Heatmap by Entity")

            gm_data = []
            for entity in pl_data['entity'].unique():
                for month in trailing_months:
                    kpis = aggregator.calculate_kpis(entity, month)
                    if 'gross_margin_pct' in kpis:
                        gm_data.append({
                            'entity': entity,
                            'month': month,
                            'gm_pct': kpis['gross_margin_pct']
                        })

            if gm_data:
                gm_df = pd.DataFrame(gm_data)
                gm_pivot = gm_df.pivot(index='entity', columns='month', values='gm_pct')
                # Only use months that actually exist in the data
                available_months = [m for m in trailing_months if m in gm_pivot.columns]
                gm_pivot = gm_pivot[available_months]

                fig = px.imshow(
                    gm_pivot,
                    labels=dict(x="Month", y="Entity", color="GM %"),
                    title="Gross Margin % by Entity and Month",
                    color_continuous_scale="RdYlGn",
                    aspect="auto"
                )
                st.plotly_chart(fig, width='stretch')

    # Tab 3: Entity Drilldown
    with tab3:
        st.header(f"Entity Drilldown - {selected_entity}")

        # Month range selector
        num_months = st.slider("Number of months to display", 1, 12, 6)
        display_months = calendar.get_trailing_months(selected_month, num_months)

        # Toggle for CY vs YoY
        view_mode = st.radio("View Mode", ["Actual CY", "Year-over-Year Delta"])

        # Get P&L table
        pl_table = aggregator.get_pl_table(
            selected_entity,
            display_months,
            scenario='actual_cy',
            include_totals=True
        )

        # Apply account search filter
        if account_search:
            pl_table = pl_table[
                pl_table['account'].str.contains(account_search, case=False, na=False)
            ]

        # Format display
        if not pl_table.empty:
            display_table = pl_table.copy()

            # Format numeric columns
            for col in display_months:
                if col in display_table.columns:
                    display_table[col] = display_table[col].apply(
                        lambda x: f"${x:,.0f}" if pd.notna(x) else ""
                    )

            # Highlight totals
            def highlight_totals(row):
                if row['is_total']:
                    return ['background-color: #f0f0f0'] * len(row)
                return [''] * len(row)

            styled_table = display_table.style.apply(highlight_totals, axis=1)

            st.dataframe(styled_table, width='stretch', height=600)
        else:
            st.info("No data available for selected filters.")

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.info(f"Latest month in data: **{latest_month}**")
    st.sidebar.info(f"Total records: **{len(pl_data):,}**")


if __name__ == '__main__':
    main()
