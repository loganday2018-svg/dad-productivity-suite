"""
Executive P&L Dashboard - Redesigned for COO

Launch with: streamlit run app_v2.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

# Add modules to path
sys.path.insert(0, str(Path(__file__).parent))

from modules.business import CalendarHelper
from modules.aggregator import KPIAggregator

# Page config - white theme
st.set_page_config(
    page_title="Executive P&L Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for white theme
st.markdown("""
<style>
    /* Force white background everywhere */
    .stApp {
        background-color: white;
    }
    .main {
        background-color: white;
    }
    section[data-testid="stSidebar"] {
        background-color: #f8f9fa;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 8px;
        border: 1px solid #e9ecef;
    }
    .delta-positive {
        color: #28a745;
    }
    .delta-negative {
        color: #dc3545;
    }
    /* White backgrounds for all containers */
    div[data-testid="stMetricValue"] {
        background-color: transparent;
    }
</style>
""", unsafe_allow_html=True)

# Paths
DATA_DIR = Path(__file__).parent / 'data' / 'processed'


@st.cache_data
def load_data():
    """Load processed data from parquet files."""
    pl_data_path = DATA_DIR / 'pl_data.parquet'

    if not pl_data_path.exists():
        return None

    pl_data = pd.read_parquet(pl_data_path)
    return pl_data


def format_currency(value):
    """Format value as currency in millions."""
    if pd.isna(value):
        return "N/A"
    return f"${value/1_000_000:.1f}M"


def format_delta(current, comparison):
    """Format delta with absolute and percentage change."""
    if pd.isna(current) or pd.isna(comparison) or comparison == 0:
        return ""

    delta = current - comparison
    pct = (delta / abs(comparison)) * 100

    sign = "+" if delta >= 0 else ""
    return f"{sign}${delta/1_000_000:.1f}M ({sign}{pct:.1f}%)"


def main():
    """Main dashboard application."""

    st.title("📊 Executive P&L Dashboard")

    # Load data
    pl_data = load_data()

    if pl_data is None or pl_data.empty:
        st.error("No data found. Please run data preparation first:")
        st.code("python scripts/run_prep.py data/raw/your-workbook.xlsx")
        st.stop()

    # Filter out excluded entities
    excluded_entities = ['DT', 'EVO', 'HG', 'Z']
    pl_data = pl_data[~pl_data['entity'].isin(excluded_entities)]

    # Initialize aggregator and calendar
    aggregator = KPIAggregator(pl_data)
    calendar = CalendarHelper()

    # Detect latest month
    latest_month = calendar.get_latest_month(pl_data)

    # === SIDEBAR CONTROLS ===
    st.sidebar.header("Controls")

    # Entity selector
    entities = ['Enterprise (All)'] + sorted(pl_data['entity'].unique())
    selected_entity = st.sidebar.selectbox("Entity", options=entities, index=0)

    # Period toggle
    period_mode = st.sidebar.radio(
        "Period",
        options=["Month-over-Month", "Year-over-Year", "Year-to-Date"],
        index=0
    )

    # Month selector
    available_months = sorted(
        pl_data[pl_data['month'] != 'FY']['month'].unique(),
        key=lambda x: calendar.month_to_number(x)
    )

    default_idx = available_months.index(latest_month) if latest_month in available_months else len(available_months) - 1
    selected_month = st.sidebar.selectbox(
        "Month",
        options=available_months,
        index=default_idx
    )

    # === CALCULATE KPIs ===
    if period_mode == "Year-to-Date":
        # Calculate YTD by summing Jan through selected month
        ytd_months = [m for m in available_months if calendar.month_to_number(m) <= calendar.month_to_number(selected_month)]

        # Sum all months for YTD
        ytd_data = pl_data[
            (pl_data['month'].isin(ytd_months)) &
            (pl_data['scenario'] == 'actual_cy')
        ]

        if selected_entity == 'Enterprise (All)':
            ytd_agg = ytd_data.groupby(['account', 'is_total'])['value'].sum().reset_index()
        else:
            ytd_agg = ytd_data[ytd_data['entity'] == selected_entity].groupby(['account', 'is_total'])['value'].sum().reset_index()

        # Extract KPIs from aggregated data
        current_kpis = {}
        for key, keywords in [
            ('revenue', ['Total Net Sales', 'Net Sales', 'Total Sales', 'Sales', 'Revenue']),
            ('cogs', ['TOTAL COGS', 'Total COGS', 'COGS']),
            ('gross_profit', ['Gross Profit']),
            ('sga', ['TOTAL DIRECT COSTS', 'Total Direct Costs']),
            ('ebitda', ['EBITDA, Incl Intercompany', 'EBITDA'])
        ]:
            for keyword in keywords:
                matches = ytd_agg[ytd_agg['account'].str.contains(keyword, case=False, na=False)]
                if len(matches) > 0:
                    current_kpis[key] = matches['value'].iloc[0]
                    break

        comparison_kpis = {}
        comparison_label = f"YTD through {selected_month}"

    elif period_mode == "Year-over-Year":
        comparison_data = aggregator.calculate_kpis_with_comparison(
            selected_entity,
            current_month=selected_month,
            comparison_month=selected_month,  # Same month
            current_scenario='actual_cy',
            comparison_scenario='actual_py'
        )
        current_kpis = comparison_data['current']
        comparison_kpis = comparison_data['comparison']
        comparison_label = f"vs PY {selected_month}"
    else:  # Month-over-Month
        # Get previous month
        current_month_idx = available_months.index(selected_month)
        if current_month_idx > 0:
            previous_month = available_months[current_month_idx - 1]
            comparison_data = aggregator.calculate_kpis_with_comparison(
                selected_entity,
                current_month=selected_month,
                comparison_month=previous_month,
                current_scenario='actual_cy',
                comparison_scenario='actual_cy'  # Previous month, same year
            )
            current_kpis = comparison_data['current']
            comparison_kpis = comparison_data['comparison']
            comparison_label = f"vs {previous_month}"
        else:
            # First month - no comparison
            current_kpis = aggregator.calculate_kpis(selected_entity, selected_month, 'actual_cy')
            comparison_kpis = {}
            comparison_label = ""

    # === HERO METRICS ===
    if period_mode == "Year-to-Date":
        st.header(f"Year-to-Date Metrics - {selected_entity} (Jan - {selected_month})")
    else:
        st.header(f"Key Metrics - {selected_entity} ({selected_month})")

    # Two rows of cards
    metrics = [
        ('Revenue', 'revenue'),
        ('COGS', 'cogs'),
        ('Gross Profit', 'gross_profit'),
        ('Direct Costs', 'sga'),
        ('EBITDA', 'ebitda')
    ]

    cols = st.columns(5)

    for idx, (label, key) in enumerate(metrics):
        with cols[idx]:
            current_val = current_kpis.get(key, 0)

            # Only show delta if not in YTD mode
            if period_mode == "Year-to-Date":
                st.metric(
                    label=label,
                    value=format_currency(current_val)
                )
            else:
                comparison_val = comparison_kpis.get(key, None) if comparison_kpis else None
                st.metric(
                    label=label,
                    value=format_currency(current_val),
                    delta=format_delta(current_val, comparison_val) if comparison_val is not None else None
                )

    # === INSIGHTS SECTION ===
    st.header("Performance Insights")

    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Performance Snapshot", "YoY Waterfall", "Trend Overview", "Entity Deep Dive", "Anomaly Radar"])

    with tab1:
        if period_mode == "Year-to-Date":
            st.subheader(f"Year-to-Date Performance (Jan - {selected_month})")

            # Single bar chart for YTD
            chart_data = []
            for label, key in metrics:
                current_val = current_kpis.get(key, 0)
                chart_data.append({'KPI': label, 'Value': current_val})

            chart_df = pd.DataFrame(chart_data)

            fig = px.bar(
                chart_df,
                x='KPI',
                y='Value',
                title=f"Year-to-Date Metrics through {selected_month}",
                labels={'Value': 'Amount ($)'}
            )

            fig.update_layout(
                plot_bgcolor='white',
                paper_bgcolor='white',
                showlegend=False
            )

            fig.update_traces(marker_color='#0066cc')

            st.plotly_chart(fig, width='stretch')

        else:
            st.subheader("Current vs Comparison Period")

            if comparison_kpis:
                # Paired column chart
                chart_data = []

                if period_mode == "Year-over-Year":
                    current_label = 'Current Year'
                    comparison_period_label = 'Prior Year'
                    chart_title = f"{selected_month} - Current Year vs Prior Year"
                else:
                    current_label = selected_month
                    comparison_period_label = comparison_label.replace('vs ', '')
                    chart_title = f"{selected_month} vs {comparison_period_label}"

                for label, key in metrics:
                    current_val = current_kpis.get(key, 0)
                    comp_val = comparison_kpis.get(key, 0)

                    chart_data.append({'KPI': label, 'Period': current_label, 'Value': current_val})
                    chart_data.append({'KPI': label, 'Period': comparison_period_label, 'Value': comp_val})

                chart_df = pd.DataFrame(chart_data)

                fig = px.bar(
                    chart_df,
                    x='KPI',
                    y='Value',
                    color='Period',
                    barmode='group',
                    title=chart_title,
                    labels={'Value': 'Amount ($)'}
                )

                fig.update_layout(
                    plot_bgcolor='white',
                    paper_bgcolor='white'
                )

                st.plotly_chart(fig, width='stretch')
            else:
                st.info("No comparison period available.")

    with tab2:
        st.subheader("Year-over-Year Bridge Analysis")

        if period_mode == "Year-over-Year" and comparison_kpis:
            # Build waterfall for Revenue to EBITDA
            py_revenue = comparison_kpis.get('revenue', 0)
            cy_revenue = current_kpis.get('revenue', 0)
            py_cogs = comparison_kpis.get('cogs', 0)
            cy_cogs = current_kpis.get('cogs', 0)
            py_sga = comparison_kpis.get('sga', 0)
            cy_sga = current_kpis.get('sga', 0)
            cy_ebitda = current_kpis.get('ebitda', 0)

            # Calculate deltas
            revenue_delta = cy_revenue - py_revenue
            cogs_delta = cy_cogs - py_cogs
            sga_delta = cy_sga - py_sga

            # Waterfall data
            waterfall_data = {
                'x': [
                    f'PY {selected_month}\nRevenue',
                    'Revenue\nChange',
                    'COGS\nChange',
                    'Direct Costs\nChange',
                    f'CY {selected_month}\nEBITDA'
                ],
                'measure': ['absolute', 'relative', 'relative', 'relative', 'total'],
                'y': [
                    py_revenue,
                    revenue_delta,
                    -cogs_delta,  # Negative because COGS reduces profit
                    -sga_delta,   # Negative because costs reduce profit
                    cy_ebitda
                ],
                'text': [
                    format_currency(py_revenue),
                    f"{'+' if revenue_delta >= 0 else ''}{format_currency(revenue_delta)}",
                    f"{'+' if -cogs_delta >= 0 else ''}{format_currency(-cogs_delta)}",
                    f"{'+' if -sga_delta >= 0 else ''}{format_currency(-sga_delta)}",
                    format_currency(cy_ebitda)
                ]
            }

            fig = go.Figure(go.Waterfall(
                x=waterfall_data['x'],
                y=waterfall_data['y'],
                measure=waterfall_data['measure'],
                text=waterfall_data['text'],
                textposition='outside',
                connector={'line': {'color': 'rgb(63, 63, 63)'}},
                increasing={'marker': {'color': '#28a745'}},
                decreasing={'marker': {'color': '#dc3545'}},
                totals={'marker': {'color': '#0066cc'}}
            ))

            fig.update_layout(
                title=f"Year-over-Year Bridge: {selected_month} PY → CY",
                showlegend=False,
                plot_bgcolor='white',
                paper_bgcolor='white',
                height=500
            )

            st.plotly_chart(fig, width='stretch')

            # Summary text
            ebitda_change = cy_ebitda - comparison_kpis.get('ebitda', 0)
            st.markdown(f"""
            **Summary:** EBITDA changed by **{format_currency(ebitda_change)}**
            ({'+' if ebitda_change >= 0 else ''}{(ebitda_change / abs(comparison_kpis.get('ebitda', 1)) * 100):.1f}%)
            from prior year.
            """)
        else:
            st.info("Switch to 'Year-over-Year' mode to see the waterfall analysis.")

    with tab3:
        st.subheader("Trend Overview - Last 6 Months")

        # Get trailing 6 months
        trailing_months = calendar.get_trailing_months(selected_month, 6)

        # Allow KPI selection
        selected_kpi = st.radio("Select KPI", ['Revenue', 'Gross Profit', 'EBITDA'], horizontal=True)

        kpi_map = {
            'Revenue': 'revenue',
            'Gross Profit': 'gross_profit',
            'EBITDA': 'ebitda'
        }

        kpi_key = kpi_map[selected_kpi]

        # Build trend data
        trend_data = []
        for month in trailing_months:
            cy_kpis = aggregator.calculate_kpis(selected_entity, month, 'actual_cy')
            py_kpis = aggregator.calculate_kpis(selected_entity, month, 'actual_py')

            if kpi_key in cy_kpis:
                trend_data.append({'Month': month, 'Period': 'Current Year', 'Value': cy_kpis[kpi_key]})

            if kpi_key in py_kpis:
                trend_data.append({'Month': month, 'Period': 'Prior Year', 'Value': py_kpis[kpi_key]})

        if trend_data:
            trend_df = pd.DataFrame(trend_data)

            fig = px.line(
                trend_df,
                x='Month',
                y='Value',
                color='Period',
                title=f"{selected_kpi} Trend - 6 Month View",
                markers=True,
                labels={'Value': 'Amount ($)'}
            )

            fig.update_layout(
                plot_bgcolor='white',
                paper_bgcolor='white'
            )

            st.plotly_chart(fig, width='stretch')
        else:
            st.warning("No trend data available.")

    with tab4:
        st.subheader("Entity Subtotal Analysis")

        # Get list of entities (excluding Enterprise)
        entity_list = [e for e in entities if e != 'Enterprise (All)']

        if selected_entity == 'Enterprise (All)':
            st.info("Select a specific entity from the sidebar to see detailed subtotal analysis.")

            # Show comparison table across all entities
            st.markdown("### Subtotal Comparison Across Entities")

            # Get all "Subtotal X" rows
            all_subtotal_rows = pl_data[
                (pl_data['account'].str.startswith('Subtotal', na=False))
            ]['account'].unique()

            comparison_data = []
            for entity in entity_list:
                entity_data = pl_data[
                    (pl_data['entity'] == entity) &
                    (pl_data['month'] == selected_month) &
                    (pl_data['scenario'] == 'actual_cy') &
                    (pl_data['account'].str.startswith('Subtotal', na=False))
                ]

                row = {'Entity': entity}
                for subtotal in all_subtotal_rows:
                    matches = entity_data[entity_data['account'] == subtotal]
                    if len(matches) > 0:
                        row[subtotal] = matches['value'].iloc[0]
                    else:
                        row[subtotal] = 0

                comparison_data.append(row)

            if comparison_data:
                comp_df = pd.DataFrame(comparison_data)

                # Create numeric df for heatmap before formatting
                numeric_df = comp_df.copy()
                numeric_cols = [col for col in all_subtotal_rows if col in numeric_df.columns]

                # Apply subdued heatmap styling
                def highlight_values(data):
                    # Normalize each column separately
                    styled = pd.DataFrame('', index=data.index, columns=data.columns)
                    for col in numeric_cols:
                        if col in data.columns:
                            col_data = pd.to_numeric(data[col], errors='coerce')
                            if col_data.notna().sum() > 0:
                                # Normalize to 0-1 range
                                col_min = col_data.min()
                                col_max = col_data.max()
                                if col_max != col_min:
                                    normalized = (col_data - col_min) / (col_max - col_min)
                                    # Apply subdued green color scale (light green to slightly darker green)
                                    for idx in data.index:
                                        if pd.notna(normalized[idx]):
                                            intensity = int(220 - (normalized[idx] * 60))  # Range from 220 to 160
                                            styled.loc[idx, col] = f'background-color: rgb({intensity}, 240, {intensity})'
                    return styled

                # Format currency after styling logic is set up
                display_df = comp_df.copy()
                for col in all_subtotal_rows:
                    if col in display_df.columns:
                        display_df[col] = display_df[col].apply(lambda x: format_currency(x) if pd.notna(x) else "N/A")

                # Apply styling
                styled_df = display_df.style.apply(lambda x: highlight_values(numeric_df), axis=None)

                st.dataframe(styled_df, width='stretch', hide_index=True, use_container_width=True)

        else:
            # Show detailed subtotals for selected entity
            st.markdown(f"### Subtotal Details for {selected_entity}")

            entity_subtotals = pl_data[
                (pl_data['entity'] == selected_entity) &
                (pl_data['month'] == selected_month) &
                (pl_data['scenario'] == 'actual_cy') &
                (pl_data['account'].str.startswith('Subtotal', na=False))
            ].sort_values('account')

            if not entity_subtotals.empty:
                # Create a clean display table
                display_df = entity_subtotals[['account', 'value']].copy()
                display_df.columns = ['Subtotal', 'Value']

                # Calculate heatmap colors before formatting
                numeric_values = display_df['Value'].copy()
                val_min = numeric_values.min()
                val_max = numeric_values.max()

                # Create color map
                color_map = {}
                if val_max != val_min:
                    for idx, val in numeric_values.items():
                        normalized = (val - val_min) / (val_max - val_min)
                        intensity = int(220 - (normalized * 60))
                        color_map[idx] = f'background-color: rgb({intensity}, 240, {intensity})'
                else:
                    for idx in numeric_values.index:
                        color_map[idx] = ''

                # Format values as currency
                display_df['Value'] = display_df['Value'].apply(format_currency)

                # Apply styling using the color map
                def apply_colors(row):
                    return ['', color_map.get(row.name, '')]

                styled_df = display_df.style.apply(apply_colors, axis=1)

                st.dataframe(styled_df, width='stretch', hide_index=True, use_container_width=True)

                # Add comparison if in YoY mode
                if period_mode == "Year-over-Year":
                    st.markdown("### Year-over-Year Comparison")

                    py_subtotals = pl_data[
                        (pl_data['entity'] == selected_entity) &
                        (pl_data['month'] == selected_month) &
                        (pl_data['scenario'] == 'actual_py') &
                        (pl_data['account'].str.startswith('Subtotal', na=False))
                    ]

                    # Merge CY and PY
                    comparison = entity_subtotals.merge(
                        py_subtotals[['account', 'value']],
                        on='account',
                        how='left',
                        suffixes=('_cy', '_py')
                    )

                    comparison['Delta'] = comparison['value_cy'] - comparison['value_py']
                    comparison['Delta %'] = (comparison['Delta'] / comparison['value_py'].abs()) * 100

                    # Format for display
                    comp_display = comparison[['account', 'value_cy', 'value_py', 'Delta', 'Delta %']].copy()
                    comp_display.columns = ['Subtotal', 'Current Year', 'Prior Year', 'Change ($)', 'Change (%)']

                    comp_display['Current Year'] = comp_display['Current Year'].apply(format_currency)
                    comp_display['Prior Year'] = comp_display['Prior Year'].apply(format_currency)
                    comp_display['Change ($)'] = comp_display['Change ($)'].apply(format_currency)
                    comp_display['Change (%)'] = comp_display['Change (%)'].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "N/A")

                    st.dataframe(comp_display, width='stretch', hide_index=True)
            else:
                st.warning(f"No subtotal data found for {selected_entity} in {selected_month}.")

    with tab5:
        st.info("Anomaly detection module coming soon.")

    # === FOOTER ===
    st.sidebar.markdown("---")
    st.sidebar.info(f"**Latest month:** {latest_month}")
    st.sidebar.info(f"**Total records:** {len(pl_data):,}")
    st.sidebar.info(f"**Entities:** {len(pl_data['entity'].unique())}")


if __name__ == '__main__':
    main()
