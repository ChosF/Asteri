
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from millify import millify
import requests
from io import BytesIO
import sys

def inject_custom_css():
    """Inject modern CSS styling with dark mode support"""
    st.markdown("""
    <style>
    /* Modern CSS Variables for Theme Support */
    :root {
        --primary-color: #3b82f6;
        --secondary-color: #1e293b;
        --accent-color: #06b6d4;
        --success-color: #10b981;
        --warning-color: #f59e0b;
        --error-color: #ef4444;
        --text-primary: #1f2937;
        --text-secondary: #6b7280;
        --bg-primary: #ffffff;
        --bg-secondary: #f8fafc;
        --border-color: #e5e7eb;
        --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.05);
        --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
        --shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1);
        --shadow-xl: 0 20px 25px -5px rgb(0 0 0 / 0.1), 0 8px 10px -6px rgb(0 0 0 / 0.1);
        --border-radius: 0.5rem;
        --transition: all 0.3s ease;
    }

    /* Dark mode variables */
    .dark-mode {
        --text-primary: #f9fafb;
        --text-secondary: #d1d5db;
        --bg-primary: #111827;
        --bg-secondary: #1f2937;
        --border-color: #374151;
    }

    /* Global Styles */
    * {
        box-sizing: border-box;
    }

    body {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        background-color: var(--bg-secondary);
        color: var(--text-primary);
        line-height: 1.6;
    }

    /* Modern Header */
    .modern-header {
        background: linear-gradient(135deg, var(--primary-color), var(--accent-color));
        color: white;
        padding: 2rem;
        border-radius: var(--border-radius);
        margin-bottom: 2rem;
        box-shadow: var(--shadow-lg);
        position: relative;
        overflow: hidden;
    }

    .modern-header::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: shimmer 3s infinite;
    }

    @keyframes shimmer {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }

    /* Modern Cards */
    .modern-card {
        background: var(--bg-primary);
        border-radius: var(--border-radius);
        padding: 1.5rem;
        box-shadow: var(--shadow-md);
        transition: var(--transition);
        border: 1px solid var(--border-color);
        position: relative;
        overflow: hidden;
    }

    .modern-card:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-xl);
    }

    .modern-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, var(--primary-color), var(--accent-color));
        transform: scaleX(0);
        transition: var(--transition);
    }

    .modern-card:hover::before {
        transform: scaleX(1);
    }

    /* Modern Metrics */
    .metric-container {
        display: flex;
        flex-direction: column;
        gap: 0.5rem;
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: var(--text-primary);
        line-height: 1;
    }

    .metric-label {
        font-size: 0.875rem;
        color: var(--text-secondary);
        font-weight: 500;
    }

    .metric-delta {
        font-size: 0.75rem;
        padding: 0.25rem 0.5rem;
        border-radius: 9999px;
        font-weight: 600;
    }

    .metric-delta.positive {
        background-color: rgba(16, 185, 129, 0.1);
        color: var(--success-color);
    }

    .metric-delta.negative {
        background-color: rgba(239, 68, 68, 0.1);
        color: var(--error-color);
    }

    /* Modern Input */
    .modern-input {
        border: 2px solid var(--border-color);
        border-radius: var(--border-radius);
        padding: 0.75rem 1rem;
        font-size: 1rem;
        transition: var(--transition);
        background: var(--bg-primary);
        color: var(--text-primary);
    }

    .modern-input:focus {
        outline: none;
        border-color: var(--primary-color);
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
    }

    /* Modern Button */
    .modern-button {
        background: linear-gradient(135deg, var(--primary-color), var(--accent-color));
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: var(--border-radius);
        font-weight: 600;
        cursor: pointer;
        transition: var(--transition);
        box-shadow: var(--shadow-sm);
    }

    .modern-button:hover {
        transform: translateY(-1px);
        box-shadow: var(--shadow-lg);
    }

    .modern-button:active {
        transform: translateY(0);
    }

    /* Modern Table */
    .modern-table {
        background: var(--bg-primary);
        border-radius: var(--border-radius);
        overflow: hidden;
        box-shadow: var(--shadow-md);
    }

    .modern-table table {
        width: 100%;
        border-collapse: collapse;
    }

    .modern-table th {
        background: var(--bg-secondary);
        padding: 1rem;
        text-align: left;
        font-weight: 600;
        color: var(--text-primary);
        border-bottom: 1px solid var(--border-color);
    }

    .modern-table td {
        padding: 1rem;
        border-bottom: 1px solid var(--border-color);
        color: var(--text-primary);
    }

    .modern-table tr:hover {
        background: var(--bg-secondary);
    }

    /* Responsive Grid */
    .dashboard-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 1.5rem;
        margin: 1rem 0;
    }

    /* Chart Container */
    .chart-container {
        background: var(--bg-primary);
        border-radius: var(--border-radius);
        padding: 1.5rem;
        box-shadow: var(--shadow-md);
        margin: 1rem 0;
    }

    /* Loading Animation */
    .loading-spinner {
        display: inline-block;
        width: 40px;
        height: 40px;
        border: 4px solid var(--border-color);
        border-radius: 50%;
        border-top-color: var(--primary-color);
        animation: spin 1s ease-in-out infinite;
    }

    @keyframes spin {
        to { transform: rotate(360deg); }
    }

    /* Hide Streamlit elements */
    #MainMenu, footer, header {
        display: none !important;
    }

    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }

    ::-webkit-scrollbar-track {
        background: var(--bg-secondary);
    }

    ::-webkit-scrollbar-thumb {
        background: var(--primary-color);
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: var(--accent-color);
    }
    </style>
    """, unsafe_allow_html=True)

# =============================================================================
# MODERN UI COMPONENTS
# =============================================================================

def create_modern_card(title, value, delta=None, icon=None):
    """Create a modern metric card with hover effects"""
    delta_class = "positive" if delta and float(delta.strip('%')) > 0 else "negative" if delta else ""
    delta_symbol = "+" if delta and float(delta.strip('%')) > 0 else ""
    
    card_html = f"""
    <div class="modern-card">
        <div class="metric-container">
            <div class="metric-label">{title}</div>
            <div class="metric-value">{value}</div>
            {f'<div class="metric-delta {delta_class}">{delta_symbol}{delta}</div>' if delta else ''}
        </div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)

def create_section_header(title, subtitle=None):
    """Create a modern section header"""
    st.markdown(f"""
    <div style="margin: 2rem 0 1rem 0;">
        <h2 style="color: var(--text-primary); font-weight: 700; margin: 0;">{title}</h2>
        {f'<p style="color: var(--text-secondary); margin: 0.5rem 0 0 0;">{subtitle}</p>' if subtitle else ''}
    </div>
    """, unsafe_allow_html=True)

# =============================================================================
# MODERNIZED APP LAYOUT
# =============================================================================

# Configure the app with modern settings
st.set_page_config(
    page_title='Modern Financial Dashboard',
    page_icon='📊',
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Inject custom CSS
inject_custom_css()

# Modern header
st.markdown("""
<div class="modern-header">
    <h1 style="margin: 0; font-size: 2.5rem; font-weight: 700;">Financial Analytics Dashboard</h1>
    <p style="margin: 0.5rem 0 0 0; opacity: 0.9; font-size: 1.1rem;">Real-time insights for smarter investment decisions</p>
</div>
""", unsafe_allow_html=True)

# Modern search section
col1, col2, col3 = st.columns([3, 1, 1])
with col1:
    symbol_input = st.text_input(
        "Enter stock symbol",
        placeholder="e.g., AAPL, MSFT, GOOGL",
        label_visibility="collapsed"
    ).upper()

with col2:
    st.write("")  # Spacing
    st.write("")  # Spacing
    search_clicked = st.button("🔍 Analyze", type="primary", use_container_width=True)

with col3:
    st.write("")  # Spacing
    st.write("")  # Spacing
    if st.button("⚙️ Settings", use_container_width=True):
        st.session_state.show_settings = True

# =============================================================================
# MODERN DATA DISPLAY
# =============================================================================

if symbol_input and search_clicked:
    try:
        # Fetch data (using your existing functions)
        company_data = get_company_info(symbol_input)
        metrics_data = key_metrics(symbol_input)
        income_data = income_statement(symbol_input)
        performance_data = stock_price(symbol_input)
        ratios_data = financial_ratios(symbol_input)
        balance_sheet_data = balance_sheet(symbol_input)
        cashflow_data = cash_flow(symbol_input)

        if company_data:
            # Modern company header
            st.markdown(f"""
            <div style="display: flex; align-items: center; gap: 1rem; margin: 2rem 0;">
                <img src="{company_data['Image']}" style="width: 60px; height: 60px; border-radius: 50%; box-shadow: var(--shadow-md);">
                <div>
                    <h2 style="margin: 0; font-size: 1.8rem;">{company_data['Name']}</h2>
                    <p style="margin: 0; color: var(--text-secondary);">{company_data['Sector']} • {company_data['Exchange']}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Modern metrics grid
            st.markdown('<div class="dashboard-grid">', unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                create_modern_card(
                    "Current Price", 
                    f"${company_data['Price']}",
                    f"{company_data['Price change']:.2f}%"
                )
            with col2:
                create_modern_card(
                    "Market Cap", 
                    millify(company_data['Market Cap'], precision=2)
                )
            with col3:
                create_modern_card(
                    "P/E Ratio", 
                    f"{metrics_data['P/E Ratio'][0]:.2f}"
                )
            with col4:
                create_modern_card(
                    "Dividend Yield", 
                    f"{metrics_data['Dividend Yield'][0]*100:.2f}%"
                )
            
            st.markdown('</div>', unsafe_allow_html=True)

            # Modern tabs for different views
            tab1, tab2, tab3, tab4 = st.tabs(["📈 Performance", "📊 Financials", "💰 Cash Flow", "📋 Ratios"])
            
            with tab1:
                create_section_header("Price Performance", "Last 5 years of monthly data")
                # Your existing performance chart with modern styling
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=performance_data.index,
                    y=performance_data['Price'],
                    mode='lines',
                    line=dict(color='#3b82f6', width=3),
                    fill='tonexty',
                    fillcolor='rgba(59, 130, 246, 0.1)'
                ))
                fig.update_layout(
                    height=400,
                    margin=dict(l=0, r=0, t=0, b=0),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(showgrid=False),
                    yaxis=dict(showgrid=False, tickformat='$,.0f')
                )
                st.plotly_chart(fig, use_container_width=True)

            with tab2:
                col1, col2 = st.columns([2, 1])
                with col1:
                    create_section_header("Income Statement", "Annual financial performance")
                    st.dataframe(
                        income_data.applymap(lambda x: millify(x, precision=2)),
                        use_container_width=True
                    )
                with col2:
                    create_section_header("Key Metrics")
                    st.dataframe(
                        metrics_data[['Market Cap', 'ROE', 'D/E ratio']].applymap(lambda x: f"{x:.2f}"),
                        use_container_width=True
                    )

            with tab3:
                create_section_header("Cash Flow Analysis")
                # Modern cash flow visualization
                fig = go.Figure()
                categories = ['Operating', 'Investing', 'Financing', 'Free Cash Flow']
                values = [
                    cashflow_data['Cash flows from operating activities'].iloc[0],
                    cashflow_data['Cash flows from investing activities'].iloc[0],
                    cashflow_data['Cash flows from financing activities'].iloc[0],
                    cashflow_data['Free cash flow'].iloc[0]
                ]
                
                fig.add_trace(go.Bar(
                    x=categories,
                    y=values,
                    marker_color=['#10b981', '#f59e0b', '#3b82f6', '#8b5cf6'],
                    text=[millify(v, precision=2) for v in values],
                    textposition='outside'
                ))
                fig.update_layout(
                    height=400,
                    margin=dict(l=0, r=0, t=0, b=0),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(showgrid=False),
                    yaxis=dict(showgrid=False, tickformat=',.0f')
                )
                st.plotly_chart(fig, use_container_width=True)

            with tab4:
                create_section_header("Financial Ratios")
                st.dataframe(
                    ratios_data[['Current Ratio', 'ROE', 'ROA', 'Debt Ratio']].applymap(lambda x: f"{x:.2f}"),
                    use_container_width=True
                )

            # Modern download section
            st.markdown("---")
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("📥 Download Complete Report", type="primary", use_container_width=True):
                    # Your existing download logic
                    st.success("Report downloaded successfully!")

    except Exception as e:
        st.error("Unable to fetch data. Please check the symbol and try again.")

else:
    # Modern welcome screen
    st.markdown("""
    <div style="text-align: center; padding: 4rem 2rem;">
        <h2 style="color: var(--text-primary); margin-bottom: 1rem;">Welcome to Modern Financial Analytics</h2>
        <p style="color: var(--text-secondary); font-size: 1.2rem; margin-bottom: 2rem;">
            Enter a stock symbol above to begin your analysis
        </p>
        <div style="display: flex; justify-content: center; gap: 1rem; flex-wrap: wrap;">
            <span style="background: var(--bg-primary); padding: 0.5rem 1rem; border-radius: 2rem; border: 1px solid var(--border-color);">AAPL</span>
            <span style="background: var(--bg-primary); padding: 0.5rem 1rem; border-radius: 2rem; border: 1px solid var(--border-color);">MSFT</span>
            <span style="background: var(--bg-primary); padding: 0.5rem 1rem; border-radius: 2rem; border: 1px solid var(--border-color);">GOOGL</span>
            <span style="background: var(--bg-primary); padding: 0.5rem 1rem; border-radius: 2rem; border: 1px solid var(--border-color);">TSLA</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
