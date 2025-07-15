
# Standard library imports
import sys
from io import BytesIO

# Third-party imports
import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objs as go
from millify import millify

# =============================================================================
# CONFIGURATION
# =============================================================================

# Configure the app page
st.set_page_config(
    page_title='Financial Dashboard',
    page_icon='📈',
    layout="centered",
)

# =============================================================================
# UTILITIES (from utils.py)
# =============================================================================

def config_menu_footer() -> None:
    """
    Hides the Streamlit menu and replaces footer.
    """
    app_style = """
        <style>
            #MainMenu {
              visibility: hidden;
            }
            footer {
                visibility: hidden;
            }
            footer:before {
            content:"Copyright © 2023 Abel Tavares";
            visibility: visible;
            display: block;
            position: relative;
            text-align: center;
            }
        </style>
    """
    st.markdown(app_style, unsafe_allow_html=True)

def get_delta(df: pd.DataFrame, key: str) -> str:
    """
    Calculates the real percentage difference between the first two values for a given key in a Pandas DataFrame.
    """
    if key not in df.columns:
        return f"Key '{key}' not found in DataFrame columns."

    if len(df) < 2:
        return "DataFrame must contain at least two rows."

    val1 = df[key][1]
    val2 = df[key][0]

    # Handle cases where either value is negative or zero
    if val1 <= 0 or val2 <= 0:
        delta = (val2 - val1) / abs(val1) * 100
    else:
        delta = (val2 - val1) / val1 * 100

    # Round to two decimal places and return the result
    return f"{delta:.2f}%"

def empty_lines(n: int) -> None:
    """
    Inserts empty lines to separate content.
    """
    for _ in range(n):
        st.write("")

def generate_card(text: str) -> None:
    """
    Generates a styled card with a title and icon.
    """
    st.markdown(f"""
        <div style='border: 1px solid #e6e6e6; border-radius: 5px; padding: 10px; display: flex; justify-content: center; align-items: center'>
            <i class='fas fa-chart-line' style='font-size: 24px; color: #0072C6; margin-right: 10px'></i>
            <h3 style='text-align: center'>{text}</h3>
        </div>
         """, unsafe_allow_html=True)

def color_highlighter(val: str) -> str:
    """
    Returns CSS styling for a pandas DataFrame cell based on whether its value is positive or negative.
    """
    if val.startswith('-'):
        return 'color: rgba(255, 0, 0, 0.9);'
    else:
        return None

# =============================================================================
# DATA (from data.py)
# =============================================================================

# Load API keys from Streamlit secrets
FMP_API_KEY = st.secrets["ZPODKN7Q87COJ0IR"]
ALPHA_API_KEY = st.secrets["OoJcYpvMo94etCgLpr1s6TABcmhr7AWT"]

def get_company_info(symbol: str) -> dict:
    """
    Returns a dictionary containing information about a company with the given stock symbol.
    """
    api_endpoint = f'https://financialmodelingprep.com/api/v3/profile/{symbol}/'
    params = {'apikey': FMP_API_KEY}
    try:
        response = requests.get(api_endpoint, params=params)
        response.raise_for_status()
        data = response.json()[0]
        company_info = {
            'Name': data['companyName'],
            'Exchange': data['exchangeShortName'],
            'Currency': data['currency'],
            'Country': data['country'],
            'Sector': data['sector'],
            'Market Cap': data['mktCap'],
            'Price': data['price'],
            'Beta': data['beta'],
            'Price change': data['changes'],
            'Website': data['website'],
            'Image': data['image']
        }
        return company_info
    except Exception as e:
        st.error(f"Error fetching company info: {e}")
        return None

def get_stock_price(symbol: str) -> pd.DataFrame:
    """
    Returns a Pandas DataFrame containing the monthly adjusted closing prices of a given stock symbol for the last 5 years.
    """
    api_endpoint = 'https://www.alphavantage.co/query'
    params = {
        'function': 'TIME_SERIES_MONTHLY_ADJUSTED',
        'symbol': symbol,
        'apikey': ALPHA_API_KEY,
    }
    try:
        response = requests.get(api_endpoint, params=params)
        response.raise_for_status()
        data = response.json()['Monthly Adjusted Time Series']
        df = pd.DataFrame.from_dict(data, orient='index')
        df.index = pd.to_datetime(df.index)
        df = df[:12*5]
        df = df[['4. close']].astype(float)
        df = df.rename(columns={'4. close': 'Price'})
        return df
    except Exception as e:
        st.error(f"Error fetching stock price: {e}")
        return None

def get_income_statement(symbol: str) -> pd.DataFrame:
    """
    Retrieves the income statement data for a given stock symbol from the Financial Modeling Prep API.
    """
    api_endpoint = f'https://financialmodelingprep.com/api/v3/income-statement/{symbol}/'
    params = {'limit': 5, 'apikey': FMP_API_KEY}
    try:
        income_statement_data = []
        response = requests.get(api_endpoint, params=params)
        response.raise_for_status()
        response_data = response.json()
        for report in response_data:
            year = report['calendarYear']
            income_statement_data.append({
                'Year': year,
                'Revenue': report['revenue'],
                '(-) Cost of Revenue': report['costOfRevenue'],
                '= Gross Profit': report['grossProfit'],
                '(-) Operating Expense': report['operatingExpenses'],
                '= Operating Income': report['operatingIncome'],
                '(+-) Other Income/Expenses': report['totalOtherIncomeExpensesNet'],
                '= Income Before Tax': report['incomeBeforeTax'],
                '(+-) Tax Income/Expense': report['incomeTaxExpense'],
                '= Net Income': report['netIncome'],
            })
        return pd.DataFrame(income_statement_data).set_index('Year')
    except Exception as e:
        st.error(f"Error fetching income statement: {e}")
        return None

def get_balance_sheet(symbol: str) -> pd.DataFrame:
    """
    Retrieves the balance sheet data for a given stock symbol.
    """
    api_endpoint = f'https://financialmodelingprep.com/api/v3/balance-sheet-statement/{symbol}'
    params = {'limit': 5, 'apikey': FMP_API_KEY}
    try:
        balance_sheet_data = []
        response = requests.get(api_endpoint, params=params)
        response.raise_for_status()
        response_data = response.json()
        for report in response_data:
            year = report['calendarYear']
            balance_sheet_data.append({
                'Year': year,
                'Assets': report['totalAssets'],
                'Current Assets': report['totalCurrentAssets'],
                'Non-Current Assets': report['totalNonCurrentAssets'],
                'Current Liabilities': report['totalCurrentLiabilities'],
                'Non-Current Liabilities': report['totalNonCurrentLiabilities'],
                'Liabilities': report['totalLiabilities'],
                'Equity': report['totalEquity']
            })
        return pd.DataFrame(balance_sheet_data).set_index('Year')
    except Exception as e:
        st.error(f"Error fetching balance sheet: {e}")
        return None

def get_cash_flow(symbol: str) -> pd.DataFrame:
    """
    Retrieve cash flow data for a given stock symbol from the Financial Modeling Prep API.
    """
    api_endpoint = f'https://financialmodelingprep.com/api/v3/cash-flow-statement/{symbol}'
    params = {'limit': 5, 'apikey': FMP_API_KEY}
    try:
        cashflow_data = []
        response = requests.get(api_endpoint, params=params)
        response.raise_for_status()
        response_data = response.json()
        for report in response_data:
            year = report['date'].split('-')[0]
            cashflow_data.append({
                'Year': year,
                "Cash flows from operating activities": report['netCashProvidedByOperatingActivities'],
                'Cash flows from investing activities': report['netCashUsedForInvestingActivites'],
                'Cash flows from financing activities': report['netCashUsedProvidedByFinancingActivities'],
                'Free cash flow': report['freeCashFlow']
            })
        return pd.DataFrame(cashflow_data).set_index('Year')
    except Exception as e:
        st.error(f"Error fetching cash flow: {e}")
        return None

def get_key_metrics(symbol: str) -> pd.DataFrame:
    """
    Returns a Pandas DataFrame containing the key financial metrics of a given company symbol for the last 10 years.
    """
    api_endpoint = f'https://financialmodelingprep.com/api/v3/key-metrics/{symbol}'
    params = {'limit': 5, 'apikey': FMP_API_KEY}
    try:
        metrics_data = []
        response = requests.get(api_endpoint, params=params)
        response.raise_for_status()
        response_data = response.json()
        for report in response_data:
            year = report['date'].split('-')[0]
            metrics_data.append({
                'Year': year,
                "Market Cap": report['marketCap'],
                'Working Capital': report['workingCapital'],
                'D/E ratio': report['debtToEquity'],
                'P/E Ratio': report['peRatio'],
                'ROE': report['roe'],
                'Dividend Yield': report['dividendYield']
            })
        return pd.DataFrame(metrics_data).set_index('Year')
    except Exception as e:
        st.error(f"Error fetching key metrics: {e}")
        return None

def get_financial_ratios(symbol: str) -> pd.DataFrame:
    """
    Fetches financial ratios for a given stock symbol using the Financial Modeling Prep API.
    """
    api_endpoint = f'https://financialmodelingprep.com/api/v3/ratios/{symbol}'
    params = {'limit': 5, 'apikey': FMP_API_KEY}
    try:
        ratios_data = []
        response = requests.get(api_endpoint, params=params)
        response.raise_for_status()
        response_data = response.json()
        for report in response_data:
            year = report['date'].split('-')[0]
            ratios_data.append({
                'Year': year,
                'Current Ratio': report['currentRatio'],
                'Quick Ratio': report['quickRatio'],
                'Cash Ratio': report['cashRatio'],
                'Days of Sales Outstanding': report['daysOfSalesOutstanding'],
                'Days of Inventory Outstanding': report['daysOfInventoryOutstanding'],
                'Operating Cycle': report['operatingCycle'],
                'Days of Payables Outstanding': report['daysOfPayablesOutstanding'],
                'Cash Conversion Cycle': report['cashConversionCycle'],
                'Gross Profit Margin': report['grossProfitMargin'],
                'Operating Profit Margin': report['operatingProfitMargin'],
                'Pretax Profit Margin': report['pretaxProfitMargin'],
                'Net Profit Margin': report['netProfitMargin'],
                'Effective Tax Rate': report['effectiveTaxRate'],
                'Return on Assets': report['returnOnAssets'],
                'Return on Equity': report['returnOnEquity'],
                'Return on Capital Employed': report['returnOnCapitalEmployed'],
                'Net Income per EBT': report['netIncomePerEBT'],
                'EBT per EBIT': report['ebtPerEbit'],
                'EBIT per Revenue': report['ebitPerRevenue'],
                'Debt Ratio': report['debtRatio'],
                'Debt Equity Ratio': report['debtEquityRatio'],
                'Long-term Debt to Capitalization': report['longTermDebtToCapitalization'],
                'Total Debt to Capitalization': report['totalDebtToCapitalization'],
                'Interest Coverage': report['interestCoverage'],
                'Cash Flow to Debt Ratio': report['cashFlowToDebtRatio'],
                'Company Equity Multiplier': report['companyEquityMultiplier'],
                'Receivables Turnover': report['receivablesTurnover'],
                'Payables Turnover': report['payablesTurnover'],
                'Inventory Turnover': report['inventoryTurnover'],
                'Fixed Asset Turnover': report['fixedAssetTurnover'],
                'Asset Turnover': report['assetTurnover'],
                'Operating Cash Flow per Share': report['operatingCashFlowPerShare'],
                'Free Cash Flow per Share': report['freeCashFlowPerShare'],
                'Cash per Share': report['cashPerShare'],
                'Payout Ratio': report['payoutRatio'],
                'Operating Cash Flow Sales Ratio': report['operatingCashFlowSalesRatio'],
                'Free Cash Flow Operating Cash Flow Ratio': report['freeCashFlowOperatingCashFlowRatio'],
                'Cash Flow Coverage Ratios': report['cashFlowCoverageRatios'],
                'Price to Book Value Ratio': report['priceToBookRatio'],
                'Price to Earnings Ratio': report['priceEarningsRatio'],
                'Price to Sales Ratio': report['priceToSalesRatio'],
                'Dividend Yield': report['dividendYield'],
                'Enterprise Value to EBITDA': report['enterpriseValueMultiple'],
                'Price to Fair Value': report['priceFairValue']
            })
        return pd.DataFrame(ratios_data).set_index('Year')
    except Exception as e:
        st.error(f"Error fetching financial ratios: {e}")
        return None

# =============================================================================
# CACHED DATA FUNCTIONS
# =============================================================================

@st.cache_data(ttl=60*60*24*30)
def company_info(symbol):
    return get_company_info(symbol)

@st.cache_data(ttl=60*60*24*30)
def income_statement(symbol):
    return get_income_statement(symbol)

@st.cache_data(ttl=60*60*24*30)
def balance_sheet(symbol):
    return get_balance_sheet(symbol)

@st.cache_data(ttl=60*60*24*30)
def stock_price(symbol):
    return get_stock_price(symbol)

@st.cache_data(ttl=60*60*24*30)
def financial_ratios(symbol):
    return get_financial_ratios(symbol)

@st.cache_data(ttl=60*60*24*30)
def key_metrics(symbol):
    return get_key_metrics(symbol)

@st.cache_data(ttl=60*60*24*30)
def cash_flow(symbol):
    return get_cash_flow(symbol)

@st.cache_data(ttl=60*60*24*30)
def delta(df,key):
    return get_delta(df,key)

# =============================================================================
# MAIN APP (from app.py)
# =============================================================================

# Configure the menu and footer
config_menu_footer()

# Display the app title
st.title("Financial Dashboard 📈")

# Initialize the state of the button
if 'btn_clicked' not in st.session_state:
    st.session_state['btn_clicked'] = False

# Define a callback function for when the "Go" button is clicked
def callback():
    st.session_state['btn_clicked'] = True

# Create a text input field for the user to enter a stock ticker
symbol_input = st.text_input("Enter a stock ticker").upper()

# Check if the "Go" button has been clicked
if st.button('Go', on_click=callback) or st.session_state['btn_clicked']:
    if not symbol_input:
        st.warning('Please input a ticker.')
        st.stop()

    try:
        company_data = company_info(symbol_input)
        metrics_data = key_metrics(symbol_input)
        income_data = income_statement(symbol_input)
        performance_data = stock_price(symbol_input)
        ratios_data = financial_ratios(symbol_input)
        balance_sheet_data = balance_sheet(symbol_input)
        cashflow_data = cash_flow(symbol_input)
    except Exception:
        st.error('Not possible to retrieve data for that ticker. Please check if its valid and try again.')
        sys.exit()

    if company_data is None:
        st.error('Failed to retrieve company data. Please check the ticker and try again.')
        st.stop()

    # Display dashboard
    empty_lines(2)
    try:
        # Display company info
        col1, col2 = st.columns((8.5,1.5))
        with col1:
            generate_card(company_data['Name'])
        with col2:
            image_html = f"<a href='{company_data['Website']}' target='_blank'><img src='{company_data['Image']}' alt='{company_data['Name']}' height='75' width='95'></a>"
            st.markdown(image_html, unsafe_allow_html=True)

        col3, col4, col5, col6, col7 = st.columns((0.2,1.4,1.4,2,2.6))

        with col4:
            empty_lines(1)
            st.metric(label="Price", value=company_data['Price'], delta=company_data['Price change'])
            empty_lines(2)

        with col5:
            empty_lines(1)
            generate_card(company_data['Currency'])
            empty_lines(2)

        with col6:
            empty_lines(1)
            generate_card(company_data['Exchange'])
            empty_lines(2)

        with col7:
            empty_lines(1)
            generate_card(company_data['Sector'])
            empty_lines(2)

        # Define columns for key metrics and IS
        col8, col9, col10 = st.columns((2,2,3))

        # Display key metrics
        with col8:
            empty_lines(3)
            st.metric(label="Market Cap", value=millify(metrics_data['Market Cap'][0], precision=2), delta=delta(metrics_data,'Market Cap'))
            st.write("")
            st.metric(label="D/E Ratio", value = round(metrics_data['D/E ratio'][0],2), delta=delta(metrics_data,'D/E ratio'))
            st.write("")
            st.metric(label="ROE", value = str(round(metrics_data['ROE'][0] * 100, 2)) + '%', delta=delta(metrics_data,'ROE'))

        with col9:
            empty_lines(3)
            st.metric(label="Working Capital", value = millify(metrics_data['Working Capital'][0], precision = 2), delta=delta(metrics_data,'Working Capital'))
            st.write("")
            st.metric(label="P/E Ratio", value = round(metrics_data['P/E Ratio'][0],2), delta=delta(metrics_data,'P/E Ratio'))
            st.write("")
            if metrics_data['Dividend Yield'][0] == 0:
                st.metric(label="Dividends (yield)", value = '0')
            else:
                st.metric(label="Dividends (yield)", value = str(round(metrics_data['Dividend Yield'][0]* 100, 2)) + '%', delta=delta(metrics_data,'Dividend Yield'))

        with col10:
            income_statement_data = income_data.T
            st.markdown('**Income Statement**')
            year = st.selectbox('All numbers in thousands', income_statement_data.columns, label_visibility='collapsed')
            income_statement_data = income_statement_data.loc[:, [year]]
            income_statement_data = income_statement_data.applymap(lambda x: millify(x, precision=2))
            income_statement_data = income_statement_data.style.applymap(color_highlighter)
            headers = {'selector': 'th:not(.index_name)', 'props': [('color', 'black')]}
            income_statement_data.set_table_styles([headers])
            st.table(income_statement_data)

        # Configure the plots bar
        config = {
            'displaylogo': False,
            'modeBarButtonsToRemove': ['zoom2d', 'pan2d', 'select2d', 'lasso2d', 'hoverClosestCartesian', 'hoverCompareCartesian', 'autoScale2d', 'toggleSpikelines', 'resetScale2d', 'zoomIn2d', 'zoomOut2d', 'hoverClosest3d', 'hoverClosestGeo', 'hoverClosestGl2d', 'hoverClosestPie', 'toggleHover', 'resetViews', 'toggleSpikeLines', 'resetViewMapbox', 'resetGeo', 'hoverClosestGeo', 'sendDataToCloud', 'hoverClosestGl']
        }

        # Display market performance
        line_color = 'rgb(60, 179, 113)' if performance_data.iloc[0]['Price'] > performance_data.iloc[-1]['Price'] else 'rgb(255, 87, 48)'
        fig = go.Figure(
            go.Scatter(
                x=performance_data.index,
                y=performance_data['Price'],
                mode='lines',
                name='Price',
                line=dict(color=line_color)
            )
        )
        fig.update_layout(
            title={'text': 'Market Performance'},
            dragmode='pan',
            xaxis=dict(fixedrange=True),
            yaxis=dict(fixedrange=True)
        )
        st.plotly_chart(fig, config=config, use_container_width=True)

        # Display net income
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=income_data.index,
                y=income_data["= Net Income"],
                mode="lines+markers",
                line=dict(color="purple"),
                marker=dict(size=5)
            )
        )
        fig.update_layout(
            title="Net Income",
            dragmode='pan',
            xaxis=dict(tickmode='array', tickvals=income_data.index, fixedrange=True),
            yaxis=dict(fixedrange=True),
        )
        st.plotly_chart(fig, config=config, use_container_width=True)

        # Display profitability margins
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=ratios_data.index,
            x=ratios_data['Gross Profit Margin'],
            name='Gross Profit Margin',
            marker=dict(color='rgba(60, 179, 113, 0.85)'),
            orientation='h',
        ))
        fig.add_trace(go.Bar(
            y=ratios_data.index,
            x=ratios_data['Operating Profit Margin'],
            name='EBIT Margin',
            marker=dict(color='rgba(30, 144, 255, 0.85)'),
            orientation='h',
        ))
        fig.add_trace(go.Bar(
            y=ratios_data.index,
            x=ratios_data['Net Profit Margin'],
            name='Net Profit Margin',
            marker=dict(color='rgba(173, 216, 230, 0.85)'),
            orientation='h',
        ))
        fig.update_layout(
            title='Profitability Margins',
            bargap=0.1,
            dragmode='pan',
            xaxis=dict(fixedrange=True, tickformat='.0%'),
            yaxis=dict(fixedrange=True)
        )
        st.plotly_chart(fig, config=config, use_container_width=True)

        # Display balance sheet
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=balance_sheet_data.index,
            y=balance_sheet_data['Assets'],
            name='Assets',
            marker=dict(color='rgba(60, 179, 113, 0.85)'),
            width=0.3,
        ))
        fig.add_trace(go.Bar(
            x=balance_sheet_data.index,
            y=balance_sheet_data['Liabilities'],
            name='Liabilities',
            marker=dict(color='rgba(255, 99, 71, 0.85)'),
            width=0.3,
        ))
        fig.add_trace(go.Scatter(
            x=balance_sheet_data.index,
            y=balance_sheet_data['Equity'],
            mode='lines+markers',
            name='Equity',
            line=dict(color='rgba(173, 216, 230, 1)', width=2),
            marker=dict(symbol='circle', size=8, color='rgba(173, 216, 230, 1)', line=dict(width=1, color='rgba(173, 216, 230, 1)'))
        ))
        fig.update_layout(
            title='Balance Sheet',
            bargap=0.4,
            dragmode='pan',
            xaxis=dict(fixedrange=True),
            yaxis=dict(fixedrange=True)
        )
        st.plotly_chart(fig, config=config, use_container_width=True)

        # Display ROE and ROA
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=ratios_data.index,
            y=ratios_data['Return on Equity'],
            name='ROE',
            line=dict(color='rgba(60, 179, 113, 0.85)'),
        ))
        fig.add_trace(go.Scatter(
            x=ratios_data.index,
            y=ratios_data['Return on Assets'],
            name='ROA',
            line=dict(color='rgba(30, 144, 255, 0.85)'),
        ))
        fig.update_layout(
            title='ROE and ROA',
            dragmode='pan',
            xaxis=dict(fixedrange=True),
            yaxis=dict(fixedrange=True, tickformat='.0%')
        )
        st.plotly_chart(fig, config=config, use_container_width=True)

        # Display cash flows
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=cashflow_data.index,
            y=cashflow_data['Cash flows from operating activities'],
            name='Cash flows from operating activities',
            marker=dict(color='rgba(60, 179, 113, 0.85)'),
            width=0.3,
        ))
        fig.add_trace(go.Bar(
            x=cashflow_data.index,
            y=cashflow_data['Cash flows from investing activities'],
            name='Cash flows from investing activities',
            marker=dict(color='rgba(30, 144, 255, 0.85)'),
            width=0.3,
        ))
        fig.add_trace(go.Bar(
            x=cashflow_data.index,
            y=cashflow_data['Cash flows from financing activities'],
            name='Cash flows from financing activities',
            marker=dict(color='rgba(173, 216, 230, 0.85)'),
            width=0.3,
        ))
        fig.add_trace(go.Scatter(
            x=cashflow_data.index,
            y=cashflow_data['Free cash flow'],
            mode='lines+markers',
            name='Free cash flow',
            line=dict(color='rgba(255, 140, 0, 1)', width=2),
            marker=dict(symbol='circle', size=5, color='rgba(255, 140, 0, 1)', line=dict(width=0.8, color='rgba(255, 140, 0, 1)'))
        ))
        fig.update_layout(
            title='Cash flows',
            bargap=0.1,
            xaxis=dict(fixedrange=True),
            yaxis=dict(fixedrange=True)
        )
        st.plotly_chart(fig, config=config, use_container_width=True)

        # Display financial ratios table
        empty_lines(1)
        st.markdown('**Financial Ratios**')
        ratios_table = ratios_data.rename(columns={
            'Days of Sales Outstanding': 'Days of Sales Outstanding (days)',
            'Days of Inventory Outstanding': 'Days of Inventory Outstanding (days)',
            'Operating Cycle': 'Operating Cycle (days)',
            'Days of Payables Outstanding': 'Days of Payables Outstanding (days)',
            'Cash Conversion Cycle': 'Cash Conversion Cycle (days)',
            'Gross Profit Margin': 'Gross Profit Margin (%)',
            'Operating Profit Margin': 'Operating Profit Margin (%)',
            'Pretax Profit Margin': 'Pretax Profit Margin (%)',
            'Net Profit Margin': 'Net Profit Margin (%)',
            'Effective Tax Rate': 'Effective Tax Rate (%)',
            'Return on Assets': 'Return on Assets (%)',
            'Return on Equity': 'Return on Equity (%)',
            'Return on Capital Employed': 'Return on Capital Employed (%)',
            'EBIT per Revenue': 'EBIT per Revenue (%)',
            'Debt Ratio': 'Debt Ratio (%)',
            'Long-term Debt to Capitalization': 'Long-term Debt to Capitalization (%)',
            'Total Debt to Capitalization': 'Total Debt to Capitalization (%)',
            'Payout Ratio': 'Payout Ratio (%)',
            'Operating Cash Flow Sales Ratio': 'Operating Cash Flow Sales Ratio (%)',
            'Free Cash Flow Operating Cash Flow Ratio': 'Free Cash Flow Operating Cash Flow Ratio (%)',
            'Dividend Yield': 'Dividend Yield (%)',
        })

        for col in ratios_table.columns:
            if "%" in col:
                ratios_table[col] = ratios_table[col] * 100

        ratios_table = round(ratios_table.T, 2)
        ratios_table = ratios_table.sort_index(axis=1, ascending=True)
        st.dataframe(ratios_table, width=800, height=400)

    except Exception as e:
        st.error('Not possible to develop dashboard. Please try again.')
        sys.exit()

    # Add download button
    empty_lines(3)
    try:
        company_df = pd.DataFrame.from_dict(company_data, orient='index').reset_index().rename(columns={'index':'Key', 0:'Value'}).set_index('Key')
        metrics_df = metrics_data.round(2).T
        income_df = income_data.round(2)
        ratios_df = ratios_data.round(2).T
        balance_df = balance_sheet_data.round(2).T
        cashflow_df = cashflow_data.T

        income_df.columns = income_df.columns.str.replace(r'[\/\(\)\-\+=]\s?', '', regex=True)
        income_df = income_df.T

        dfs = {
            'Stock': company_df,
            'Market Performance': performance_data,
            'Income Statement': income_df,
            'Balance Sheet': balance_df,
            'Cash flow': cashflow_df,
            'Key Metrics': metrics_df,
            'Financial Ratios': ratios_table
        }

        output = BytesIO()
        writer = pd.ExcelWriter(output, engine='xlsxwriter')
        for sheet_name, df in dfs.items():
            if sheet_name == 'Market Performance':
                df.index.name = 'Date'
                df = df.reset_index()
                df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
                df.to_excel(writer, sheet_name=sheet_name, index=False)
            else:
                df.to_excel(writer, sheet_name=sheet_name, index=True)
            writer.sheets[sheet_name].autofit()
        writer.close()

        data = output.getvalue()
        st.download_button(
            label='Download ' + symbol_input + ' Financial Data (.xlsx)',
            data=data,
            file_name=symbol_input + '_financial_data.xlsx',
            mime='application/octet-stream'
        )
    except Exception:
        st.info('Data not available for download')
