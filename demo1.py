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


# Configure the app page
st.set_page_config(
    page_title="Financial Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def apply_modern_css():
    """
    Apply modern CSS styling with glassmorphism effects, hover animations, and improved UI
    """
    modern_css = """
    <style>
        /* Import Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

        /* Root variables for LİGHT MODE (default) */
        :root {
            --primary-color: #667eea;
            --secondary-color: #764ba2;
            --accent-color: #f093fb;
            --text-primary: #1a1a1a;
            --text-secondary: #666666;
            --background-gradient: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            --glass-bg: rgba(255, 255, 255, 0.4);
            --glass-border: rgba(255, 255, 255, 0.2);
            --shadow-light: 0 8px 32px rgba(31, 38, 135, 0.2);
            --shadow-hover: 0 15px 35px rgba(31, 38, 135, 0.25);
            --border-radius: 16px;
            --transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }

        /* Root variables for DARK MODE */
        @media (prefers-color-scheme: dark) {
            :root {
                --text-primary: #f0f2f6;
                --text-secondary: #a0a0a0;
                --background-gradient: linear-gradient(135deg, #232526 0%, #414345 100%);
                --glass-bg: rgba(40, 40, 40, 0.5);
                --glass-border: rgba(255, 255, 255, 0.15);
                --shadow-light: 0 8px 32px rgba(0, 0, 0, 0.3);
                --shadow-hover: 0 15px 35px rgba(0, 0, 0, 0.35);
            }

            .main-header p {
                color: var(--text-secondary) !important;
            }

            .stDataFrame th {
                background: rgba(255, 255, 255, 0.1);
                color: var(--text-primary);
            }

            .stDataFrame td {
                background: rgba(255, 255, 255, 0.05);
                color: var(--text-primary);
            }
        }


        /* Global styles */
        .stApp {
            font-family: 'Inter', sans-serif;
            background: var(--background-gradient);
            min-height: 100vh;
        }

        /* Main container */
        .main .block-container {
            max-width: 1200px;
            padding: 2rem 1rem;
            margin: 0 auto;
        }

        /* Header styling */
        .main-header {
            background: var(--glass-bg);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid var(--glass-border);
            border-radius: var(--border-radius);
            padding: 2rem;
            margin-bottom: 2rem;
            box-shadow: var(--shadow-light);
            transition: var(--transition);
            z-index: 999; /* Ensure it's above other content but below sticky */
        }

        .main-header:hover {
            transform: translateY(-2px);
            box-shadow: var(--shadow-hover);
        }

        .main-header h1 {
            color: var(--text-primary);
            font-weight: 700;
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
            text-align: center;
            transition: var(--transition);
        }

        /* Sticky Header Styling */
        @keyframes slideDown {
            from { transform: translateY(-100%); }
            to { transform: translateY(0); }
        }

        .sticky-header {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            border-radius: 0;
            margin-bottom: 0;
            padding: 1rem 2rem;
            animation: slideDown 0.5s ease-in-out;
            z-index: 1000;
        }

        .sticky-header h1 {
            font-size: 1.8rem;
            margin-bottom: 0;
        }

        .sticky-header p {
            display: none; /* Hide paragraph in sticky mode */
        }


        /* Input styling */
        .stTextInput > div > div > input {
            background: var(--glass-bg);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid var(--glass-border);
            border-radius: 12px;
            padding: 0.75rem 1rem;
            font-size: 1.1rem;
            font-weight: 500;
            color: var(--text-primary);
            transition: var(--transition);
        }

        .stTextInput > div > div > input:focus {
            border-color: var(--primary-color);
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }

        /* Button styling */
        .stButton > button {
            background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
            border: none;
            border-radius: 12px;
            padding: 0.75rem 2rem;
            font-weight: 600;
            font-size: 1.1rem;
            color: white;
            transition: var(--transition);
            box-shadow: var(--shadow-light);
        }

        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: var(--shadow-hover);
            background: linear-gradient(135deg, var(--secondary-color), var(--primary-color));
        }

        /* Glass card styling */
        .glass-card {
            background: var(--glass-bg);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid var(--glass-border);
            border-radius: var(--border-radius);
            padding: 1.5rem;
            margin-bottom: 1rem;
            box-shadow: var(--shadow-light);
            transition: var(--transition);
        }

        .glass-card:hover {
            transform: translateY(-3px);
            box-shadow: var(--shadow-hover);
        }

        /* Company info card */
        .company-info {
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: var(--glass-bg);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid var(--glass-border);
            border-radius: var(--border-radius);
            padding: 1.5rem 2rem;
            margin-bottom: 2rem;
            box-shadow: var(--shadow-light);
            transition: var(--transition);
        }

        .company-info:hover {
            transform: translateY(-2px);
            box-shadow: var(--shadow-hover);
        }

        .company-name {
            font-size: 1.8rem;
            font-weight: 700;
            color: var(--text-primary);
            margin-bottom: 0.25rem;
            text-align: left;
        }

        .company-logo {
            margin-left: 1.5rem;
        }

        .company-logo img {
            height: 60px;
            width: 60px;
            object-fit: contain;
            border-radius: 12px;
            background: rgba(255, 255, 255, 0.7);
            padding: 5px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
            transition: var(--transition);
        }

        .company-logo img:hover {
            transform: scale(1.05);
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.15);
        }

        /* Metrics styling */
        .metric-container {
            background: var(--glass-bg);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid var(--glass-border);
            border-radius: var(--border-radius);
            padding: 1.5rem;
            margin-bottom: 1rem;
            box-shadow: var(--shadow-light);
            transition: var(--transition);
            text-align: center;
        }

        .metric-container:hover {
            transform: translateY(-3px);
            box-shadow: var(--shadow-hover);
        }

        .stMetric {
            background: transparent;
        }

        .stMetric > div {
            background: var(--glass-bg);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid var(--glass-border);
            border-radius: 12px;
            padding: 1.5rem;
            transition: var(--transition);
        }

        .stMetric > div:hover {
            transform: translateY(-2px);
            box-shadow: var(--shadow-hover);
        }

        .stMetric label {
            font-weight: 600;
            color: var(--text-secondary);
            font-size: 0.9rem;
        }

        .stMetric [data-testid="metric-value"] {
            font-size: 1.8rem;
            font-weight: 700;
            color: var(--text-primary);
        }

        /* Info cards */
        .info-card {
            background: var(--glass-bg);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid var(--glass-border);
            border-radius: var(--border-radius);
            padding: 1rem;
            text-align: center;
            box-shadow: var(--shadow-light);
            transition: var(--transition);
            margin-bottom: 1rem;
        }

        .info-card:hover {
            transform: translateY(-3px);
            box-shadow: var(--shadow-hover);
        }

        .info-card h3 {
            color: var(--text-primary);
            font-weight: 600;
            margin: 0;
            font-size: 1.1rem;
        }

        /* Chart container */
        .chart-container {
            background: var(--glass-bg);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid var(--glass-border);
            border-radius: var(--border-radius);
            padding: 1.5rem;
            margin-bottom: 2rem;
            box-shadow: var(--shadow-light);
            transition: var(--transition);
            color: var(--text-primary); /* This will color the chart text */
        }

        .chart-container:hover {
            transform: translateY(-2px);
            box-shadow: var(--shadow-hover);
        }

        /* Table styling */
        .stDataFrame {
            background: var(--glass-bg);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid var(--glass-border);
            border-radius: var(--border-radius);
            overflow: hidden;
            box-shadow: var(--shadow-light);
        }

        .stDataFrame table {
            background: transparent;
        }

        .stDataFrame th {
            background: rgba(102, 126, 234, 0.1);
            color: var(--text-primary);
            font-weight: 600;
        }

        .stDataFrame td {
            background: rgba(255, 255, 255, 0.05);
            color: var(--text-primary);
        }

        /* Selectbox styling */
        .stSelectbox > div > div {
            background: var(--glass-bg);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid var(--glass-border);
            border-radius: 12px;
            color: var(--text-primary);
        }

        /* Download button */
        .stDownloadButton > button {
            background: linear-gradient(135deg, var(--accent-color), var(--primary-color));
            border: none;
            border-radius: 12px;
            padding: 0.75rem 2rem;
            font-weight: 600;
            color: white;
            transition: var(--transition);
            box-shadow: var(--shadow-light);
        }

        .stDownloadButton > button:hover {
            transform: translateY(-2px);
            box-shadow: var(--shadow-hover);
        }

        /* Responsive design */
        @media (max-width: 768px) {
            .main .block-container {
                padding: 1rem 0.5rem;
            }

            .main-header h1 {
                font-size: 2rem;
            }

            .glass-card {
                padding: 1rem;
            }
        }

        /* Loading animation */
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }

        .loading {
            animation: pulse 1.5s ease-in-out infinite;
        }

        /* Hide Streamlit elements */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}

        /* Custom footer */
        .custom-footer {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            background: var(--glass-bg);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border-top: 1px solid var(--glass-border);
            padding: 0.5rem; /* Thinner footer */
            text-align: center;
            color: var(--text-secondary);
            font-size: 0.9rem;
            z-index: 1001;
        }
    </style>

    <script>
        document.addEventListener("DOMContentLoaded", function() {
            const header = document.querySelector('.main-header');
            if (!header) return;

            // Create a placeholder to prevent content from jumping when header becomes sticky
            const placeholder = document.createElement('div');
            placeholder.style.display = 'none';
            header.parentNode.insertBefore(placeholder, header);

            const stickyPoint = header.offsetTop;

            window.addEventListener('scroll', function() {
                if (window.pageYOffset > stickyPoint) {
                    if (!header.classList.contains('sticky-header')) {
                        placeholder.style.height = header.offsetHeight + 'px';
                        placeholder.style.marginBottom = getComputedStyle(header).marginBottom;
                        placeholder.style.display = 'block';
                        header.classList.add('sticky-header');
                    }
                } else {
                    if (header.classList.contains('sticky-header')) {
                        header.classList.remove('sticky-header');
                        placeholder.style.display = 'none';
                    }
                }
            });
        });
    </script>
    """
    st.markdown(modern_css, unsafe_allow_html=True)


def create_glass_card(content, hover_effect=True):
    """Create a glass morphism card with content"""
    hover_class = "glass-card" if hover_effect else "glass-card-no-hover"
    return f"""
    <div class="{hover_class}">
        {content}
    </div>
    """


def create_info_card(title, icon="📊"):
    """Create an info card with glassmorphism effect"""
    return f"""
    <div class="info-card">
        <h3>{icon} {title}</h3>
    </div>
    """


def create_metric_card(label, value, delta=None):
    """Create a metric card with glassmorphism effect"""
    delta_html = f"<div class='metric-delta'>{delta}</div>" if delta else ""
    return f"""
    <div class="metric-container">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>
    """


def get_delta(df: pd.DataFrame, key: str) -> str:
    """Calculate percentage difference between the first two values"""
    if key not in df.columns:
        return f"Key '{key}' not found in DataFrame columns."

    if len(df) < 2:
        return "DataFrame must contain at least two rows."

    val1 = df[key][1]  # Second most recent
    val2 = df[key][0]  # Most recent

    if pd.isna(val1) or pd.isna(val2):
        return "N/A"

    if val1 == 0:
        if val2 == 0:
            return "0.00%"  # No change if both are zero
        else:
            return (
                "Inf%" if val2 > 0 else "-Inf%"
            )  # Infinite change if previous was zero
    else:
        delta = (val2 - val1) / val1 * 100

    # Add a sign for positive deltas for consistency
    return f"{delta:+.2f}%"


def color_highlighter(val: str) -> str:
    """Returns CSS styling for DataFrame cells"""
    if isinstance(val, str) and val.startswith("-"):
        return "color: rgba(255, 77, 77, 0.9);"
    elif isinstance(val, str) and not val.startswith("-") and val != "N/A":
        try:
            float_val = float(val.replace("%", "").replace("+", ""))
            if float_val > 0:
                return "color: rgba(46, 204, 113, 0.9);"
        except ValueError:
            pass  # Not a numerical string, do nothing
    elif isinstance(val, (int, float)):
        if val < 0:
            return "color: rgba(255, 77, 77, 0.9);"
        elif val > 0:
            return "color: rgba(46, 204, 113, 0.9);"
    return ""  # Default or no specific color


# =============================================================================
# DATA FUNCTIONS (cached for performance)
# =============================================================================

# Load API keys
FMP_API_KEY = ["OoJcYpvMo94etCgLpr1s6TABcmhr7AWT"]
ALPHA_API_KEY = ["ZPODKN7Q87COJ0IR"]


@st.cache_data(ttl=60 * 60 * 24 * 30)
def get_company_info(symbol: str) -> dict:
    """Returns company information for the given stock symbol"""
    api_endpoint = f"https://financialmodelingprep.com/api/v3/profile/{symbol}/"
    params = {"apikey": FMP_API_KEY[0]}  # Access the key from the list
    try:
        response = requests.get(api_endpoint, params=params)
        response.raise_for_status()
        data = response.json()
        if not data:  # Check if the list is empty
            return None
        data = data[0]
        return {
            "Name": data.get("companyName"),
            "Exchange": data.get("exchangeShortName"),
            "Currency": data.get("currency"),
            "Country": data.get("country"),
            "Sector": data.get("sector"),
            "Market Cap": data.get("mktCap"),
            "Price": data.get("price"),
            "Beta": data.get("beta"),
            "Price change": data.get("changes"),
            "Website": data.get("website"),
            "Image": data.get("image"),
        }
    except requests.exceptions.RequestException as e:
        st.error(
            f"Network error or invalid API response fetching company info: {e}"
        )
        return None
    except (IndexError, KeyError) as e:
        st.error(
            f"Data parsing error for company info. Ticker might be invalid or data is missing: {e}"
        )
        return None
    except Exception as e:
        st.error(
            f"An unexpected error occurred while fetching company info: {e}"
        )
        return None


@st.cache_data(ttl=60 * 60 * 24 * 30)
def get_stock_price(symbol: str) -> pd.DataFrame:
    """Returns monthly stock prices for the last 5 years"""
    api_endpoint = "https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_MONTHLY_ADJUSTED",
        "symbol": symbol,
        "apikey": ALPHA_API_KEY[0],  # Access the key from the list
    }
    try:
        response = requests.get(api_endpoint, params=params)
        response.raise_for_status()
        data = response.json()
        if "Monthly Adjusted Time Series" not in data:
            st.warning(
                "Alpha Vantage API rate limit likely hit or no data for this symbol. Try again in a minute."
            )
            return None

        df = pd.DataFrame.from_dict(
            data["Monthly Adjusted Time Series"], orient="index"
        )
        df.index = pd.to_datetime(df.index)
        df = df[: 12 * 5]  # Get last 5 years (60 months)
        df = df[["4. close"]].astype(float)
        df = df.rename(columns={"4. close": "Price"})
        # Sort index in ascending order for plotting
        df = df.sort_index()
        return df
    except requests.exceptions.RequestException as e:
        st.error(
            f"Network error or invalid API response fetching stock price: {e}"
        )
        return None
    except KeyError as e:
        st.error(
            f"Data parsing error for stock price. Ticker might be invalid or data is missing: {e}"
        )
        return None
    except Exception as e:
        st.error(
            f"An unexpected error occurred while fetching stock price: {e}"
        )
        return None


@st.cache_data(ttl=60 * 60 * 24 * 30)
def get_income_statement(symbol: str) -> pd.DataFrame:
    """Retrieves income statement data"""
    api_endpoint = (
        f"https://financialmodelingprep.com/api/v3/income-statement/{symbol}/"
    )
    params = {"limit": 5, "apikey": FMP_API_KEY[0]}
    try:
        income_statement_data = []
        response = requests.get(api_endpoint, params=params)
        response.raise_for_status()
        response_data = response.json()
        if not response_data:
            return None
        for report in response_data:
            year = report.get("calendarYear")
            if year:  # Ensure year exists
                income_statement_data.append(
                    {
                        "Year": year,
                        "Revenue": report.get("revenue"),
                        "(-) Cost of Revenue": report.get("costOfRevenue"),
                        "= Gross Profit": report.get("grossProfit"),
                        "(-) Operating Expense": report.get("operatingExpenses"),
                        "= Operating Income": report.get("operatingIncome"),
                        "(+-) Other Income/Expenses": report.get(
                            "totalOtherIncomeExpensesNet"
                        ),
                        "= Income Before Tax": report.get("incomeBeforeTax"),
                        "(+-) Tax Income/Expense": report.get(
                            "incomeTaxExpense"
                        ),
                        "= Net Income": report.get("netIncome"),
                    }
                )
        # Sort by year in descending order for recent years first in table
        df = (
            pd.DataFrame(income_statement_data)
            .set_index("Year")
            .sort_index(ascending=False)
        )
        return df
    except requests.exceptions.RequestException as e:
        st.error(
            f"Network error or invalid API response fetching income statement: {e}"
        )
        return None
    except (KeyError, TypeError) as e:
        st.error(
            f"Data parsing error for income statement. Ticker might be invalid or data is missing: {e}"
        )
        return None
    except Exception as e:
        st.error(
            f"An unexpected error occurred while fetching income statement: {e}"
        )
        return None


@st.cache_data(ttl=60 * 60 * 24 * 30)
def get_balance_sheet(symbol: str) -> pd.DataFrame:
    """Retrieves balance sheet data"""
    api_endpoint = f"https://financialmodelingprep.com/api/v3/balance-sheet-statement/{symbol}"
    params = {"limit": 5, "apikey": FMP_API_KEY[0]}
    try:
        balance_sheet_data = []
        response = requests.get(api_endpoint, params=params)
        response.raise_for_status()
        response_data = response.json()
        if not response_data:
            return None
        for report in response_data:
            year = report.get("calendarYear")
            if year:
                balance_sheet_data.append(
                    {
                        "Year": year,
                        "Assets": report.get("totalAssets"),
                        "Current Assets": report.get("totalCurrentAssets"),
                        "Non-Current Assets": report.get(
                            "totalNonCurrentAssets"
                        ),
                        "Current Liabilities": report.get(
                            "totalCurrentLiabilities"
                        ),
                        "Non-Current Liabilities": report.get(
                            "totalNonCurrentLiabilities"
                        ),
                        "Liabilities": report.get("totalLiabilities"),
                        "Equity": report.get("totalEquity"),
                    }
                )
        df = (
            pd.DataFrame(balance_sheet_data)
            .set_index("Year")
            .sort_index(ascending=False)
        )
        return df
    except requests.exceptions.RequestException as e:
        st.error(
            f"Network error or invalid API response fetching balance sheet: {e}"
        )
        return None
    except (KeyError, TypeError) as e:
        st.error(
            f"Data parsing error for balance sheet. Ticker might be invalid or data is missing: {e}"
        )
        return None
    except Exception as e:
        st.error(
            f"An unexpected error occurred while fetching balance sheet: {e}"
        )
        return None


@st.cache_data(ttl=60 * 60 * 24 * 30)
def get_cash_flow(symbol: str) -> pd.DataFrame:
    """Retrieve cash flow data"""
    api_endpoint = f"https://financialmodelingprep.com/api/v3/cash-flow-statement/{symbol}"
    params = {"limit": 5, "apikey": FMP_API_KEY[0]}
    try:
        cashflow_data = []
        response = requests.get(api_endpoint, params=params)
        response.raise_for_status()
        response_data = response.json()
        if not response_data:
            return None
        for report in response_data:
            year = (
                report.get("date").split("-")[0] if report.get("date") else None
            )
            if year:
                cashflow_data.append(
                    {
                        "Year": year,
                        "Cash flows from operating activities": report.get(
                            "netCashProvidedByOperatingActivities"
                        ),
                        "Cash flows from investing activities": report.get(
                            "netCashUsedForInvestingActivites"
                        ),
                        "Cash flows from financing activities": report.get(
                            "netCashUsedProvidedByFinancingActivities"
                        ),
                        "Free cash flow": report.get("freeCashFlow"),
                    }
                )
        df = (
            pd.DataFrame(cashflow_data)
            .set_index("Year")
            .sort_index(ascending=False)
        )
        return df
    except requests.exceptions.RequestException as e:
        st.error(
            f"Network error or invalid API response fetching cash flow: {e}"
        )
        return None
    except (KeyError, TypeError) as e:
        st.error(
            f"Data parsing error for cash flow. Ticker might be invalid or data is missing: {e}"
        )
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred while fetching cash flow: {e}")
        return None


@st.cache_data(ttl=60 * 60 * 24 * 30)
def get_key_metrics(symbol: str) -> pd.DataFrame:
    """Returns key financial metrics"""
    api_endpoint = (
        f"https://financialmodelingprep.com/api/v3/key-metrics/{symbol}"
    )
    params = {"limit": 5, "apikey": FMP_API_KEY[0]}
    try:
        metrics_data = []
        response = requests.get(api_endpoint, params=params)
        response.raise_for_status()
        response_data = response.json()
        if not response_data:
            return None
        for report in response_data:
            year = (
                report.get("date").split("-")[0] if report.get("date") else None
            )
            if year:
                metrics_data.append(
                    {
                        "Year": year,
                        "Market Cap": report.get("marketCap"),
                        "Working Capital": report.get("workingCapital"),
                        "D/E ratio": report.get("debtToEquity"),
                        "P/E Ratio": report.get("peRatio"),
                        "ROE": report.get("roe"),
                        "Dividend Yield": report.get("dividendYield"),
                    }
                )
        df = (
            pd.DataFrame(metrics_data)
            .set_index("Year")
            .sort_index(ascending=False)
        )
        return df
    except requests.exceptions.RequestException as e:
        st.error(
            f"Network error or invalid API response fetching key metrics: {e}"
        )
        return None
    except (KeyError, TypeError) as e:
        st.error(
            f"Data parsing error for key metrics. Ticker might be invalid or data is missing: {e}"
        )
        return None
    except Exception as e:
        st.error(
            f"An unexpected error occurred while fetching key metrics: {e}"
        )
        return None


@st.cache_data(ttl=60 * 60 * 24 * 30)
def get_financial_ratios(symbol: str) -> pd.DataFrame:
    """Fetches financial ratios"""
    api_endpoint = f"https://financialmodelingprep.com/api/v3/ratios/{symbol}"
    params = {"limit": 5, "apikey": FMP_API_KEY[0]}
    try:
        ratios_data = []
        response = requests.get(api_endpoint, params=params)
        response.raise_for_status()
        response_data = response.json()
        if not response_data:
            return None
        for report in response_data:
            year = (
                report.get("date").split("-")[0] if report.get("date") else None
            )
            if year:
                ratios_data.append(
                    {
                        "Year": year,
                        "Current Ratio": report.get("currentRatio"),
                        "Quick Ratio": report.get("quickRatio"),
                        "Cash Ratio": report.get("cashRatio"),
                        "Days of Sales Outstanding": report.get(
                            "daysOfSalesOutstanding"
                        ),
                        "Days of Inventory Outstanding": report.get(
                            "daysOfInventoryOutstanding"
                        ),
                        "Operating Cycle": report.get("operatingCycle"),
                        "Days of Payables Outstanding": report.get(
                            "daysOfPayablesOutstanding"
                        ),
                        "Cash Conversion Cycle": report.get(
                            "cashConversionCycle"
                        ),
                        "Gross Profit Margin": report.get("grossProfitMargin"),
                        "Operating Profit Margin": report.get(
                            "operatingProfitMargin"
                        ),
                        "Pretax Profit Margin": report.get(
                            "pretaxProfitMargin"
                        ),
                        "Net Profit Margin": report.get("netProfitMargin"),
                        "Effective Tax Rate": report.get("effectiveTaxRate"),
                        "Return on Assets": report.get("returnOnAssets"),
                        "Return on Equity": report.get("returnOnEquity"),
                        "Return on Capital Employed": report.get(
                            "returnOnCapitalEmployed"
                        ),
                        "Net Income per EBT": report.get("netIncomePerEBT"),
                        "EBT per EBIT": report.get("ebtPerEbit"),
                        "EBIT per Revenue": report.get("ebitPerRevenue"),
                        "Debt Ratio": report.get("debtRatio"),
                        "Debt Equity Ratio": report.get("debtEquityRatio"),
                        "Long-term Debt to Capitalization": report.get(
                            "longTermDebtToCapitalization"
                        ),
                        "Total Debt to Capitalization": report.get(
                            "totalDebtToCapitalization"
                        ),
                        "Interest Coverage": report.get("interestCoverage"),
                        "Cash Flow to Debt Ratio": report.get(
                            "cashFlowToDebtRatio"
                        ),
                        "Company Equity Multiplier": report.get(
                            "companyEquityMultiplier"
                        ),
                        "Receivables Turnover": report.get(
                            "receivablesTurnover"
                        ),
                        "Payables Turnover": report.get("payablesTurnover"),
                        "Inventory Turnover": report.get("inventoryTurnover"),
                        "Fixed Asset Turnover": report.get(
                            "fixedAssetTurnover"
                        ),
                        "Asset Turnover": report.get("assetTurnover"),
                        "Operating Cash Flow per Share": report.get(
                            "operatingCashFlowPerShare"
                        ),
                        "Free Cash Flow per Share": report.get(
                            "freeCashFlowPerShare"
                        ),
                        "Cash per Share": report.get("cashPerShare"),
                        "Payout Ratio": report.get("payoutRatio"),
                        "Operating Cash Flow Sales Ratio": report.get(
                            "operatingCashFlowSalesRatio"
                        ),
                        "Free Cash Flow Operating Cash Flow Ratio": report.get(
                            "freeCashFlowOperatingCashFlowRatio"
                        ),
                        "Cash Flow Coverage Ratios": report.get(
                            "cashFlowCoverageRatios"
                        ),
                        "Price to Book Value Ratio": report.get(
                            "priceToBookRatio"
                        ),
                        "Price to Earnings Ratio": report.get(
                            "priceEarningsRatio"
                        ),
                        "Price to Sales Ratio": report.get(
                            "priceToSalesRatio"
                        ),
                        "Dividend Yield": report.get("dividendYield"),
                        "Enterprise Value to EBITDA": report.get(
                            "enterpriseValueMultiple"
                        ),
                        "Price to Fair Value": report.get("priceFairValue"),
                    }
                )
        df = (
            pd.DataFrame(ratios_data)
            .set_index("Year")
            .sort_index(ascending=False)
        )
        return df
    except requests.exceptions.RequestException as e:
        st.error(
            f"Network error or invalid API response fetching financial ratios: {e}"
        )
        return None
    except (KeyError, TypeError) as e:
        st.error(
            f"Data parsing error for financial ratios. Ticker might be invalid or data is missing: {e}"
        )
        return None
    except Exception as e:
        st.error(
            f"An unexpected error occurred while fetching financial ratios: {e}"
        )
        return None


def main():
    """Main application function"""
    # Apply modern CSS styling
    apply_modern_css()

    # Header
    st.markdown(
        """
    <div class="main-header">
        <h1>📈 Financial Dashboard</h1>
        <p style="text-align: center; color: #666666; font-size: 1.1rem;">
            Professional financial analysis with real-time data
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Initialize session state
    if "btn_clicked" not in st.session_state:
        st.session_state["btn_clicked"] = False

    def callback():
        st.session_state["btn_clicked"] = True

    # Input section
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        symbol_input = st.text_input(
            "",
            placeholder="Enter stock ticker (e.g., AAPL, TSLA, MSFT)",
            help="Enter a valid stock symbol to analyze",
        ).upper()

        if st.button(
            "🔍 Analyze Stock", on_click=callback, use_container_width=True
        ):
            pass

    # Main dashboard
    if st.session_state["btn_clicked"]:
        if not symbol_input:
            st.warning("⚠️ Please enter a stock ticker symbol.")
            return

        try:
            # Fetch data with loading indicators
            with st.spinner("📊 Fetching financial data..."):
                company_data = get_company_info(symbol_input)
                if company_data is None:
                    st.error(
                        "❌ Failed to retrieve company data. Please check the ticker symbol."
                    )
                    return

                metrics_data = get_key_metrics(symbol_input)
                income_data = get_income_statement(symbol_input)
                performance_data = get_stock_price(symbol_input)
                ratios_data = get_financial_ratios(symbol_input)
                balance_sheet_data = get_balance_sheet(symbol_input)
                cashflow_data = get_cash_flow(symbol_input)

            # --- Check if essential data is available before proceeding ---
            if metrics_data is None or metrics_data.empty:
                st.warning(
                    "Could not retrieve key metrics for this symbol. Some dashboard elements might be empty."
                )
            if income_data is None or income_data.empty:
                st.warning(
                    "Could not retrieve income statement for this symbol."
                )
            if performance_data is None or performance_data.empty:
                st.warning(
                    "Could not retrieve historical stock prices for this symbol. Chart will not be displayed."
                )
            if ratios_data is None or ratios_data.empty:
                st.warning("Could not retrieve financial ratios for this symbol.")
            if balance_sheet_data is None or balance_sheet_data.empty:
                st.warning("Could not retrieve balance sheet for this symbol.")
            if cashflow_data is None or cashflow_data.empty:
                st.warning(
                    "Could not retrieve cash flow statement for this symbol."
                )

            # Company Information Header
            st.markdown("### 🏢 Company Overview")

            image_url = company_data.get("Image")
            website_url = company_data.get("Website")
            logo_html = ""
            if image_url:
                logo_html = f"""
                <div class="company-logo">
                    <a href="{website_url if website_url else '#'}" target="_blank">
                        <img src="{image_url}" alt="{company_data.get('Name', 'Logo')}">
                    </a>
                </div>
                """

            company_info_html = f"""
            <div class="company-info">
                <div>
                    <div class="company-name">{company_data.get('Name', 'N/A')}</div>
                    <div style="color: var(--text-secondary);">
                        {company_data.get('Sector', 'N/A')} • {company_data.get('Exchange', 'N/A')} • {company_data.get('Country', 'N/A')}
                    </div>
                </div>
                {logo_html}
            </div>
            """
            st.markdown(company_info_html, unsafe_allow_html=True)

            # Key Metrics Row
            st.markdown("### 📊 Key Metrics")
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "💰 Stock Price",
                    f"${company_data.get('Price', 0.0):.2f}",
                    f"{company_data.get('Price change', 0.0):.2f}",
                )

            with col2:
                if (
                    metrics_data is not None
                    and not metrics_data.empty
                    and "Market Cap" in metrics_data.columns
                ):
                    st.metric(
                        "🏦 Market Cap",
                        millify(metrics_data["Market Cap"][0], precision=2),
                        get_delta(metrics_data, "Market Cap"),
                    )
                else:
                    st.metric("🏦 Market Cap", "N/A", "N/A")

            with col3:
                if (
                    metrics_data is not None
                    and not metrics_data.empty
                    and "Working Capital" in metrics_data.columns
                ):
                    st.metric(
                        "💼 Working Capital",
                        millify(
                            metrics_data["Working Capital"][0], precision=2
                        ),
                        get_delta(metrics_data, "Working Capital"),
                    )
                else:
                    st.metric("💼 Working Capital", "N/A", "N/A")

            with col4:
                if (
                    metrics_data is not None
                    and not metrics_data.empty
                    and "P/E Ratio" in metrics_data.columns
                ):
                    st.metric(
                        "📈 P/E Ratio",
                        f"{metrics_data['P/E Ratio'][0]:.2f}",
                        get_delta(metrics_data, "P/E Ratio"),
                    )
                else:
                    st.metric("📈 P/E Ratio", "N/A", "N/A")

            # Financial Performance Chart
            st.markdown("### 💹 Stock Performance (5-Year Trend)")
            with st.container():
                st.markdown(
                    '<div class="chart-container">', unsafe_allow_html=True
                )
                if performance_data is not None and not performance_data.empty:
                    fig = go.Figure()
                    fig.add_trace(
                        go.Scatter(
                            x=performance_data.index,
                            y=performance_data["Price"],
                            mode="lines",
                            name="Stock Price",
                            line=dict(color="#667eea", width=3),
                            fill="tozeroy",
                            fillcolor="rgba(102, 126, 234, 0.1)",
                        )
                    )
                    fig.update_layout(
                        title="Monthly Adjusted Close Price",
                        xaxis_title="Date",
                        yaxis_title=f"Price ({company_data.get('Currency', 'USD')})",
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(
                            family="Inter, sans-serif"
                        ),  # Let CSS handle color
                        xaxis=dict(gridcolor="rgba(102, 126, 234, 0.2)"),
                        yaxis=dict(gridcolor="rgba(102, 126, 234, 0.2)"),
                        hovermode="x unified",
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info(
                        "No historical stock price data available for charting."
                    )
                st.markdown("</div>", unsafe_allow_html=True)

            # Financial Statements
            st.markdown("### 📄 Financial Statements")

            def format_statement(df):
                """Formats financial statement DataFrame for display"""
                if df is None or df.empty:
                    return pd.DataFrame().T  # Return an empty styled dataframe
                formatted_df = df.T.applymap(
                    lambda x: millify(x, precision=2)
                    if isinstance(x, (int, float)) and abs(x) >= 1000
                    else f"{x:,.2f}"
                    if isinstance(x, (int, float))
                    else x
                )
                return formatted_df.style.applymap(color_highlighter)

            def to_csv(df):
                """Converts DataFrame to CSV for download"""
                if df is None:
                    return b""  # Return empty bytes if df is None
                output = BytesIO()
                df.to_csv(output, index=True, encoding="utf-8")
                return output.getvalue()

            tab1, tab2, tab3 = st.tabs(
                ["Income Statement", "Balance Sheet", "Cash Flow"]
            )

            with tab1:
                st.markdown(
                    create_info_card("Income Statement"),
                    unsafe_allow_html=True,
                )
                if income_data is not None and not income_data.empty:
                    st.dataframe(
                        format_statement(income_data), use_container_width=True
                    )
                    st.download_button(
                        label="📥 Download as CSV",
                        data=to_csv(income_data),
                        file_name=f"{symbol_input}_income_statement.csv",
                        mime="text/csv",
                    )
                else:
                    st.info("Income Statement data not available.")

            with tab2:
                st.markdown(
                    create_info_card("Balance Sheet"), unsafe_allow_html=True
                )
                if balance_sheet_data is not None and not balance_sheet_data.empty:
                    st.dataframe(
                        format_statement(balance_sheet_data),
                        use_container_width=True,
                    )
                    st.download_button(
                        label="📥 Download as CSV",
                        data=to_csv(balance_sheet_data),
                        file_name=f"{symbol_input}_balance_sheet.csv",
                        mime="text/csv",
                    )
                else:
                    st.info("Balance Sheet data not available.")

            with tab3:
                st.markdown(
                    create_info_card("Cash Flow Statement"),
                    unsafe_allow_html=True,
                )
                if cashflow_data is not None and not cashflow_data.empty:
                    st.dataframe(
                        format_statement(cashflow_data),
                        use_container_width=True,
                    )
                    st.download_button(
                        label="📥 Download as CSV",
                        data=to_csv(cashflow_data),
                        file_name=f"{symbol_input}_cash_flow.csv",
                        mime="text/csv",
                    )
                else:
                    st.info("Cash Flow Statement data not available.")

            # Financial Ratios
            st.markdown("### 🧮 Financial Ratios Analysis")

            if ratios_data is not None and not ratios_data.empty:
                ratio_categories = {
                    "Profitability": [
                        "Gross Profit Margin",
                        "Operating Profit Margin",
                        "Net Profit Margin",
                        "Return on Assets",
                        "Return on Equity",
                    ],
                    "Liquidity": ["Current Ratio", "Quick Ratio", "Cash Ratio"],
                    "Solvency": [
                        "Debt Ratio",
                        "Debt Equity Ratio",
                        "Interest Coverage",
                    ],
                    "Efficiency": [
                        "Asset Turnover",
                        "Inventory Turnover",
                        "Operating Cycle",
                        "Cash Conversion Cycle",
                    ],
                    "Valuation": [
                        "Price to Earnings Ratio",
                        "Price to Book Value Ratio",
                        "Price to Sales Ratio",
                        "Dividend Yield",
                    ],
                }

                # Filter categories to only include ratios present in the fetched data
                available_categories = {}
                for category, ratios in ratio_categories.items():
                    available_ratios = [
                        r for r in ratios if r in ratios_data.columns
                    ]
                    if available_ratios:
                        available_categories[category] = available_ratios

                if not available_categories:
                    st.info(
                        "No financial ratios available for display under any category."
                    )
                else:
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        selected_category = st.selectbox(
                            "Select Ratio Category",
                            options=list(available_categories.keys()),
                            index=0,
                            label_visibility="collapsed",
                        )

                    st.markdown(f"#### {selected_category} Ratios")
                    filtered_ratios = ratios_data[
                        available_categories[selected_category]
                    ].T
                    st.dataframe(
                        filtered_ratios.style.format("{:.2f}").applymap(
                            color_highlighter
                        ),
                        use_container_width=True,
                    )

                    st.download_button(
                        label="📥 Download All Ratios as CSV",
                        data=to_csv(ratios_data),
                        file_name=f"{symbol_input}_financial_ratios.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
            else:
                st.info("Financial Ratios data not available for this symbol.")

        except Exception as e:
            st.error(
                f"An unexpected error occurred during dashboard generation: {e}"
            )
            st.warning("Please verify the ticker symbol or try again later.")

    # Custom Footer
    st.markdown(
        """
    <div class="custom-footer">
        No me mates Vélez
    </div>
    """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
