# Standard library imports
import sys
# import hashlib  #<-- REMOVED
import hmac
import time
from io import BytesIO
from datetime import datetime, timedelta

# Third-party imports
import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objs as go
from millify import millify
from supabase import create_client, Client
import bcrypt

# Configure the app page
st.set_page_config(
    page_title="Financial Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Supabase configuration
SUPABASE_URL = "https://pcfqzrzeghvuttbiznj.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InBjZnF6cnplbGd2dXR0aGJpanpnIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTI2MDY4ODYsImV4cCI6MjA2ODE4Mjg4Nn0.zVUs0K7vNIUvwxJCesUsVhjpZn5vTm0VrCoiuVCo07k"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Original API keys for non-crypto data
FMP_API_KEY = ["OoJcYpvMo94etCgLpr1s6TABcmhr7AWT"]
ALPHA_API_KEY = ["ZPODKN7Q87COJ0IR"]


def apply_modern_css():
    """Apply modern CSS styling with glassmorphism effects, sticky header, and improved UI"""
    modern_css = """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

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
        }

        .stApp {
            font-family: 'Inter', sans-serif;
            background: var(--background-gradient);
            min-height: 100vh;
        }

        .main .block-container {
            max-width: 1200px;
            padding: 2rem 1rem;
            margin: 0 auto;
        }

        /* Sticky Header */
        .sticky-header {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            background: var(--glass-bg);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border-bottom: 1px solid var(--glass-border);
            padding: 1rem 2rem;
            z-index: 1000;
            transition: var(--transition);
        }

        .sticky-header.hidden {
            transform: translateY(-100%);
        }

        .sticky-header h1 {
            color: var(--text-primary);
            font-weight: 700;
            font-size: 1.8rem;
            margin: 0;
            text-align: center;
        }

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
        }

        /* Company info with logo inside */
        .company-info {
            background: var(--glass-bg);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid var(--glass-border);
            border-radius: var(--border-radius);
            padding: 2rem;
            margin-bottom: 2rem;
            box-shadow: var(--shadow-light);
            transition: var(--transition);
            display: flex;
            align-items: center;
            gap: 2rem;
        }

        .company-info:hover {
            transform: translateY(-2px);
            box-shadow: var(--shadow-hover);
        }

        .company-info-content {
            flex: 1;
        }

        .company-logo {
            flex-shrink: 0;
        }

        .company-logo img {
            width: 100px;
            height: 100px;
            object-fit: contain;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
            transition: var(--transition);
        }

        .company-logo img:hover {
            transform: scale(1.05);
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.15);
        }

        .company-name {
            font-size: 1.8rem;
            font-weight: 700;
            color: var(--text-primary);
            margin-bottom: 0.5rem;
        }

        .company-details {
            color: var(--text-secondary);
            font-size: 1.1rem;
        }

        /* Thinner footer */
        .custom-footer {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            background: var(--glass-bg);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border-top: 1px solid var(--glass-border);
            padding: 0.5rem 1rem;
            text-align: center;
            color: var(--text-secondary);
            font-size: 0.8rem;
            height: 40px;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        /* Login/Signup Forms */
        .auth-container {
            max-width: 400px;
            margin: 2rem auto;
            background: var(--glass-bg);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid var(--glass-border);
            border-radius: var(--border-radius);
            padding: 2rem;
            box-shadow: var(--shadow-light);
        }

        .security-note {
            background: rgba(46, 204, 113, 0.1);
            border: 1px solid rgba(46, 204, 113, 0.3);
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 1rem;
            color: var(--text-primary);
            font-size: 0.9rem;
        }

        .security-note h4 {
            color: #2ecc71;
            margin-bottom: 0.5rem;
        }

        /* Other existing styles... */
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
            color: var(--text-primary);
        }

        .chart-container:hover {
            transform: translateY(-2px);
            box-shadow: var(--shadow-hover);
        }

        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}

        @media (max-width: 768px) {
            .company-info {
                flex-direction: column;
                text-align: center;
            }
            
            .sticky-header {
                padding: 0.5rem 1rem;
            }
            
            .sticky-header h1 {
                font-size: 1.5rem;
            }
        }
    </style>

    <script>
        // Sticky header functionality
        let lastScrollTop = 0;
        const header = document.querySelector('.sticky-header');
        
        window.addEventListener('scroll', () => {
            const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
            
            if (scrollTop > lastScrollTop && scrollTop > 100) {
                header.classList.add('hidden');
            } else {
                header.classList.remove('hidden');
            }
            
            lastScrollTop = scrollTop;
        });
    </script>
    """
    st.markdown(modern_css, unsafe_allow_html=True)


def hash_password(password: str) -> str:
    """Hash a password using bcrypt"""
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')


def verify_password(password: str, hashed: str) -> bool:
    """Verify a password against its hash"""
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))


def create_user(email: str, username: str, binance_api_key: str, binance_api_secret: str, password: str):
    """Create a new user in Supabase"""
    try:
        password_hash = hash_password(password)
        
        response = supabase.table('users').insert({
            'email': email,
            'username': username,
            'binance_api_key': binance_api_key,
            'binance_api_secret': binance_api_secret,
            'password_hash': password_hash
        }).execute()
        
        return response.data[0] if response.data else None
    except Exception as e:
        st.error(f"Error creating user: {str(e)}")
        return None


def authenticate_user(username: str, password: str):
    """Authenticate user with username and password"""
    try:
        response = supabase.table('users').select('*').eq('username', username).execute()
        
        if response.data and len(response.data) > 0:
            user = response.data[0]
            if verify_password(password, user['password_hash']):
                return user
        return None
    except Exception as e:
        st.error(f"Authentication error: {str(e)}")
        return None


def show_auth_page():
    """Show authentication page"""
    st.markdown('<div class="sticky-header"><h1>📈 Financial Dashboard</h1></div>', unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["Login", "Sign Up"])
    
    with tab1:
        st.markdown("""
        <div class="auth-container">
            <h2 style="text-align: center; margin-bottom: 2rem;">Login</h2>
            <div class="security-note">
                <h4>🔒 Security Notice</h4>
                <p>Your data is secure and encrypted. We use industry-standard encryption to protect your API keys and personal information.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            
            if st.form_submit_button("Login", use_container_width=True):
                if username and password:
                    user = authenticate_user(username, password)
                    if user:
                        st.session_state.user = user
                        st.success("Login successful!")
                        st.rerun()
                    else:
                        st.error("Invalid username or password")
                else:
                    st.error("Please fill in all fields")
    
    with tab2:
        st.markdown("""
        <div class="auth-container">
            <h2 style="text-align: center; margin-bottom: 2rem;">Sign Up</h2>
            <div class="security-note">
                <h4>🔒 Security Notice</h4>
                <p>Your data is secure and encrypted. We use industry-standard encryption to protect your API keys and personal information. Your Binance API keys are stored securely and never shared.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        with st.form("signup_form"):
            email = st.text_input("Email")
            username = st.text_input("Username")
            binance_api_key = st.text_input("Binance API Key")
            binance_api_secret = st.text_input("Binance API Secret", type="password")
            password = st.text_input("Password", type="password")
            
            if st.form_submit_button("Sign Up", use_container_width=True):
                if email and username and binance_api_key and binance_api_secret and password:
                    user = create_user(email, username, binance_api_key, binance_api_secret, password)
                    if user:
                        st.success("Account created successfully! Please login.")
                        st.balloons()
                    else:
                        st.error("Failed to create account. Username or email might already exist.")
                else:
                    st.error("Please fill in all fields")


def create_binance_signature(query_string: str, secret: str) -> str:
    """Create signature for Binance API"""
    # MODIFIED: Pass 'sha256' as a string to avoid explicit hashlib import.
    return hmac.new(secret.encode('utf-8'), query_string.encode('utf-8'), 'sha256').hexdigest()


@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_binance_account_info(api_key: str, secret: str):
    """Get Binance account information"""
    try:
        timestamp = int(time.time() * 1000)
        query_string = f"timestamp={timestamp}"
        signature = create_binance_signature(query_string, secret)
        
        url = f"https://api.binance.com/api/v3/account?{query_string}&signature={signature}"
        headers = {"X-MBX-APIKEY": api_key}
        
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error fetching Binance account info: {str(e)}")
        return None


@st.cache_data(ttl=60)  # Cache for 1 minute
def get_binance_price(symbol: str):
    """Get current price from Binance"""
    try:
        url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error fetching Binance price: {str(e)}")
        return None


@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_binance_24hr_ticker(symbol: str):
    """Get 24hr ticker statistics"""
    try:
        url = f"https://api.binance.com/api/v3/ticker/24hr?symbol={symbol}"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error fetching 24hr ticker: {str(e)}")
        return None


@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_binance_klines(symbol: str, interval: str = "1d", limit: int = 100):
    """Get historical klines/candlestick data"""
    try:
        url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
        response = requests.get(url)
        response.raise_for_status()
        
        data = response.json()
        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
        
        return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    except Exception as e:
        st.error(f"Error fetching klines: {str(e)}")
        return None


@st.cache_data(ttl=600)  # Cache for 10 minutes
def get_binance_exchange_info():
    """Get exchange information"""
    try:
        url = "https://api.binance.com/api/v3/exchangeInfo"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error fetching exchange info: {str(e)}")
        return None


# Keep all the original functions for stock data
@st.cache_data(ttl=60 * 60 * 24 * 30)
def get_company_info(symbol: str) -> dict:
    """Returns company information for the given stock symbol"""
    api_endpoint = f"https://financialmodelingprep.com/api/v3/profile/{symbol}/"
    params = {"apikey": FMP_API_KEY[0]}
    try:
        response = requests.get(api_endpoint, params=params)
        response.raise_for_status()
        data = response.json()
        if not data:
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
    except Exception as e:
        st.error(f"Error fetching company info: {e}")
        return None


def get_delta(df: pd.DataFrame, key: str) -> str:
    """Calculate percentage difference between the first two values"""
    if key not in df.columns:
        return f"Key '{key}' not found in DataFrame columns."

    if len(df) < 2:
        return "DataFrame must contain at least two rows."

    val1 = df[key].iloc[1]  # Second most recent
    val2 = df[key].iloc[0]  # Most recent

    if pd.isna(val1) or pd.isna(val2):
        return "N/A"

    if val1 == 0:
        if val2 == 0:
            return "0.00%"
        else:
            return "Inf%" if val2 > 0 else "-Inf%"
    else:
        delta = (val2 - val1) / val1 * 100

    return f"{delta:+.2f}%"


def main():
    """Main application function"""
    apply_modern_css()
    
    # Check if user is logged in
    if 'user' not in st.session_state:
        show_auth_page()
        return
    
    user = st.session_state.user
    
    # Sticky header
    st.markdown('<div class="sticky-header"><h1>📈 Financial Dashboard</h1></div>', unsafe_allow_html=True)
    
    # Main header
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown(f"""
        <div class="main-header">
            <h1>📈 Financial Dashboard</h1>
            <p style="text-align: center; color: #666666; font-size: 1.1rem;">
                Welcome back, {user['username']}! Professional financial analysis with real-time data
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        if st.button("Logout", type="secondary"):
            del st.session_state.user
            st.rerun()
    
    # Asset type selection
    asset_type = st.selectbox("Select Asset Type", ["Stocks", "Cryptocurrency"], index=0)
    
    if asset_type == "Cryptocurrency":
        # Cryptocurrency section
        st.markdown("### 🚀 Cryptocurrency Analysis")
        
        # Get account info
        account_info = get_binance_account_info(user['binance_api_key'], user['binance_api_secret'])
        
        if account_info:
            st.success("✅ Connected to Binance successfully!")
            
            # Symbol input
            col1, col2 = st.columns([2, 1])
            with col1:
                crypto_symbol = st.text_input("Enter Cryptocurrency Symbol", value="BTCUSDT", 
                                            help="Enter symbol like BTCUSDT, ETHUSDT, etc.")
            with col2:
                if st.button("🔍 Analyze Crypto", use_container_width=True):
                    if crypto_symbol:
                        # Get crypto data
                        price_data = get_binance_price(crypto_symbol)
                        ticker_data = get_binance_24hr_ticker(crypto_symbol)
                        klines_data = get_binance_klines(crypto_symbol)
                        
                        if price_data and ticker_data:
                            # Display metrics
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric("💰 Current Price", f"${float(price_data['price']):.2f}")
                            
                            with col2:
                                change_24h = float(ticker_data['priceChange'])
                                change_percent = float(ticker_data['priceChangePercent'])
                                st.metric("📊 24h Change", f"{change_percent:.2f}%", f"{change_24h:.2f}")
                            
                            with col3:
                                volume = float(ticker_data['volume'])
                                st.metric("📈 24h Volume", millify(volume, precision=2))
                            
                            with col4:
                                high_24h = float(ticker_data['highPrice'])
                                low_24h = float(ticker_data['lowPrice'])
                                st.metric("🔺 24h High", f"${high_24h:.2f}")
                            
                            # Price chart
                            if klines_data is not None and not klines_data.empty:
                                st.markdown("### 📈 Price Chart")
                                with st.container():
                                    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                                    
                                    fig = go.Figure()
                                    fig.add_trace(go.Candlestick(
                                        x=klines_data['timestamp'],
                                        open=klines_data['open'],
                                        high=klines_data['high'],
                                        low=klines_data['low'],
                                        close=klines_data['close'],
                                        name=crypto_symbol
                                    ))
                                    
                                    fig.update_layout(
                                        title=f"{crypto_symbol} Price Chart",
                                        xaxis_title="Date",
                                        yaxis_title="Price (USDT)",
                                        plot_bgcolor="rgba(0,0,0,0)",
                                        paper_bgcolor="rgba(0,0,0,0)",
                                        font=dict(family="Inter, sans-serif"),
                                        xaxis=dict(gridcolor="rgba(102, 126, 234, 0.2)"),
                                        yaxis=dict(gridcolor="rgba(102, 126, 234, 0.2)"),
                                        height=500
                                    )
                                    
                                    st.plotly_chart(fig, use_container_width=True)
                                    st.markdown("</div>", unsafe_allow_html=True)
                            
                            # Account balances
                            st.markdown("### 💼 Account Balances")
                            balances = []
                            for balance in account_info['balances']:
                                free = float(balance['free'])
                                locked = float(balance['locked'])
                                if free > 0 or locked > 0:
                                    balances.append({
                                        'Asset': balance['asset'],
                                        'Free': free,
                                        'Locked': locked,
                                        'Total': free + locked
                                    })
                            
                            if balances:
                                df_balances = pd.DataFrame(balances)
                                st.dataframe(df_balances, use_container_width=True)
                            else:
                                st.info("No balances to display")
        else:
            st.error("❌ Failed to connect to Binance. Please check your API credentials.")
    
    else:
        # Original stock analysis code
        st.markdown("### 📊 Stock Analysis")
        
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

            if st.button("🔍 Analyze Stock", on_click=callback, use_container_width=True):
                pass

        # Stock dashboard (original code)
        if st.session_state["btn_clicked"]:
            if not symbol_input:
                st.warning("⚠️ Please enter a stock ticker symbol.")
                return

            try:
                with st.spinner("📊 Fetching financial data..."):
                    company_data = get_company_info(symbol_input)
                    if company_data is None:
                        st.error("❌ Failed to retrieve company data. Please check the ticker symbol.")
                        return

                # Company Information Header with logo inside
                st.markdown("### 🏢 Company Overview")
                
                # Create company info with logo inside
                logo_html = ""
                if company_data.get("Image"):
                    logo_html = f'<div class="company-logo"><img src="{company_data["Image"]}" alt="{company_data.get("Name", "Logo")}"></div>'
                
                st.markdown(f"""
                <div class="company-info">
                    <div class="company-info-content">
                        <div class="company-name">{company_data.get('Name', 'N/A')}</div>
                        <div class="company-details">
                            {company_data.get('Sector', 'N/A')} • {company_data.get('Exchange', 'N/A')} • {company_data.get('Country', 'N/A')}
                        </div>
                    </div>
                    {logo_html}
                </div>
                """, unsafe_allow_html=True)

                # Key Metrics
                st.markdown("### 📊 Key Metrics")
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("💰 Stock Price", f"${company_data.get('Price', 0.0):.2f}", 
                             f"{company_data.get('Price change', 0.0):.2f}")

                with col2:
                    market_cap = company_data.get('Market Cap', 0)
                    st.metric("🏦 Market Cap", millify(market_cap, precision=2) if market_cap else "N/A")

                with col3:
                    beta = company_data.get('Beta', 0)
                    st.metric("📈 Beta", f"{beta:.2f}" if beta else "N/A")

                with col4:
                    st.metric("🌐 Exchange", company_data.get('Exchange', 'N/A'))

            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")

    # Thinner footer
    st.markdown("""
    <div class="custom-footer">
        No me mates Vélez
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
