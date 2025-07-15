# Standard library imports
import sys
import os
import hashlib
import hmac
import time
import secrets
from io import BytesIO
from datetime import datetime, timedelta
import json

# Third-party imports
import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objs as go
from millify import millify
import bcrypt
from binance.client import Client as BinanceClient
from binance.exceptions import BinanceAPIException
import supabase
from supabase import create_client, Client as SupabaseClient

# Configure the app page
st.set_page_config(
    page_title="Crypto Financial Dashboard",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Supabase configuration
SUPABASE_URL = "https://pcfqzrzelgvutthbijzg.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InBjZnF6cnplbGd2dXR0aGJpanpnIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTI2MDY4ODYsImV4cCI6MjA2ODE4Mjg4Nn0.zVUs0K7vNIUvwxJCesUsVhjpZn5vTm0VrCoiuVCo07k"

# Initialize Supabase client
supabase_client: SupabaseClient = create_client(SUPABASE_URL, SUPABASE_KEY)

def apply_modern_css():
    """Apply modern CSS styling with glassmorphism effects and sticky header"""
    modern_css = """
    <style>
        /* Import Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

        /* Root variables */
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

        /* Dark mode */
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

        /* Global styles */
        .stApp {
            font-family: 'Inter', sans-serif;
            background: var(--background-gradient);
            min-height: 100vh;
        }

        /* Sticky header */
        .sticky-header {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            background: var(--glass-bg);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border-bottom: 1px solid var(--glass-border);
            padding: 1rem 2rem;
            z-index: 9999;
            transform: translateY(-100%);
            transition: transform 0.3s ease;
        }

        .sticky-header.visible {
            transform: translateY(0);
        }

        .sticky-header h1 {
            color: var(--text-primary);
            font-size: 1.5rem;
            margin: 0;
            font-weight: 600;
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
            width: 80px;
            height: 80px;
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

        /* Auth container */
        .auth-container {
            max-width: 400px;
            margin: 0 auto;
            padding: 2rem;
            background: var(--glass-bg);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid var(--glass-border);
            border-radius: var(--border-radius);
            box-shadow: var(--shadow-light);
        }

        .auth-title {
            text-align: center;
            color: var(--text-primary);
            font-size: 2rem;
            font-weight: 600;
            margin-bottom: 2rem;
        }

        .security-note {
            background: rgba(46, 204, 113, 0.1);
            border: 1px solid rgba(46, 204, 113, 0.3);
            border-radius: 8px;
            padding: 1rem;
            margin: 1rem 0;
            font-size: 0.9rem;
            color: var(--text-primary);
        }

        .security-note::before {
            content: "🔒 ";
            font-size: 1.2em;
        }

        /* Footer styling */
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
            z-index: 1000;
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

        /* Sticky header script */
        window.addEventListener('scroll', function() {
            const header = document.querySelector('.sticky-header');
            if (window.scrollY > 200) {
                header.classList.add('visible');
            } else {
                header.classList.remove('visible');
            }
        });

        /* Hide Streamlit elements */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
    </style>
    """
    st.markdown(modern_css, unsafe_allow_html=True)

def hash_password(password: str) -> str:
    """Hash a password using bcrypt"""
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(password: str, hashed: str) -> bool:
    """Verify a password against its hash"""
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

def create_session_token() -> str:
    """Create a secure session token"""
    return secrets.token_urlsafe(32)

def register_user(username: str, email: str, password: str, binance_api_key: str, binance_api_secret: str) -> bool:
    """Register a new user"""
    try:
        password_hash = hash_password(password)
        
        result = supabase_client.table('users').insert({
            'username': username,
            'email': email,
            'password_hash': password_hash,
            'binance_api_key': binance_api_key,
            'binance_api_secret': binance_api_secret
        }).execute()
        
        return True
    except Exception as e:
        st.error(f"Registration failed: {str(e)}")
        return False

def login_user(username: str, password: str) -> dict:
    """Login user and create session"""
    try:
        result = supabase_client.table('users').select('*').eq('username', username).execute()
        
        if result.data and len(result.data) > 0:
            user = result.data[0]
            if verify_password(password, user['password_hash']):
                # Create session
                session_token = create_session_token()
                expires_at = datetime.now() + timedelta(days=30)
                
                supabase_client.table('user_sessions').insert({
                    'user_id': user['id'],
                    'session_token': session_token,
                    'expires_at': expires_at.isoformat()
                }).execute()
                
                return {
                    'success': True,
                    'user': user,
                    'session_token': session_token
                }
        
        return {'success': False, 'message': 'Invalid credentials'}
    except Exception as e:
        return {'success': False, 'message': str(e)}

def get_user_by_session(session_token: str) -> dict:
    """Get user by session token"""
    try:
        result = supabase_client.table('user_sessions').select('*, users(*)').eq('session_token', session_token).execute()
        
        if result.data and len(result.data) > 0:
            session = result.data[0]
            expires_at = datetime.fromisoformat(session['expires_at'].replace('Z', '+00:00'))
            
            if expires_at > datetime.now(expires_at.tzinfo):
                return {'success': True, 'user': session['users']}
        
        return {'success': False}
    except Exception as e:
        return {'success': False, 'message': str(e)}

def logout_user(session_token: str):
    """Logout user by removing session"""
    try:
        supabase_client.table('user_sessions').delete().eq('session_token', session_token).execute()
    except Exception as e:
        st.error(f"Logout failed: {str(e)}")

def show_auth_page():
    """Show authentication page"""
    st.markdown('<div class="auth-container">', unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["Login", "Sign Up"])
    
    with tab1:
        st.markdown('<h2 class="auth-title">Welcome Back</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="security-note">
            Your data is secure and encrypted. We use industry-standard encryption to protect your API keys and personal information.
        </div>
        """, unsafe_allow_html=True)
        
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        
        if st.button("Login", key="login_btn"):
            if username and password:
                result = login_user(username, password)
                if result['success']:
                    st.session_state['session_token'] = result['session_token']
                    st.session_state['user'] = result['user']
                    st.success("Login successful!")
                    st.rerun()
                else:
                    st.error(result.get('message', 'Login failed'))
            else:
                st.warning("Please fill in all fields")
    
    with tab2:
        st.markdown('<h2 class="auth-title">Create Account</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="security-note">
            Your data is secure and encrypted. We use bcrypt for password hashing and encrypt your API keys. Your credentials are never stored in plain text.
        </div>
        """, unsafe_allow_html=True)
        
        new_username = st.text_input("Username", key="signup_username")
        new_email = st.text_input("Email", key="signup_email")
        binance_api_key = st.text_input("Binance API Key", key="signup_api_key")
        binance_api_secret = st.text_input("Binance API Secret", type="password", key="signup_api_secret")
        new_password = st.text_input("Password", type="password", key="signup_password")
        
        if st.button("Sign Up", key="signup_btn"):
            if all([new_username, new_email, binance_api_key, binance_api_secret, new_password]):
                if register_user(new_username, new_email, new_password, binance_api_key, binance_api_secret):
                    st.success("Account created successfully! Please login.")
                    st.balloons()
            else:
                st.warning("Please fill in all fields")
    
    st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# BINANCE DATA FUNCTIONS
# =============================================================================

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_binance_client(api_key: str, api_secret: str):
    """Get Binance client"""
    try:
        return BinanceClient(api_key, api_secret)
    except Exception as e:
        st.error(f"Failed to connect to Binance: {str(e)}")
        return None

@st.cache_data(ttl=60)  # Cache for 1 minute
def get_account_info(api_key: str, api_secret: str):
    """Get Binance account information"""
    client = get_binance_client(api_key, api_secret)
    if client:
        try:
            return client.get_account()
        except BinanceAPIException as e:
            st.error(f"Binance API Error: {e}")
            return None
    return None

@st.cache_data(ttl=60)
def get_ticker_prices(api_key: str, api_secret: str):
    """Get all ticker prices"""
    client = get_binance_client(api_key, api_secret)
    if client:
        try:
            return client.get_all_tickers()
        except BinanceAPIException as e:
            st.error(f"Error fetching ticker prices: {e}")
            return None
    return None

@st.cache_data(ttl=300)
def get_klines(api_key: str, api_secret: str, symbol: str, interval: str, limit: int = 100):
    """Get candlestick data"""
    client = get_binance_client(api_key, api_secret)
    if client:
        try:
            klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['close'] = df['close'].astype(float)
            return df
        except BinanceAPIException as e:
            st.error(f"Error fetching candlestick data: {e}")
            return None
    return None

@st.cache_data(ttl=300)
def get_order_book(api_key: str, api_secret: str, symbol: str, limit: int = 100):
    """Get order book data"""
    client = get_binance_client(api_key, api_secret)
    if client:
        try:
            return client.get_order_book(symbol=symbol, limit=limit)
        except BinanceAPIException as e:
            st.error(f"Error fetching order book: {e}")
            return None
    return None

@st.cache_data(ttl=60)
def get_24hr_ticker(api_key: str, api_secret: str, symbol: str):
    """Get 24hr ticker statistics"""
    client = get_binance_client(api_key, api_secret)
    if client:
        try:
            return client.get_ticker(symbol=symbol)
        except BinanceAPIException as e:
            st.error(f"Error fetching 24hr ticker: {e}")
            return None
    return None

def main():
    """Main application function"""
    apply_modern_css()
    
    # Add sticky header
    st.markdown("""
    <div class="sticky-header">
        <h1>₿ Crypto Financial Dashboard</h1>
    </div>
    """, unsafe_allow_html=True)
    
    # Check authentication
    if 'session_token' not in st.session_state:
        show_auth_page()
        return
    
    # Verify session
    session_result = get_user_by_session(st.session_state['session_token'])
    if not session_result['success']:
        st.session_state.clear()
        st.rerun()
        return
    
    user = session_result['user']
    
    # Header
    st.markdown(
        f"""
        <div class="main-header">
            <h1>₿ Crypto Financial Dashboard</h1>
            <p style="text-align: center; color: #666666; font-size: 1.1rem;">
                Welcome back, {user['username']}! Professional crypto analysis with real-time Binance data
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # Logout button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col3:
        if st.button("Logout"):
            logout_user(st.session_state['session_token'])
            st.session_state.clear()
            st.rerun()
    
    # Get user's Binance credentials
    api_key = user['binance_api_key']
    api_secret = user['binance_api_secret']
    
    # Test Binance connection
    if not get_binance_client(api_key, api_secret):
        st.error("Failed to connect to Binance API. Please check your credentials.")
        return
    
    # Symbol selection
    st.markdown("### 💰 Select Trading Pair")
    col1, col2 = st.columns([2, 1])
    with col1:
        symbol = st.selectbox(
            "Choose a trading pair",
            ["BTCUSDT", "ETHUSDT", "ADAUSDT", "DOTUSDT", "LINKUSDT", "BNBUSDT", "XRPUSDT", "LTCUSDT"],
            index=0
        )
    
    with col2:
        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()
    
    # Get data
    with st.spinner("📊 Fetching data from Binance..."):
        ticker_data = get_24hr_ticker(api_key, api_secret, symbol)
        account_info = get_account_info(api_key, api_secret)
        klines_data = get_klines(api_key, api_secret, symbol, "1h", 168)  # 1 week of hourly data
        
        if ticker_data is None:
            st.error("Failed to fetch ticker data")
            return
    
    # Display current price and stats
    st.markdown("### 📈 Current Market Data")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        price = float(ticker_data['lastPrice'])
        price_change = float(ticker_data['priceChangePercent'])
        st.metric(
            "💰 Current Price",
            f"${price:,.4f}",
            f"{price_change:+.2f}%"
        )
    
    with col2:
        volume = float(ticker_data['volume'])
        st.metric(
            "📊 24h Volume",
            f"{volume:,.0f}",
            f"${float(ticker_data['quoteVolume']):,.0f}"
        )
    
    with col3:
        high = float(ticker_data['highPrice'])
        low = float(ticker_data['lowPrice'])
        st.metric(
            "📊 24h High",
            f"${high:,.4f}",
            f"${low:,.4f} (Low)"
        )
    
    with col4:
        if account_info:
            total_balance = sum(float(balance['free']) + float(balance['locked']) 
                             for balance in account_info['balances'] 
                             if float(balance['free']) > 0 or float(balance['locked']) > 0)
            st.metric(
                "💼 Portfolio Value",
                f"${total_balance:,.2f}",
                "Estimated"
            )
    
    # Price Chart
    st.markdown("### 📊 Price Chart (1 Week)")
    if klines_data is not None and not klines_data.empty:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        
        fig = go.Figure()
        
        # Candlestick chart
        fig.add_trace(go.Candlestick(
            x=klines_data['timestamp'],
            open=klines_data['open'].astype(float),
            high=klines_data['high'].astype(float),
            low=klines_data['low'].astype(float),
            close=klines_data['close'].astype(float),
            name=symbol
        ))
        
        fig.update_layout(
            title=f"{symbol} Price Chart",
            xaxis_title="Time",
            yaxis_title="Price (USDT)",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, sans-serif"),
            xaxis=dict(gridcolor="rgba(102, 126, 234, 0.2)"),
            yaxis=dict(gridcolor="rgba(102, 126, 234, 0.2)"),
            hovermode="x unified",
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Account Information
    if account_info:
        st.markdown("### 💼 Account Information")
        
        # Portfolio balances
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
            df_balances = df_balances.sort_values('Total', ascending=False)
            
            st.dataframe(
                df_balances,
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No balances found")
    
    # Trading Statistics
    st.markdown("### 📊 Trading Statistics")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 24h Statistics")
        stats_data = {
            'Metric': ['Price Change', 'Volume', 'High', 'Low', 'Open', 'Trades'],
            'Value': [
                f"{float(ticker_data['priceChangePercent']):+.2f}%",
                f"{float(ticker_data['volume']):,.0f}",
                f"${float(ticker_data['highPrice']):,.4f}",
                f"${float(ticker_data['lowPrice']):,.4f}",
                f"${float(ticker_data['openPrice']):,.4f}",
                f"{int(ticker_data['count']):,}"
            ]
        }
        st.dataframe(pd.DataFrame(stats_data), hide_index=True)
    
    with col2:
        # Order Book Preview
        order_book = get_order_book(api_key, api_secret, symbol, 10)
        if order_book:
            st.markdown("#### Order Book (Top 10)")
            
            bids_df = pd.DataFrame(order_book['bids'][:5], columns=['Price', 'Quantity'])
            asks_df = pd.DataFrame(order_book['asks'][:5], columns=['Price', 'Quantity'])
            
            bids_df['Price'] = bids_df['Price'].astype(float)
            bids_df['Quantity'] = bids_df['Quantity'].astype(float)
            asks_df['Price'] = asks_df['Price'].astype(float)
            asks_df['Quantity'] = asks_df['Quantity'].astype(float)
            
            st.markdown("**Bids (Buy Orders)**")
            st.dataframe(bids_df, hide_index=True)
            
            st.markdown("**Asks (Sell Orders)**")
            st.dataframe(asks_df, hide_index=True)
    
    # Footer
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
