# Standard library imports
import sys
from io import BytesIO
import os
import time
import hmac
import hashlib
import json
from datetime import datetime, timedelta
from urllib.parse import urlencode

# Third-party imports
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from binance.client import Client
from binance.exceptions import BinanceAPIException


# Configure the app page
st.set_page_config(
    page_title="Financial Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def millify(num, precision=2):
    """Custom implementation of millify function"""
    if num == 0:
        return "0"
    
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    
    return f"{num:.{precision}f}{['', 'K', 'M', 'B', 'T'][magnitude]}"


def apply_modern_css():
    """
    Apply modern CSS styling with glassmorphism effects, hover animations, improved UI, and sticky header
    """
    modern_css = """
    <style>
        /* Import Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

        /* Root variables for LIGHT MODE (default) */
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

        /* Sticky header implementation */
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
            transform: translateY(-100%);
            transition: transform 0.3s ease-in-out;
        }

        .sticky-header.visible {
            transform: translateY(0);
        }

        .sticky-header h2 {
            color: var(--text-primary);
            font-weight: 600;
            font-size: 1.5rem;
            margin: 0;
            text-align: center;
        }

        /* Add padding to body when sticky header is visible */
        .sticky-header-spacer {
            height: 80px;
        }

        /* Main header styling */
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

        /* Login section styling */
        .login-section {
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

        .login-section:hover {
            transform: translateY(-2px);
            box-shadow: var(--shadow-hover);
        }

        /* Company overview with logo styling */
        .company-overview {
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

        .company-overview:hover {
            transform: translateY(-2px);
            box-shadow: var(--shadow-hover);
        }

        .company-info {
            flex: 1;
        }

        .company-logo {
            flex-shrink: 0;
        }

        .company-logo img {
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
            margin-bottom: 1rem;
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
            color: var(--text-primary);
        }

        .chart-container:hover {
            transform: translateY(-2px);
            box-shadow: var(--shadow-hover);
        }

        /* Custom thinner footer */
        .custom-footer {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            background: var(--glass-bg);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border-top: 1px solid var(--glass-border);
            padding: 0.5rem;
            text-align: center;
            color: var(--text-secondary);
            font-size: 0.8rem;
            height: 40px;
            line-height: 1.5;
        }

        /* Responsive design */
        @media (max-width: 768px) {
            .main .block-container {
                padding: 1rem 0.5rem;
            }

            .main-header h1 {
                font-size: 2rem;
            }

            .company-overview {
                flex-direction: column;
                text-align: center;
            }

            .glass-card {
                padding: 1rem;
            }
        }

        /* Hide Streamlit elements */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}

        /* Sticky header scroll behavior */
        .sticky-header-script {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 1px;
            pointer-events: none;
            z-index: 9999;
        }
    </style>
    
    <script>
        // Sticky header functionality
        window.addEventListener('scroll', function() {
            const scrollPosition = window.pageYOffset;
            const stickyHeader = document.querySelector('.sticky-header');
            
            if (scrollPosition > 200) {
                stickyHeader.classList.add('visible');
            } else {
                stickyHeader.classList.remove('visible');
            }
        });
    </script>
    """
    st.markdown(modern_css, unsafe_allow_html=True)


def login_section():
    """Create login section for Binance API credentials"""
    st.markdown("### 🔐 Login to Binance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        api_key = st.text_input(
            "Binance API Key",
            type="password",
            placeholder="Enter your Binance API Key",
            help="Your Binance API Key for accessing market data"
        )
    
    with col2:
        api_secret = st.text_input(
            "Binance API Secret",
            type="password",
            placeholder="Enter your Binance API Secret",
            help="Your Binance API Secret for authentication"
        )
    
    if st.button("🚀 Connect to Binance", use_container_width=True):
        if api_key and api_secret:
            try:
                # Test the connection
                client = Client(api_key, api_secret)
                client.get_server_time()
                st.session_state["binance_client"] = client
                st.session_state["logged_in"] = True
                st.success("✅ Successfully connected to Binance!")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Failed to connect to Binance: {str(e)}")
        else:
            st.warning("⚠️ Please enter both API Key and Secret")


# =============================================================================
# BINANCE DATA FUNCTIONS
# =============================================================================

@st.cache_data(ttl=60 * 5)
def get_binance_symbol_info(client, symbol):
    """Get symbol information from Binance"""
    try:
        # Get symbol info
        symbol_info = client.get_symbol_info(symbol)
        
        # Get 24hr ticker
        ticker = client.get_ticker(symbol=symbol)
        
        # Get order book
        order_book = client.get_order_book(symbol=symbol, limit=10)
        
        return {
            "symbol": symbol_info["symbol"],
            "status": symbol_info["status"],
            "base_asset": symbol_info["baseAsset"],
            "quote_asset": symbol_info["quoteAsset"],
            "price": float(ticker["lastPrice"]),
            "price_change": float(ticker["priceChange"]),
            "price_change_percent": float(ticker["priceChangePercent"]),
            "volume": float(ticker["volume"]),
            "high_24h": float(ticker["highPrice"]),
            "low_24h": float(ticker["lowPrice"]),
            "open_price": float(ticker["openPrice"]),
            "bid_price": float(order_book["bids"][0][0]) if order_book["bids"] else 0,
            "ask_price": float(order_book["asks"][0][0]) if order_book["asks"] else 0,
            "count": int(ticker["count"]),
            "quote_volume": float(ticker["quoteVolume"]),
        }
    except Exception as e:
        st.error(f"Error fetching symbol info: {str(e)}")
        return None


@st.cache_data(ttl=60 * 5)
def get_binance_klines(client, symbol, interval="1d", limit=365):
    """Get historical klines/candlestick data"""
    try:
        klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
        
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        # Convert to proper data types
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['open'] = df['open'].astype(float)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)
        df['close'] = df['close'].astype(float)
        df['volume'] = df['volume'].astype(float)
        
        df = df.set_index('timestamp')
        return df
    except Exception as e:
        st.error(f"Error fetching klines: {str(e)}")
        return None


@st.cache_data(ttl=60 * 30)
def get_binance_trades(client, symbol, limit=100):
    """Get recent trades"""
    try:
        trades = client.get_recent_trades(symbol=symbol, limit=limit)
        
        df = pd.DataFrame(trades)
        if not df.empty:
            df['time'] = pd.to_datetime(df['time'], unit='ms')
            df['price'] = df['price'].astype(float)
            df['qty'] = df['qty'].astype(float)
        
        return df
    except Exception as e:
        st.error(f"Error fetching trades: {str(e)}")
        return None


@st.cache_data(ttl=60 * 10)
def get_binance_depth(client, symbol, limit=20):
    """Get order book depth"""
    try:
        depth = client.get_order_book(symbol=symbol, limit=limit)
        
        bids_df = pd.DataFrame(depth['bids'], columns=['price', 'quantity'])
        asks_df = pd.DataFrame(depth['asks'], columns=['price', 'quantity'])
        
        bids_df = bids_df.astype(float)
        asks_df = asks_df.astype(float)
        
        return {
            'bids': bids_df,
            'asks': asks_df,
            'last_update_id': depth['lastUpdateId']
        }
    except Exception as e:
        st.error(f"Error fetching depth: {str(e)}")
        return None


@st.cache_data(ttl=60 * 60)
def get_binance_exchange_info(client):
    """Get exchange information"""
    try:
        info = client.get_exchange_info()
        
        # Extract symbols
        symbols = []
        for symbol_info in info['symbols']:
            if symbol_info['status'] == 'TRADING':
                symbols.append({
                    'symbol': symbol_info['symbol'],
                    'base_asset': symbol_info['baseAsset'],
                    'quote_asset': symbol_info['quoteAsset'],
                    'status': symbol_info['status'],
                })
        
        return {
            'symbols': symbols,
            'server_time': info['serverTime'],
            'rate_limits': info['rateLimits']
        }
    except Exception as e:
        st.error(f"Error fetching exchange info: {str(e)}")
        return None


def create_sticky_header():
    """Create sticky header"""
    st.markdown("""
    <div class="sticky-header" id="sticky-header">
        <h2>📈 Financial Dashboard</h2>
    </div>
    """, unsafe_allow_html=True)


def main():
    """Main application function"""
    # Apply modern CSS styling
    apply_modern_css()
    
    # Create sticky header
    create_sticky_header()

    # Check if user is logged in
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False

    # Header
    st.markdown(
        """
    <div class="main-header">
        <h1>📈 Financial Dashboard</h1>
        <p style="text-align: center; color: #666666; font-size: 1.1rem;">
            Professional cryptocurrency analysis with real-time Binance data
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Login section
    if not st.session_state["logged_in"]:
        st.markdown(
            '<div class="login-section">',
            unsafe_allow_html=True,
        )
        login_section()
        st.markdown('</div>', unsafe_allow_html=True)
        return

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
            placeholder="Enter crypto symbol (e.g., BTCUSDT, ETHUSDT, ADAUSDT)",
            help="Enter a valid cryptocurrency symbol to analyze",
        ).upper()

        if st.button(
            "🔍 Analyze Cryptocurrency", on_click=callback, use_container_width=True
        ):
            pass

    # Main dashboard
    if st.session_state["btn_clicked"]:
        if not symbol_input:
            st.warning("⚠️ Please enter a cryptocurrency symbol.")
            return

        client = st.session_state["binance_client"]

        try:
            # Fetch data with loading indicators
            with st.spinner("📊 Fetching cryptocurrency data..."):
                symbol_info = get_binance_symbol_info(client, symbol_input)
                if symbol_info is None:
                    st.error(
                        "❌ Failed to retrieve symbol data. Please check the symbol."
                    )
                    return

                klines_data = get_binance_klines(client, symbol_input)
                trades_data = get_binance_trades(client, symbol_input)
                depth_data = get_binance_depth(client, symbol_input)

            # Company Overview with logo inside
            st.markdown("### 🏢 Cryptocurrency Overview")
            st.markdown(f"""
            <div class="company-overview">
                <div class="company-info">
                    <div class="company-name">{symbol_info['symbol']}</div>
                    <div class="company-details">
                        {symbol_info['base_asset']} / {symbol_info['quote_asset']} • Status: {symbol_info['status']}
                    </div>
                </div>
                <div class="company-logo">
                    <img src="https://cryptologos.cc/logos/{symbol_info['base_asset'].lower()}-{symbol_info['base_asset'].lower()}-logo.png" 
                         alt="{symbol_info['base_asset']} Logo"
                         style="height: 80px; width: 80px; object-fit: contain;"
                         onerror="this.src='https://via.placeholder.com/80x80?text={symbol_info['base_asset']}'">
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Key Metrics Row
            st.markdown("### 📊 Key Metrics")
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "💰 Current Price",
                    f"${symbol_info['price']:.4f}",
                    f"{symbol_info['price_change']:+.4f} ({symbol_info['price_change_percent']:+.2f}%)",
                )

            with col2:
                st.metric(
                    "📈 24h High",
                    f"${symbol_info['high_24h']:.4f}",
                    f"{((symbol_info['high_24h'] - symbol_info['price']) / symbol_info['price'] * 100):+.2f}%"
                )

            with col3:
                st.metric(
                    "📉 24h Low",
                    f"${symbol_info['low_24h']:.4f}",
                    f"{((symbol_info['low_24h'] - symbol_info['price']) / symbol_info['price'] * 100):+.2f}%"
                )

            with col4:
                st.metric(
                    "📊 24h Volume",
                    millify(symbol_info['volume'], precision=2),
                    f"Quote: {millify(symbol_info['quote_volume'], precision=2)}"
                )

            # Price Chart
            st.markdown("### 💹 Price Chart")
            with st.container():
                st.markdown(
                    '<div class="chart-container">', unsafe_allow_html=True
                )
                if klines_data is not None and not klines_data.empty:
                    fig = go.Figure()
                    
                    # Add candlestick chart
                    fig.add_trace(go.Candlestick(
                        x=klines_data.index,
                        open=klines_data['open'],
                        high=klines_data['high'],
                        low=klines_data['low'],
                        close=klines_data['close'],
                        name=symbol_input
                    ))
                    
                    fig.update_layout(
                        title=f"{symbol_input} Price Chart",
                        xaxis_title="Date",
                        yaxis_title="Price (USDT)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(family="Inter, sans-serif"),
                        xaxis=dict(gridcolor="rgba(102, 126, 234, 0.2)"),
                        yaxis=dict(gridcolor="rgba(102, 126, 234, 0.2)"),
                        hovermode="x unified",
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No price data available for charting.")
                st.markdown("</div>", unsafe_allow_html=True)

            # Trading Data
            st.markdown("### 📄 Trading Data")

            tab1, tab2, tab3 = st.tabs(["Recent Trades", "Order Book", "Market Statistics"])

            with tab1:
                st.markdown("#### Recent Trades")
                if trades_data is not None and not trades_data.empty:
                    display_trades = trades_data[['time', 'price', 'qty', 'isBuyerMaker']].head(20)
                    display_trades['side'] = display_trades['isBuyerMaker'].apply(
                        lambda x: '🔴 Sell' if x else '🟢 Buy'
                    )
                    display_trades = display_trades.drop('isBuyerMaker', axis=1)
                    display_trades.columns = ['Time', 'Price', 'Quantity', 'Side']
                    st.dataframe(display_trades, use_container_width=True)
                else:
                    st.info("No recent trades data available.")

            with tab2:
                st.markdown("#### Order Book")
                if depth_data is not None:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**🟢 Bids (Buy Orders)**")
                        bids = depth_data['bids'].head(10)
                        bids.columns = ['Price', 'Quantity']
                        st.dataframe(bids, use_container_width=True)
                    
                    with col2:
                        st.markdown("**🔴 Asks (Sell Orders)**")
                        asks = depth_data['asks'].head(10)
                        asks.columns = ['Price', 'Quantity']
                        st.dataframe(asks, use_container_width=True)
                else:
                    st.info("No order book data available.")

            with tab3:
                st.markdown("#### Market Statistics")
                stats_data = {
                    'Metric': [
                        'Current Price',
                        '24h Change',
                        '24h High',
                        '24h Low',
                        '24h Volume',
                        '24h Quote Volume',
                        'Trade Count',
                        'Bid Price',
                        'Ask Price',
                        'Spread'
                    ],
                    'Value': [
                        f"${symbol_info['price']:.4f}",
                        f"{symbol_info['price_change']:+.4f} ({symbol_info['price_change_percent']:+.2f}%)",
                        f"${symbol_info['high_24h']:.4f}",
                        f"${symbol_info['low_24h']:.4f}",
                        f"{symbol_info['volume']:.2f}",
                        f"{symbol_info['quote_volume']:.2f}",
                        f"{symbol_info['count']:,}",
                        f"${symbol_info['bid_price']:.4f}",
                        f"${symbol_info['ask_price']:.4f}",
                        f"${symbol_info['ask_price'] - symbol_info['bid_price']:.4f}"
                    ]
                }
                
                stats_df = pd.DataFrame(stats_data)
                st.dataframe(stats_df, use_container_width=True)

        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")
            st.warning("Please verify the symbol or try again later.")

    # Custom thinner Footer
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
