import streamlit as st
st.set_page_config(
    layout="wide",
    page_title="Black-Scholes Options Pricing",
    page_icon="📈"
)

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Thread
import time
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

# Enhanced CSS with better browser compatibility and consistent theming
st.markdown("""
<style>
    /* Main container styling */
    .main > div {
        padding-top: 1.5rem;
    }
    
    /* Tab styling with enhanced browser compatibility */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(0, 0, 0, 0.05);
        border-radius: 12px;
        padding: 8px;
        margin: 20px 0;
        border: 1px solid rgba(0, 0, 0, 0.1);
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
        -webkit-backdrop-filter: blur(10px);
        backdrop-filter: blur(10px);
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 10px 18px;
        margin: 0 4px;
        transition: all 0.2s ease;
        font-weight: 500;
        border: 1px solid transparent;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(0, 0, 0, 0.08);
        transform: translateY(-1px);
        border: 1px solid rgba(0, 0, 0, 0.1);
    }
    
    /* Metric container with consistent theming */
    .metric-container {
        background: rgba(248, 250, 252, 0.8);
        border: 1px solid rgba(226, 232, 240, 0.8);
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        margin-bottom: 12px;
        transition: all 0.2s ease;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    .metric-container:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        border-color: rgba(99, 102, 241, 0.2);
    }
    
    .metric-container h3 {
        font-size: 0.875rem;
        margin-bottom: 6px;
        color: rgba(71, 85, 105, 0.8);
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .metric-container h2 {
        font-size: 1.25rem;
        margin: 0;
        color: rgba(15, 23, 42, 0.9);
        font-weight: 600;
    }
    
    /* Progress container styling */
    .progress-container {
        background: rgba(248, 250, 252, 0.9);
        border: 1px solid rgba(226, 232, 240, 0.6);
        border-radius: 12px;
        padding: 20px;
        margin: 20px 0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }
    
    /* Results container */
    .results-container {
        margin: 24px 0;
    }
    
    /* Separator */
    .metric-separator {
        margin: 30px 0;
        border-bottom: 1px solid rgba(226, 232, 240, 0.6);
    }
    
    /* Sidebar enhancements */
    .sidebar .stNumberInput, 
    .sidebar .stSelectbox, 
    .sidebar .stSlider {
        margin-bottom: 18px;
    }
    
    /* Expander content */
    .expander-content {
        background: rgba(248, 250, 252, 0.5);
        border-radius: 8px;
        padding: 18px;
        border: 1px solid rgba(226, 232, 240, 0.4);
    }
    
    /* Dark mode compatibility */
    @media (prefers-color-scheme: dark) {
        .stTabs [data-baseweb="tab-list"] {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
        }
        
        .stTabs [data-baseweb="tab"]:hover {
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid rgba(255, 255, 255, 0.15);
        }
        
        .metric-container {
            background: rgba(30, 41, 59, 0.8);
            border: 1px solid rgba(71, 85, 105, 0.4);
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
        }
        
        .metric-container:hover {
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
            border-color: rgba(99, 102, 241, 0.4);
        }
        
        .metric-container h3 {
            color: rgba(226, 232, 240, 0.7);
        }
        
        .metric-container h2 {
            color: rgba(248, 250, 252, 0.95);
        }
        
        .progress-container {
            background: rgba(30, 41, 59, 0.9);
            border: 1px solid rgba(71, 85, 105, 0.4);
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
        }
        
        .expander-content {
            background: rgba(30, 41, 59, 0.5);
            border: 1px solid rgba(71, 85, 105, 0.3);
        }
        
        .metric-separator {
            border-bottom-color: rgba(71, 85, 105, 0.4);
        }
    }
    
    /* Enhanced responsiveness */
    @media (max-width: 768px) {
        .metric-container {
            padding: 12px;
            margin-bottom: 8px;
        }
        
        .metric-container h2 {
            font-size: 1.1rem;
        }
        
        .metric-container h3 {
            font-size: 0.8rem;
        }
        
        .progress-container {
            padding: 16px;
            margin: 16px 0;
        }
    }
    
    /* Enhanced focus states for accessibility */
    .stTabs [data-baseweb="tab"]:focus {
        outline: 2px solid rgba(99, 102, 241, 0.5);
        outline-offset: 2px;
    }
</style>
""", unsafe_allow_html=True)

### Core Functions ######################################################
def BlackScholes(r, S, K, T, sigma, tipo='C'):
    ''' 
    r : Interest Rate
    S : Spot Price
    K : Strike Price
    T : Days due expiration / 365
    sigma : Annualized Volatility 
    '''
    d1 = (np.log(S/K) + (r + sigma**2/2)*T)/(sigma * np.sqrt(T)) 
    d2 = d1 - sigma * np.sqrt(T)
    try: 
        if tipo == 'C': 
            precio = S * norm.cdf(d1, 0, 1) - K * np.exp(-r * T)*norm.cdf(d2, 0, 1)
        elif tipo == 'P': 
            precio = K * np.exp(-r * T)*norm.cdf(-d2, 0, 1) - S * norm.cdf(-d1, 0, 1)
    except: 
        print('Error in BlackScholes calculation')
    return precio

def calculate_row_threaded(args):
    """Calculate a single row of the heatmap matrix using threading"""
    spot_price, volatilities, strike, interest_rate, T, option_type = args
    row = []
    for vol in volatilities:
        bs_result = BlackScholes(interest_rate, spot_price, strike, T, vol, option_type)
        row.append(round(bs_result, 2))
    return row

def HeatMapMatrix(spot_prices, volatilities, strike, interest_rate, days_to_exp, option_type='C'):
    """Create heatmap matrix using concurrent processing"""
    T = days_to_exp / 365
    
    # Prepare arguments for threading
    args_list = [(spot_price, volatilities, strike, interest_rate, T, option_type) 
                 for spot_price in spot_prices]
    
    # Use ThreadPoolExecutor for concurrent processing
    with ThreadPoolExecutor(max_workers=min(len(spot_prices), 8)) as executor:
        # Submit all tasks
        future_to_row = {executor.submit(calculate_row_threaded, args): i 
                        for i, args in enumerate(args_list)}
        
        # Initialize matrix
        matrix = np.zeros((len(spot_prices), len(volatilities)))
        
        # Collect results as they complete
        for future in as_completed(future_to_row):
            row_index = future_to_row[future]
            try:
                row_data = future.result()
                matrix[row_index] = row_data
            except Exception as exc:
                st.error(f'Row {row_index} generated an exception: {exc}')
    
    return matrix

@st.cache_data
def simulate_paths(ns, days_to_maturity, steps, volatility, risk_free_rate, underlying_price):
    """Cached simulation function with threading for path generation"""
    dt = (days_to_maturity / 365) / steps
    
    def generate_path_chunk(chunk_size):
        """Generate a chunk of simulation paths"""
        z = np.random.normal(0, np.sqrt(dt), (steps, chunk_size))
        paths = np.vstack([
            np.ones(chunk_size), 
            np.exp((risk_free_rate - 0.5 * volatility**2) * dt + volatility * z)
        ]).cumprod(axis=0)
        return underlying_price * paths
    
    # Determine chunk size and number of chunks for threading
    chunk_size = max(1, ns // 4)  # Divide into 4 chunks
    chunks = []
    remaining = ns
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        while remaining > 0:
            current_chunk_size = min(chunk_size, remaining)
            futures.append(executor.submit(generate_path_chunk, current_chunk_size))
            remaining -= current_chunk_size
        
        # Collect results
        for future in as_completed(futures):
            chunks.append(future.result())
    
    # Combine all chunks
    return np.hstack(chunks)

###############################################################################################################
#### Sidebar Parameters ###############################################
with st.sidebar:
    st.header('⚙️ Option Parameters')
    
    # Move contract type to top and make horizontal
    trade_type = st.radio("Contract Type", ['Call', 'Put'], horizontal=True, help="Select the type of option contract")
    
    col1, col2 = st.columns(2)
    with col1:
        underlying_price = st.number_input('Spot Price', value=100.0, step=1.0)
        selected_strike = st.number_input('Strike/Exercise Price', value=80.0, step=1.0)
        days_to_maturity = st.number_input('Time to Maturity (days)', value=365, step=1)
    
    with col2:
        risk_free_rate = st.number_input('Risk-Free Interest Rate', value=0.1, step=0.01, format="%.3f")
        volatility = st.number_input('Annualized Volatility', value=0.2, step=0.01, format="%.3f")
    
    st.subheader('P&L Parameters')
    col3, col4 = st.columns(2)
    with col3:
        option_purchase_price = st.number_input("Option's Price", step=0.01, format="%.2f") 
    with col4:
        transaction_cost = st.number_input("Opening/Closing Cost", step=0.01, format="%.2f")
    
    st.subheader('Heatmap Parameters')
    col5, col6 = st.columns(2)
    with col5:
        min_spot_price = st.number_input('Min Spot price', value=50.0, step=1.0)
        max_spot_price = st.number_input('Max Spot price', value=110.0, step=1.0)
    with col6:
        min_vol = st.slider('Min Volatility', 0.01, 1.00, 0.01, step=0.01)
        max_vol = st.slider('Max Volatility', 0.01, 1.00, 1.00, step=0.01)
    
    grid_size = st.slider('Grid size (nxn)', 5, 20, 10)

#### Main App Layout ########################################################
st.title('📈 Black-Scholes Options Pricing')
st.write("Calculate option premiums using the Black-Scholes model with real-time heatmaps and Monte Carlo simulations.")

# Calculate current option prices
call_price = BlackScholes(risk_free_rate, underlying_price, selected_strike, days_to_maturity / 365, volatility)
put_price = BlackScholes(risk_free_rate, underlying_price, selected_strike, days_to_maturity / 365, volatility, 'P')

# Display current prices with enhanced styling
col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    st.markdown(
        f'<div class="metric-container"><h3>Call Value</h3><h2>${call_price:.3f}</h2></div>', 
        unsafe_allow_html=True
    )
with col2:
    st.markdown(
        f'<div class="metric-container"><h3>Put Value</h3><h2>${put_price:.3f}</h2></div>', 
        unsafe_allow_html=True
    )
with col3:
    put_call_parity = call_price - put_price + selected_strike * np.exp(-risk_free_rate * days_to_maturity / 365) - underlying_price
    st.markdown(
        f'<div class="metric-container"><h3>Put-Call Parity</h3><h2>${put_call_parity:.3f}</h2></div>', 
        unsafe_allow_html=True
    )

# Add separator
st.markdown('<div class="metric-separator"></div>', unsafe_allow_html=True)

# Generate spaces for calculations
spot_prices_space = np.linspace(min_spot_price, max_spot_price, grid_size)
volatilities_space = np.linspace(min_vol, max_vol, grid_size)

# Tabs for different views
tab1, tab2, tab3 = st.tabs(["📊 Fair Value Heatmap", "💰 P&L Analysis", "🎲 Monte Carlo Simulation"])

with tab1:
    st.write("Explore option values across different spot prices and volatilities")
    
    # Progress indicator
    progress_placeholder = st.empty()
    with progress_placeholder.container():
        st.markdown('<div class="progress-container">', unsafe_allow_html=True)
        st.info("🔄 Calculating heatmaps using concurrent processing...")
        progress_bar = st.progress(0)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Calculate matrices using threading
    start_time = time.time()
    
    # Use threading for both call and put calculations
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_call = executor.submit(HeatMapMatrix, spot_prices_space, volatilities_space, selected_strike, risk_free_rate, days_to_maturity, 'C')
        future_put = executor.submit(HeatMapMatrix, spot_prices_space, volatilities_space, selected_strike, risk_free_rate, days_to_maturity, 'P')
        
        progress_bar.progress(50)
        
        output_matrix_c = future_call.result()
        output_matrix_p = future_put.result()
        
        progress_bar.progress(100)
    
    calc_time = time.time() - start_time
    progress_placeholder.empty()
    
    # Create enhanced Plotly heatmaps
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Call Options Heatmap', 'Put Options Heatmap'),
        vertical_spacing=0.08
    )
    
    # Call heatmap
    fig.add_trace(
        go.Heatmap(
            z=output_matrix_c.T,
            x=[f"{x:.1f}" for x in spot_prices_space],
            y=[f"{y:.2f}" for y in volatilities_space],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Call Value", x=1.02, len=0.45, y=0.77),
            text=np.round(output_matrix_c.T, 1),
            texttemplate="%{text}",
            textfont={"size": 10}
        ),
        row=1, col=1
    )
    
    # Put heatmap
    fig.add_trace(
        go.Heatmap(
            z=output_matrix_p.T,
            x=[f"{x:.1f}" for x in spot_prices_space],
            y=[f"{y:.2f}" for y in volatilities_space],
            colorscale='Plasma',
            showscale=True,
            colorbar=dict(title="Put Value", x=1.02, len=0.45, y=0.23),
            text=np.round(output_matrix_p.T, 1),
            texttemplate="%{text}",
            textfont={"size": 10}
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        title_text="Options Fair Value Analysis",
        height=800,
        showlegend=False,
        font=dict(size=12)
    )
    fig.update_xaxes(title_text="Spot Price", row=1, col=1)
    fig.update_xaxes(title_text="Spot Price", row=2, col=1)
    fig.update_yaxes(title_text="Volatility", row=1, col=1)
    fig.update_yaxes(title_text="Volatility", row=2, col=1)
    
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.write("Analyze expected profit/loss across market scenarios")
    
    if 'output_matrix_c' in locals() and 'output_matrix_p' in locals():
        call_pl = output_matrix_c.T - option_purchase_price - 2 * transaction_cost
        put_pl = output_matrix_p.T - option_purchase_price - 2 * transaction_cost
        
        selection = 0 if trade_type == 'Call' else 1
        contract_prices = [call_price, put_price]
        pl_options = [call_pl, put_pl]
        
        specific_contract_pl = contract_prices[selection] - option_purchase_price - 2 * transaction_cost
        st.markdown(
            f'<div class="metric-container"><h3>Expected P&L (Current Parameters)</h3><h2>${specific_contract_pl:.2f}</h2></div>', 
            unsafe_allow_html=True
        )
        
        # Create P&L heatmap with Plotly
        fig = go.Figure(data=go.Heatmap(
            z=pl_options[selection],
            x=[f"{x:.1f}" for x in spot_prices_space],
            y=[f"{y:.2f}" for y in volatilities_space],
            colorscale='RdYlGn',
            zmid=0,
            showscale=True,
            colorbar=dict(title="P&L ($)"),
            text=np.round(pl_options[selection], 1),
            texttemplate="%{text}",
            textfont={"size": 10}
        ))
        
        fig.update_layout(
            title=f'{trade_type} Options P&L Analysis',
            xaxis_title='Spot Price',
            yaxis_title='Volatility',
            height=600,
            font=dict(size=12)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Please calculate the fair value heatmaps first in Tab 1")

with tab3:
    st.write('Monte Carlo simulation of underlying asset price and option P&L distributions')
    
    with st.expander("📖 Methodology"):
        st.markdown('<div class="expander-content">', unsafe_allow_html=True)
        st.write('The distribution is simulated using geometric Brownian motion:')
        st.latex(r'S(t) = S(0) e^{(\mu - \sigma^2 / 2)t + \sigma W(t)}')
        st.write('Where μ is the risk-free rate, σ is the annualized volatility, and S(0) is the initial spot price.')
        st.markdown('</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        ns = st.slider('Number of simulations', 100, 10000, 1000, 10)
    with col2:
        s_selection = st.radio('Time interval', ['Days', 'Hours', 'Minutes'], horizontal=True, 
                              help='Display granularity for visualization')
    with col3:
        timeshot = st.slider("Timeline position", 0.0, days_to_maturity / 365, days_to_maturity / 365, format="%.3f") 

    if s_selection == 'Days':
        step = days_to_maturity 
    elif s_selection == 'Hours':
        step = days_to_maturity * 24 
    elif s_selection == 'Minutes':
        step = days_to_maturity * 24 * 60 
    
    # Generate simulation paths
    start_sim_time = time.time()
    simulation_paths = simulate_paths(ns, days_to_maturity, step, volatility, risk_free_rate, underlying_price)
    
    def get_option_price(K, St, option_type='Call'):
        dynamic_index = -int(step - timeshot * 365 * (step/days_to_maturity) + 1)
        try: 
            if option_type == 'Call':
                expiration_price = np.maximum(St[dynamic_index, :] - K, 0)
            elif option_type == 'Put':
                expiration_price = np.maximum(K - St[dynamic_index, :], 0)
        except Exception as e:
            st.error(f'Error in option price calculation: {e}')
            return np.zeros(St.shape[1])
        return expiration_price

    option_prices = get_option_price(selected_strike, simulation_paths, trade_type)
    pl_results = option_prices - option_purchase_price - 2 * transaction_cost
    sim_time = time.time() - start_sim_time

    # Calculate probabilities
    otm_probability = round(sum(option_prices == 0) / len(option_prices), 3)
    itm_probability = round(1 - otm_probability, 3)
    positive_pl_proba = round(sum(pl_results > 0) / len(pl_results), 3)

    st.subheader('📈 Simulation Results')
    st.markdown('<div class="results-container">', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            f'<div class="metric-container"><h3>In-the-Money Probability</h3><h2>{itm_probability:.1%}</h2></div>', 
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            f'<div class="metric-container"><h3>Out-of-the-Money Probability</h3><h2>{otm_probability:.1%}</h2></div>', 
            unsafe_allow_html=True
        )
    with col3:
        st.markdown(
            f'<div class="metric-container"><h3>Positive P&L Probability</h3><h2>{positive_pl_proba:.1%}</h2></div>', 
            unsafe_allow_html=True
        )
    
    st.markdown('</div>', unsafe_allow_html=True)

    # Create enhanced Plotly visualizations matching original layout
    col1, col2 = st.columns(2)
    
    with col1:
        # Underlying asset price distribution (matches original)
        index_to_use = -int(step - timeshot * step + 1)
        price_data = simulation_paths[index_to_use, :]
        
        fig1 = px.histogram(
            x=price_data, 
            nbins=30,
            title=f'Expected underlying asset price distribution at day {int(timeshot * 365)}',
            labels={'x': 'Price', 'y': 'Count'},
            marginal='box',
            opacity=0.7
        )
        fig1.add_vline(x=selected_strike, line_dash="dash", line_color="red", 
                      annotation_text="Strike price", annotation_position="top right")
        fig1.update_layout(height=500, showlegend=False, font=dict(size=11))
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        # Option premium distribution (matches original layout)
        fig2 = px.histogram(
            x=option_prices,
            nbins=30,
            title=f'Expected {trade_type} premium at day {int(timeshot * 365)}',
            labels={'x': 'Price', 'y': 'Count'},
            opacity=0.7
        )
        fig2.update_layout(height=200, showlegend=False, font=dict(size=11))
        st.plotly_chart(fig2, use_container_width=True)
        
        # P&L distribution (matches original layout)
        fig3 = px.histogram(
            x=pl_results,
            nbins=30,
            title=f'Expected P&L distribution at day {int(timeshot * 365)}',
            labels={'x': 'Price', 'y': 'Count'},
            opacity=0.7
        )
        fig3.add_vline(x=0, line_dash="solid", line_color="red", 
                      annotation_text="Break-even", annotation_position="top right")
        mean_pl = np.mean(pl_results)
        fig3.add_vline(x=mean_pl, line_dash="dash", line_color="green", 
                      annotation_text=f"Mean: ${mean_pl:.2f}", annotation_position="top left")
        fig3.update_layout(height=280, showlegend=False, font=dict(size=11))
        st.plotly_chart(fig3, use_container_width=True)
