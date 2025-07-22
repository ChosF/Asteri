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

# Modern CSS with better browser compatibility and light/dark mode support
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    
    /* Tabs styling with better browser support */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(128, 128, 128, 0.1);
        border-radius: 12px;
        padding: 8px;
        margin-bottom: 20px;
        margin-top: 30px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 16px;
        margin: 0 4px;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(128, 128, 128, 0.15);
        transform: translateY(-2px);
    }
    
    /* Metric containers with fallback styling */
    .metric-container {
        background: rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        padding: 12px;
        border: 1px solid rgba(128, 128, 128, 0.2);
        transition: all 0.3s ease;
        text-align: center;
        margin-bottom: 10px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }
    
    .metric-container h3 {
        font-size: 0.85rem;
        margin-bottom: 4px;
        opacity: 0.8;
    }
    
    .metric-container h2 {
        font-size: 1.1rem;
        margin: 0;
        font-weight: 600;
    }
    
    .metric-container:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.15);
    }
    
    .metric-separator {
        margin: 25px 0;
        border-bottom: 1px solid rgba(128, 128, 128, 0.2);
    }
    
    /* Sidebar improvements */
    .sidebar .stNumberInput, .sidebar .stSelectbox, .sidebar .stSlider {
        margin-bottom: 16px;
    }
    
    .progress-container {
        background: rgba(128, 128, 128, 0.05);
        border-radius: 12px;
        padding: 16px;
        margin: 16px 0;
        border: 1px solid rgba(128, 128, 128, 0.1);
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }
    
    .expander-content {
        background: rgba(128, 128, 128, 0.03);
        border-radius: 8px;
        padding: 16px;
    }
    
    .results-container {
        margin: 20px 0;
    }
    
    /* Dark mode support */
    @media (prefers-color-scheme: dark) {
        .metric-container, .progress-container {
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid rgba(255, 255, 255, 0.15);
        }
        .stTabs [data-baseweb="tab-list"] {
            background: rgba(255, 255, 255, 0.08);
        }
        .metric-container:hover {
            box-shadow: 0 4px 15px rgba(255, 255, 255, 0.1);
        }
    }
    
    /* Light mode support */
    @media (prefers-color-scheme: light) {
        .metric-container, .progress-container {
            background: rgba(0, 0, 0, 0.03);
            border: 1px solid rgba(0, 0, 0, 0.1);
        }
        .stTabs [data-baseweb="tab-list"] {
            background: rgba(0, 0, 0, 0.05);
        }
        .metric-container:hover {
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.15);
        }
    }
    
    /* Improved browser compatibility */
    .metric-container {
        -webkit-transform: translateY(0);
        -moz-transform: translateY(0);
        -ms-transform: translateY(0);
        transform: translateY(0);
    }
    
    .metric-container:hover {
        -webkit-transform: translateY(-2px);
        -moz-transform: translateY(-2px);
        -ms-transform: translateY(-2px);
        transform: translateY(-2px);
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
    
    # Contract type moved to top and made horizontal
    trade_type = st.segmented_control("Contract type", ['Call', 'Put'], default='Call')
    
    col1, col2 = st.columns(2)
    with col1:
        underlying_price = st.number_input('Spot Price', value=100.0, step=1.0)
        selected_strike = st.number_input('Strike/Exercise Price', value=80.0, step=1.0)
        days_to_maturity = st.number_input('Time to Maturity (days)', value=365, step=1)
    
    with col2:
        risk_free_rate = st.number_input('Risk-Free Interest Rate ', value=0.1, step=0.01, format="%.3f")
        volatility = st.number_input('Annualized Volatility', value=0.2, step=0.01, format="%.3f")
    
    st.subheader('P&L Parameters')
    col3, col4 = st.columns(2)
    with col3:
        option_purchase_price = st.number_input("Option's Price") 
    with col4:
        transaction_cost = st.number_input("Opening/Closing Cost") 
    
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
st.header('Black Scholes options heatmap')
st.write("Calculates an option's arbitrage-free premium using the Black Scholes option pricing model.")

# Calculate current option prices
call_price = BlackScholes(risk_free_rate, underlying_price, selected_strike, days_to_maturity / 365, volatility)
put_price = BlackScholes(risk_free_rate, underlying_price, selected_strike, days_to_maturity / 365, volatility, 'P')

# Display current prices with custom styling
col1, col2 = st.columns(2)
with col1:
    st.markdown(f"Call value: **{round(call_price,3)}**")
with col2:
    st.markdown(f"Put value: **{round(put_price,3)}**")

# Add separator
st.markdown('<div class="metric-separator"></div>', unsafe_allow_html=True)

# Generate spaces for calculations
spot_prices_space = np.linspace(min_spot_price, max_spot_price, grid_size)
volatilities_space = np.linspace(min_vol, max_vol, grid_size)

# Tabs for different views
tab1, tab2, tab3 = st.tabs(["Option's fair value heatmap", "Option's P&L heatmap", "Expected underlying distribution"])

with tab1:
    st.write("Explore different contract's values given variations in Spot Prices and Annualized Volatilities")
    
    # Progress indicator
    progress_placeholder = st.empty()
    with progress_placeholder.container():
        st.markdown('<div class="progress-container">', unsafe_allow_html=True)
        st.info("Calculating heatmaps using concurrent processing...")
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
    
    # Create heatmaps exactly like the second program
    fig, axs = plt.subplots(2, 1, figsize=(25, 25))

    sns.heatmap(output_matrix_c.T, annot=True, fmt='.1f',
                xticklabels=[str(round(i, 2)) for i in spot_prices_space], 
                yticklabels=[str(round(i, 2)) for i in volatilities_space], 
                ax=axs[0], cbar_kws={'label': 'Call Value'})
    axs[0].set_title('Call heatmap', fontsize=20)
    axs[0].set_xlabel('Spot Price', fontsize=15)
    axs[0].set_ylabel('Annualized Volatility', fontsize=15)

    sns.heatmap(output_matrix_p.T, annot=True, fmt='.1f',
                xticklabels=[str(round(i, 2)) for i in spot_prices_space], 
                yticklabels=[str(round(i, 2)) for i in volatilities_space], 
                ax=axs[1], cbar_kws={'label': 'Put Value'})
    axs[1].set_title('Put heatmap', fontsize=20)
    axs[1].set_xlabel('Spot Price', fontsize=15)
    axs[1].set_ylabel('Annualized Volatility', fontsize=15)

    plt.tight_layout()
    st.pyplot(fig)

with tab2:
    st.write("Explore different expected P&L's from a specific contract trade given variations in the Spot Price and Annualized Volatility")
    
    if 'output_matrix_c' in locals() and 'output_matrix_p' in locals():
        fig, axs = plt.subplots(1, 1, figsize=(25, 15))

        call_pl = output_matrix_c.T - option_purchase_price - 2 * transaction_cost
        put_pl = output_matrix_p.T - option_purchase_price - 2 * transaction_cost
        pl_options = [call_pl, put_pl]
        
        selection = 0 if trade_type == 'Call' else 1
        contract_prices = [call_price, put_price]
        
        specific_contract_pl = contract_prices[selection] - option_purchase_price - 2 * transaction_cost
        st.markdown(f'Expected P&L given selected parameters: **{round(specific_contract_pl,2)}**')
        
        mapping_color = sns.diverging_palette(15, 145, s=60, as_cmap=True)
        sns.heatmap(pl_options[selection], annot=True, fmt='.1f',
                    xticklabels=[str(round(i, 2)) for i in spot_prices_space], 
                    yticklabels=[str(round(i, 2)) for i in volatilities_space], 
                    ax=axs, cmap=mapping_color, center=0)
        axs.set_title(f'{trade_type} Expected P&L', fontsize=20)
        axs.set_xlabel('Spot Price', fontsize=15)
        axs.set_ylabel('Annualized Volatility', fontsize=15)

        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.info("Please calculate the fair value heatmaps first in Tab 1")

with tab3:
    st.write('Calculate the expected distribution of the underlying asset price, the option premium and the p&l from trading the option')
    
    with st.expander("See methodology"):
        st.markdown('<div class="expander-content">', unsafe_allow_html=True)
        st.write('The distribution is obtained by simulating $N$ times the underlying asset price as a geometric brownian process during a specified time period.' \
        ' The function $S : [0, \\infty) \\mapsto [0, \\infty) $ will describe the stochastic process as: ')
        st.latex('S(t) = S(0) e^{(\\mu - \\sigma^2 / 2)t + \\sigma W(t)} ')
        st.write('Where $\\mu$ is the risk free rate, $\\sigma$ the annualized volatility of the asset you want to simulate and $S(0)$ the asset price at the beginning (spot price)')
        st.markdown('</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        ns = st.slider('Number of simulations ($N$)', 100, 10000, 1000, 10)
    with col2:
        s_selection = st.radio('Select time interval', ['Days', 'Hours', 'Minutes'], horizontal=True, 
                              help='The time interval each price point will represent. This option is merely for visual purposes.')
    with col3:
        timeshot = st.slider("Select chart's timestamp (days/year)", 0.0, days_to_maturity / 365, days_to_maturity / 365)

    if s_selection == 'Days':
        step = days_to_maturity 
    elif s_selection == 'Hours':
        step = days_to_maturity * 24 
    elif s_selection == 'Minutes':
        step = days_to_maturity * 24 * 60 
    
    # Generate simulation paths using threading
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
    otm_probability = round(sum(option_prices == 0) / len(option_prices), 2)
    itm_probability = round(1 - otm_probability, 2)
    positive_pl_proba = round(sum(pl_results > 0) / len(pl_results), 2)

    st.subheader('Results')
    st.markdown('<div class="results-container">', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("In-the money probability", itm_probability, border=True)
    col2.metric("Out-the money probability", otm_probability, border=True)
    col3.metric("Positive P&L probability", positive_pl_proba, border=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

    # Create plots matching the second program exactly
    col1, col2 = st.columns(2)
    
    with col1:
        # Underlying asset price distribution
        t3_fig1 = plt.figure(figsize=(8, 8))
        sns.histplot(simulation_paths[-int(step - timeshot * step + 1), :], kde=True, stat='probability')
        plt.xlabel('Price')
        plt.axvline(selected_strike, 0, 1, color='r', label='Strike price')
        plt.title(f'Expected underlying asset price distribution at day {int(timeshot * 365)}')
        plt.legend()
        st.pyplot(t3_fig1)

    with col2:
        # Option premium distribution
        t3_fig2 = plt.figure(figsize=(8, 3))
        sns.histplot(option_prices, kde=True, stat='probability')
        plt.xlabel('Price')
        plt.title(f'Expected {trade_type} premium at day {int(timeshot * 365)}')
        plt.legend()
        st.pyplot(t3_fig2)

        # P&L distribution
        t3_fig3 = plt.figure(figsize=(8, 3))
        sns.histplot(pl_results, kde=True, stat='probability')
        plt.xlabel('Price')
        plt.title(f'Expected P&L distribution at day {int(timeshot * 365)}')
        plt.legend()
        st.pyplot(t3_fig3)
