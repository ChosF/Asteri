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

# Custom CSS for modern UX with light/dark mode support
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        backdrop-filter: blur(10px);
        background: rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 8px;
        margin-bottom: 20px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 16px;
        margin: 0 4px;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(255, 255, 255, 0.2);
        transform: translateY(-2px);
    }
    
    .metric-container {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: all 0.3s ease;
    }
    
    .metric-container:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
    }
    
    .sidebar .stNumberInput, .sidebar .stSelectbox, .sidebar .stSlider {
        margin-bottom: 16px;
    }
    
    .sidebar {
        background: rgba(255, 255, 255, 0.02);
        backdrop-filter: blur(20px);
        border-radius: 0 16px 16px 0;
    }
    
    .progress-container {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 12px;
        padding: 16px;
        margin: 16px 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .expander-content {
        background: rgba(255, 255, 255, 0.02);
        border-radius: 8px;
        padding: 16px;
    }
    
    @media (prefers-color-scheme: dark) {
        .metric-container, .progress-container {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
    }
    
    @media (prefers-color-scheme: light) {
        .metric-container, .progress-container {
            background: rgba(0, 0, 0, 0.02);
            border: 1px solid rgba(0, 0, 0, 0.1);
        }
        .stTabs [data-baseweb="tab-list"] {
            background: rgba(0, 0, 0, 0.05);
        }
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
    
    col1, col2 = st.columns(2)
    with col1:
        underlying_price = st.number_input('Spot Price', value=100.0, step=1.0)
        selected_strike = st.number_input('Strike/Exercise Price', value=80.0, step=1.0)
        days_to_maturity = st.number_input('Time to Maturity (days)', value=365, step=1)
    
    with col2:
        risk_free_rate = st.number_input('Risk-Free Interest Rate', value=0.1, step=0.01, format="%.3f")
        volatility = st.number_input('Annualized Volatility', value=0.2, step=0.01, format="%.3f")
        trade_type = st.segmented_control("Contract type", ['Call', 'Put'], default='Call')
    
    st.subheader('📊 P&L Parameters')
    col3, col4 = st.columns(2)
    with col3:
        option_purchase_price = st.number_input("Option's Price", step=0.01, format="%.2f") 
    with col4:
        transaction_cost = st.number_input("Opening/Closing Cost", step=0.01, format="%.2f")
    
    st.subheader('🔥 Heatmap Parameters')
    col5, col6 = st.columns(2)
    with col5:
        min_spot_price = st.number_input('Min Spot price', value=50.0, step=1.0)
        max_spot_price = st.number_input('Max Spot price', value=110.0, step=1.0)
    with col6:
        min_vol = st.slider('Min Volatility', 0.01, 1.00, 0.01, step=0.01)
        max_vol = st.slider('Max Volatility', 0.01, 1.00, 1.00, step=0.01)
    
    grid_size = st.slider('Grid size (nxn)', 5, 20, 10)

#### Main App Layout ########################################################
st.title('📈 Black Scholes Options Heatmap')
st.write("Calculates an option's arbitrage-free premium using the Black Scholes option pricing model.")

# Calculate current option prices
call_price = BlackScholes(risk_free_rate, underlying_price, selected_strike, days_to_maturity / 365, volatility)
put_price = BlackScholes(risk_free_rate, underlying_price, selected_strike, days_to_maturity / 365, volatility, 'P')

# Display current prices with custom styling
col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    st.markdown(
        f'<div class="metric-container"><h3>📞 Call Value</h3><h2>${call_price:.3f}</h2></div>', 
        unsafe_allow_html=True
    )
with col2:
    st.markdown(
        f'<div class="metric-container"><h3>📋 Put Value</h3><h2>${put_price:.3f}</h2></div>', 
        unsafe_allow_html=True
    )
with col3:
    put_call_parity = call_price - put_price + selected_strike * np.exp(-risk_free_rate * days_to_maturity / 365) - underlying_price
    st.markdown(
        f'<div class="metric-container"><h3>⚖️ Put-Call Parity</h3><h2>${put_call_parity:.3f}</h2></div>', 
        unsafe_allow_html=True
    )

# Generate spaces for calculations
spot_prices_space = np.linspace(min_spot_price, max_spot_price, grid_size)
volatilities_space = np.linspace(min_vol, max_vol, grid_size)

# Tabs for different views
tab1, tab2, tab3 = st.tabs(["🔥 Option's Fair Value Heatmap", "💰 Option's P&L Heatmap", "📊 Expected Underlying Distribution"])

with tab1:
    st.write("Explore different contract's values given variations in Spot Prices and Annualized Volatilities")
    
    # Progress indicator
    progress_placeholder = st.empty()
    with progress_placeholder.container():
        st.markdown('<div class="progress-container">', unsafe_allow_html=True)
        st.info("⏳ Calculating heatmaps using concurrent processing...")
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
    
    st.success(f"✅ Calculations completed in {calc_time:.2f} seconds")
    
    # Create heatmaps
    fig, axs = plt.subplots(2, 1, figsize=(12, 16))
    
    sns.heatmap(output_matrix_c.T, annot=True, fmt='.1f',
                xticklabels=[str(round(i, 2)) for i in spot_prices_space], 
                yticklabels=[str(round(i, 2)) for i in volatilities_space], 
                ax=axs[0], cbar_kws={'label': 'Call Value'})
    axs[0].set_title('Call Options Heatmap', fontsize=16, pad=20)
    axs[0].set_xlabel('Spot Price', fontsize=12)
    axs[0].set_ylabel('Annualized Volatility', fontsize=12)

    sns.heatmap(output_matrix_p.T, annot=True, fmt='.1f',
                xticklabels=[str(round(i, 2)) for i in spot_prices_space], 
                yticklabels=[str(round(i, 2)) for i in volatilities_space], 
                ax=axs[1], cbar_kws={'label': 'Put Value'})
    axs[1].set_title('Put Options Heatmap', fontsize=16, pad=20)
    axs[1].set_xlabel('Spot Price', fontsize=12)
    axs[1].set_ylabel('Annualized Volatility', fontsize=12)

    plt.tight_layout()
    st.pyplot(fig)

with tab2:
    st.write("Explore different expected P&L's from a specific contract trade given variations in the Spot Price and Annualized Volatility")
    
    if 'output_matrix_c' in locals() and 'output_matrix_p' in locals():
        fig, axs = plt.subplots(1, 1, figsize=(12, 8))

        call_pl = output_matrix_c.T - option_purchase_price - 2 * transaction_cost
        put_pl = output_matrix_p.T - option_purchase_price - 2 * transaction_cost
        pl_options = [call_pl, put_pl]
        
        selection = 0 if trade_type == 'Call' else 1
        contract_prices = [call_price, put_price]
        
        specific_contract_pl = contract_prices[selection] - option_purchase_price - 2 * transaction_cost
        st.markdown(f'<div class="metric-container"><h3>Expected P&L given selected parameters</h3><h2>${specific_contract_pl:.2f}</h2></div>', unsafe_allow_html=True)
        
        mapping_color = sns.diverging_palette(15, 145, s=60, as_cmap=True)
        sns.heatmap(pl_options[selection], annot=True, fmt='.1f',
                    xticklabels=[str(round(i, 2)) for i in spot_prices_space], 
                    yticklabels=[str(round(i, 2)) for i in volatilities_space], 
                    ax=axs, cmap=mapping_color, center=0)
        axs.set_title(f'{trade_type} Expected P&L Heatmap', fontsize=16, pad=20)
        axs.set_xlabel('Spot Price', fontsize=12)
        axs.set_ylabel('Annualized Volatility', fontsize=12)

        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.info("Please calculate the fair value heatmaps first in Tab 1")

with tab3:
    st.write('Calculate the expected distribution of the underlying asset price, the option premium and the p&l from trading the option')
    
    with st.expander("📖 See methodology"):
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
        timeshot = st.slider("Select chart's timestamp (days/year)", 0.0, days_to_maturity / 365, days_to_maturity / 365, format="%.3f") 

    if s_selection == 'Days':
        step = days_to_maturity 
    elif s_selection == 'Hours':
        step = days_to_maturity * 24 
    elif s_selection == 'Minutes':
        step = days_to_maturity * 24 * 60 
    
    # Progress indicator for simulation
    sim_progress = st.empty()
    with sim_progress.container():
        st.markdown('<div class="progress-container">', unsafe_allow_html=True)
        st.info("🔄 Running simulation with concurrent processing...")
        sim_progress_bar = st.progress(0)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Generate simulation paths using threading
    start_sim_time = time.time()
    simulation_paths = simulate_paths(ns, days_to_maturity, step, volatility, risk_free_rate, underlying_price)
    sim_progress_bar.progress(75)
    
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
    sim_progress_bar.progress(100)
    
    sim_time = time.time() - start_sim_time
    sim_progress.empty()
    st.success(f"✅ Simulation completed in {sim_time:.2f} seconds")

    # Calculate probabilities
    otm_probability = round(sum(option_prices == 0) / len(option_prices), 3)
    itm_probability = round(1 - otm_probability, 3)
    positive_pl_proba = round(sum(pl_results > 0) / len(pl_results), 3)

    st.subheader('📊 Results')

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            f'<div class="metric-container"><h4>In-the-money probability</h4><h2>{itm_probability}</h2></div>', 
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            f'<div class="metric-container"><h4>Out-the-money probability</h4><h2>{otm_probability}</h2></div>', 
            unsafe_allow_html=True
        )
    with col3:
        st.markdown(
            f'<div class="metric-container"><h4>Positive P&L probability</h4><h2>{positive_pl_proba}</h2></div>', 
            unsafe_allow_html=True
        )

    # Create plots
    col1, col2 = st.columns(2)
    with col1:
        fig1 = plt.figure(figsize=(8, 6))
        index_to_use = -int(step - timeshot * step + 1)
        sns.histplot(simulation_paths[index_to_use, :], kde=True, stat='probability')
        plt.xlabel('Price')
        plt.axvline(selected_strike, 0, 1, color='r', label='Strike price')
        plt.title(f'Expected underlying asset price distribution at day {int(timeshot * 365)}')
        plt.legend()
        st.pyplot(fig1)

    with col2:
        fig2 = plt.figure(figsize=(8, 3))
        sns.histplot(option_prices, kde=True, stat='probability')
        plt.xlabel('Price')
        plt.title(f'Expected {trade_type} premium at day {int(timeshot * 365)}')
        st.pyplot(fig2)

        fig3 = plt.figure(figsize=(8, 3))
        sns.histplot(pl_results, kde=True, stat='probability')
        plt.xlabel('P&L')
        plt.title(f'Expected P&L distribution at day {int(timeshot * 365)}')
        st.pyplot(fig3)
