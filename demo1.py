import streamlit as st
st.set_page_config(layout="wide", page_title="Black-Scholes Options Pricing")

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from functools import partial

# Custom CSS for modern UI
st.markdown("""
<style>
    .main-header {
        padding: 2rem 0;
        text-align: center;
        border-radius: 12px;
        margin-bottom: 2rem;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .metric-container {
        background: rgba(255, 255, 255, 0.05);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
        margin-bottom: 1rem;
    }
    
    .metric-container:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(255, 255, 255, 0.05);
        padding: 4px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 16px;
        transition: all 0.3s ease;
        backdrop-filter: blur(10px);
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(255, 255, 255, 0.1);
        transform: translateY(-1px);
    }
    
    .sidebar .stSelectbox, .sidebar .stNumberInput, .sidebar .stSlider {
        margin-bottom: 1rem;
    }
    
    .stSelectbox > div > div {
        border-radius: 8px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
    }
    
    .stNumberInput > div > div > input {
        border-radius: 8px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
    }
    
    .progress-container {
        background: rgba(255, 255, 255, 0.05);
        padding: 1rem;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
        backdrop-filter: blur(10px);
    }
    
    .status-badge {
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 500;
        display: inline-block;
        backdrop-filter: blur(10px);
    }
    
    .status-success {
        background: rgba(34, 197, 94, 0.1);
        color: #22c55e;
        border: 1px solid rgba(34, 197, 94, 0.2);
    }
    
    .status-processing {
        background: rgba(59, 130, 246, 0.1);
        color: #3b82f6;
        border: 1px solid rgba(59, 130, 246, 0.2);
    }
</style>
""", unsafe_allow_html=True)

### Core Functions ######################################################
def BlackScholes(r, S, K, T, sigma, tipo='C'):
    """ 
    r : Interest Rate
    S : Spot Price
    K : Strike Price
    T : Days due expiration / 365
    sigma : Annualized Volatility 
    """
    d1 = (np.log(S/K) + (r + sigma**2/2)*T)/(sigma * np.sqrt(T)) 
    d2 = d1 - sigma * np.sqrt(T)
    
    try: 
        if tipo == 'C': 
            precio = S * norm.cdf(d1, 0, 1) - K * np.exp(-r * T) * norm.cdf(d2, 0, 1)
        elif tipo == 'P': 
            precio = K * np.exp(-r * T) * norm.cdf(-d2, 0, 1) - S * norm.cdf(-d1, 0, 1)
    except: 
        precio = 0
        
    return precio

def compute_row(i, spot_prices, volatilities, strike, interest_rate, T, option_type):
    """Compute a single row of the heatmap matrix"""
    row = np.zeros(len(volatilities))
    for j in range(len(volatilities)):
        bs_result = BlackScholes(interest_rate, spot_prices[i], strike, T, volatilities[j], option_type)
        row[j] = round(bs_result, 2)
    return i, row

def HeatMapMatrix_Parallel(spot_prices, volatilities, strike, interest_rate, days_to_exp, option_type='C', max_workers=4):
    """Parallel computation of heatmap matrix using ThreadPoolExecutor"""
    M = np.zeros((len(spot_prices), len(volatilities)))
    T = days_to_exp / 365
    
    # Use ThreadPoolExecutor for parallel computation
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Create partial function with fixed parameters
        compute_row_partial = partial(
            compute_row, 
            spot_prices=spot_prices, 
            volatilities=volatilities, 
            strike=strike, 
            interest_rate=interest_rate, 
            T=T, 
            option_type=option_type
        )
        
        # Submit all tasks
        future_to_row = {
            executor.submit(compute_row_partial, i): i 
            for i in range(len(spot_prices))
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_row):
            row_idx, row_data = future.result()
            M[row_idx] = row_data
    
    return M

def simulate_gbm_parallel(n_sims, days_to_maturity, steps, volatility, risk_free_rate, spot_price, max_workers=4):
    """Parallel simulation of Geometric Brownian Motion"""
    dt = (days_to_maturity / 365) / steps
    
    def simulate_batch(batch_size, seed):
        np.random.seed(seed)
        Z = np.random.normal(0, np.sqrt(dt), (steps, batch_size))
        paths = np.vstack([
            np.ones(batch_size), 
            np.exp((risk_free_rate - 0.5 * volatility**2) * dt + volatility * Z)
        ]).cumprod(axis=0)
        return spot_price * paths
    
    # Split simulations into batches
    batch_size = max(1, n_sims // max_workers)
    batches = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i in range(0, n_sims, batch_size):
            current_batch_size = min(batch_size, n_sims - i)
            if current_batch_size > 0:
                future = executor.submit(simulate_batch, current_batch_size, i)
                futures.append(future)
        
        for future in as_completed(futures):
            batches.append(future.result())
    
    # Combine all batches
    if batches:
        return np.hstack(batches)[:, :n_sims]  # Ensure exact number of simulations
    else:
        return simulate_batch(n_sims, 0)

###############################################################################################################
#### Sidebar Parameters ###############################################
st.sidebar.markdown("### Option Parameters")

col1, col2 = st.sidebar.columns(2)
with col1:
    underlying_price = st.number_input('Spot Price', value=100, key='spot')
with col2:
    selected_strike = st.number_input('Strike Price', value=80, key='strike')

trade_type = st.sidebar.selectbox("Contract Type", ['Call', 'Put'], key='trade_type')

col3, col4 = st.sidebar.columns(2)
with col3:
    days_to_maturity = st.number_input('Days to Maturity', value=365, key='days')
with col4:
    risk_free_rate = st.number_input('Risk-Free Rate', value=0.1, format="%.4f", key='rate')

volatility = st.sidebar.number_input('Annualized Volatility', value=0.2, format="%.4f", key='vol')

st.sidebar.markdown("### P&L Parameters")
option_purchase_price = st.sidebar.number_input("Option's Price", key='purchase_price') 
transaction_cost = st.sidebar.number_input("Opening/Closing Cost", key='transaction_cost') 

st.sidebar.markdown("### Heatmap Parameters")
col5, col6 = st.sidebar.columns(2)
with col5:
    min_spot_price = st.number_input('Min Spot', value=50, key='min_spot')
    min_vol = st.slider('Min Volatility', 0.01, 1.00, key='min_vol')
with col6:
    max_spot_price = st.number_input('Max Spot', value=110, key='max_spot')
    max_vol = st.slider('Max Volatility', 0.01, 1.00, 1.00, key='max_vol')

grid_size = st.sidebar.slider('Grid Size', 5, 20, 10, key='grid_size')

#### Variables ########################################################
spot_prices_space = np.linspace(min_spot_price, max_spot_price, grid_size)
volatilities_space = np.linspace(min_vol, max_vol, grid_size)
########################################################################

# Main Header
st.markdown('<div class="main-header">', unsafe_allow_html=True)
st.title('Black Scholes Options Heatmap')
st.markdown("Calculates an option's arbitrage-free premium using the Black Scholes option pricing model.")
st.markdown('</div>', unsafe_allow_html=True)

# Calculate current prices
call_price = BlackScholes(risk_free_rate, underlying_price, selected_strike, days_to_maturity / 365, volatility)
put_price = BlackScholes(risk_free_rate, underlying_price, selected_strike, days_to_maturity / 365, volatility, 'P')

# Display current prices with modern styling
col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    st.metric("Call Value", f"${call_price:.3f}")
    st.markdown('</div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    st.metric("Put Value", f"${put_price:.3f}")
    st.markdown('</div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    current_value = call_price if trade_type == 'Call' else put_price
    pl_current = current_value - option_purchase_price - 2 * transaction_cost
    st.metric("Current P&L", f"${pl_current:.2f}")
    st.markdown('</div>', unsafe_allow_html=True)

# Tabs for different views
tab1, tab2, tab3 = st.tabs(["Fair Value Heatmap", "P&L Heatmap", "Expected Distribution"])

with tab1:
    st.markdown("### Fair Value Analysis")
    st.write("Explore different contract values given variations in Spot Prices and Annualized Volatilities")
    
    # Progress indicator
    progress_placeholder = st.empty()
    heatmap_placeholder = st.empty()
    
    with progress_placeholder.container():
        st.markdown('<div class="progress-container">', unsafe_allow_html=True)
        st.markdown('<div class="status-badge status-processing">Computing heatmaps...</div>', unsafe_allow_html=True)
        progress_bar = st.progress(0)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Compute heatmaps in parallel
    start_time = time.time()
    
    # Update progress
    progress_bar.progress(25)
    call_matrix = HeatMapMatrix_Parallel(spot_prices_space, volatilities_space, selected_strike, risk_free_rate, days_to_maturity, 'C')
    
    progress_bar.progress(75)
    put_matrix = HeatMapMatrix_Parallel(spot_prices_space, volatilities_space, selected_strike, risk_free_rate, days_to_maturity, 'P')
    
    progress_bar.progress(100)
    computation_time = time.time() - start_time
    
    # Clear progress and show results
    progress_placeholder.empty()
    
    with heatmap_placeholder.container():
        st.markdown(f'<div class="status-badge status-success">Computed in {computation_time:.2f}s</div>', unsafe_allow_html=True)
        
        fig, axs = plt.subplots(2, 1, figsize=(20, 16))
        plt.style.use('dark_background' if st.get_option('theme.base') == 'dark' else 'default')
        
        # Call heatmap
        sns.heatmap(call_matrix.T, annot=True, fmt='.1f',
                   xticklabels=[f'{i:.1f}' for i in spot_prices_space], 
                   yticklabels=[f'{i:.2f}' for i in volatilities_space], 
                   ax=axs[0], cmap='viridis',
                   cbar_kws={'label': 'Call Value'})
        axs[0].set_title('Call Options Heatmap', fontsize=18, pad=20)
        axs[0].set_xlabel('Spot Price', fontsize=14)
        axs[0].set_ylabel('Annualized Volatility', fontsize=14)
        
        # Put heatmap
        sns.heatmap(put_matrix.T, annot=True, fmt='.1f',
                   xticklabels=[f'{i:.1f}' for i in spot_prices_space], 
                   yticklabels=[f'{i:.2f}' for i in volatilities_space], 
                   ax=axs[1], cmap='plasma',
                   cbar_kws={'label': 'Put Value'})
        axs[1].set_title('Put Options Heatmap', fontsize=18, pad=20)
        axs[1].set_xlabel('Spot Price', fontsize=14)
        axs[1].set_ylabel('Annualized Volatility', fontsize=14)
        
        plt.tight_layout()
        st.pyplot(fig)

with tab2:
    st.markdown("### Profit & Loss Analysis")
    st.write("Explore different expected P&L's from a specific contract trade given variations in the Spot Price and Annualized Volatility")
    
    if 'call_matrix' in locals() and 'put_matrix' in locals():
        fig, ax = plt.subplots(1, 1, figsize=(20, 12))
        plt.style.use('dark_background' if st.get_option('theme.base') == 'dark' else 'default')
        
        # Calculate P&L matrix
        selected_matrix = call_matrix if trade_type == 'Call' else put_matrix
        pl_matrix = selected_matrix.T - option_purchase_price - 2 * transaction_cost
        
        # Current P&L calculation
        current_option_value = call_price if trade_type == 'Call' else put_price
        current_pl = current_option_value - option_purchase_price - 2 * transaction_cost
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.metric("Expected P&L (Current Parameters)", f"${current_pl:.2f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # P&L Heatmap
        diverging_cmap = sns.diverging_palette(250, 15, s=75, as_cmap=True)
        sns.heatmap(pl_matrix, annot=True, fmt='.1f',
                   xticklabels=[f'{i:.1f}' for i in spot_prices_space], 
                   yticklabels=[f'{i:.2f}' for i in volatilities_space], 
                   ax=ax, cmap=diverging_cmap, center=0,
                   cbar_kws={'label': 'P&L ($)'})
        ax.set_title(f'{trade_type} Expected P&L Heatmap', fontsize=18, pad=20)
        ax.set_xlabel('Spot Price', fontsize=14)
        ax.set_ylabel('Annualized Volatility', fontsize=14)
        
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.info("Please compute the heatmaps in the 'Fair Value Heatmap' tab first.")

with tab3:
    st.markdown("### Monte Carlo Simulation")
    st.write('Calculate the expected distribution of the underlying asset price, the option premium and the p&l from trading the option')
    
    with st.expander("See methodology"):
        st.write('The distribution is obtained by simulating $N$ times the underlying asset price as a geometric brownian process during a specified time period.' \
        ' The function $S : [0, \infty) \mapsto [0, \infty) $ will describe the stochastic process as: ')
        st.latex('S(t) = S(0) e^{(\mu - \sigma^2 / 2)t + \sigma W(t)} ')
        st.write('Where $\mu$ is the risk free rate, $\sigma$ the annualized volatility of the asset you want to simulate and $S(0)$ the asset price at the beginning (spot price)')
    
    # Simulation parameters
    col1, col2, col3 = st.columns(3)
    with col1:
        n_sims = st.slider('Number of Simulations', 100, 10000, 1000, 50, key='n_sims')
    with col2:
        time_unit = st.selectbox('Time Interval', ['Days', 'Hours', 'Minutes'], key='time_unit',
                               help='The time interval each price point will represent. This option is merely for visual purposes.')
    with col3:
        time_fraction = st.slider("Chart Timestamp (days/year)", 0.0, days_to_maturity / 365, days_to_maturity / 365, key='time_fraction')
    
    if time_unit == 'Days':
        steps = days_to_maturity
    elif time_unit == 'Hours':
        steps = days_to_maturity * 24
    elif time_unit == 'Minutes':
        steps = days_to_maturity * 24 * 60
    
    # Run simulation button
    if st.button("Run Monte Carlo Simulation", type="primary"):
        simulation_progress = st.empty()
        
        with simulation_progress.container():
            st.markdown('<div class="progress-container">', unsafe_allow_html=True)
            st.markdown('<div class="status-badge status-processing">Running simulations...</div>', unsafe_allow_html=True)
            sim_progress = st.progress(0)
            st.markdown('</div>', unsafe_allow_html=True)
        
        start_time = time.time()
        
        # Run parallel simulation
        sim_progress.progress(50)
        simulation_paths = simulate_gbm_parallel(n_sims, days_to_maturity, steps, volatility, risk_free_rate, underlying_price)
        sim_progress.progress(100)
        
        # Calculate option prices at selected timestamp
        timestamp_index = int(steps - time_fraction * 365 * (steps / days_to_maturity) + 1)
        timestamp_index = max(0, min(timestamp_index, simulation_paths.shape[0] - 1))
        
        final_prices = simulation_paths[timestamp_index, :]
        
        if trade_type == 'Call':
            option_payoffs = np.maximum(final_prices - selected_strike, 0)
        else:
            option_payoffs = np.maximum(selected_strike - final_prices, 0)
        
        pl_results = option_payoffs - option_purchase_price - 2 * transaction_cost
        
        # Calculate probabilities
        otm_probability = np.mean(option_payoffs == 0)
        itm_probability = 1 - otm_probability
        positive_pl_probability = np.mean(pl_results > 0)
        
        simulation_time = time.time() - start_time
        simulation_progress.empty()
        
        st.markdown(f'<div class="status-badge status-success">Simulation completed in {simulation_time:.2f}s</div>', unsafe_allow_html=True)
        
        # Results metrics
        st.markdown("### Results")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.metric("In-the-Money Probability", f"{itm_probability:.2%}")
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.metric("Out-of-the-Money Probability", f"{otm_probability:.2%}")
            st.markdown('</div>', unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.metric("Positive P&L Probability", f"{positive_pl_probability:.2%}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Distribution plots
        col1, col2 = st.columns(2)
        
        with col1:
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            plt.style.use('dark_background' if st.get_option('theme.base') == 'dark' else 'default')
            
            sns.histplot(final_prices, kde=True, stat='probability', ax=ax1, alpha=0.7)
            ax1.axvline(selected_strike, color='red', linestyle='--', linewidth=2, label='Strike Price')
            ax1.set_xlabel('Asset Price')
            ax1.set_ylabel('Probability')
            ax1.set_title(f'Expected Asset Price Distribution at Day {int(time_fraction * 365)}')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            st.pyplot(fig1)
        
        with col2:
            fig2, (ax2, ax3) = plt.subplots(2, 1, figsize=(10, 8))
            plt.style.use('dark_background' if st.get_option('theme.base') == 'dark' else 'default')
            
            # Option payoff distribution
            sns.histplot(option_payoffs, kde=True, stat='probability', ax=ax2, alpha=0.7, color='orange')
            ax2.set_xlabel('Option Payoff')
            ax2.set_ylabel('Probability')
            ax2.set_title(f'Expected {trade_type} Payoff Distribution')
            ax2.grid(True, alpha=0.3)
            
            # P&L distribution
            sns.histplot(pl_results, kde=True, stat='probability', ax=ax3, alpha=0.7, color='green')
            ax3.axvline(0, color='red', linestyle='--', linewidth=2, label='Break-even')
            ax3.set_xlabel('P&L ($)')
            ax3.set_ylabel('Probability')
            ax3.set_title(f'Expected P&L Distribution')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig2)
    else:
        st.info("Click 'Run Monte Carlo Simulation' to generate probability distributions")

# Footer
st.markdown("---")
st.markdown("Black-Scholes Option Pricing Model")
