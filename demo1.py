import streamlit as st
st.set_page_config(layout="wide")
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from functools import lru_cache
import os

# Optimize matplotlib for performance
plt.style.use('fast')
plt.rcParams['figure.max_open_warning'] = 0

# Automatically determine optimal thread count
def get_optimal_threads():
    """Automatically determine optimal number of threads based on system resources."""
    cpu_count = os.cpu_count() or 4
    # For CPU-bound tasks, use CPU count; for mixed workloads, use CPU count + 2
    return min(max(cpu_count - 1, 2), 8)  # Cap at 8 threads to avoid overhead

OPTIMAL_THREADS = get_optimal_threads()

### Optimized App functions ######################################################
@lru_cache(maxsize=1000)
def blackscholes_cached(r, S, K, T, sigma, option_type='C'):
    """
    Cached Black-Scholes computation with optimized calculations.
    Uses LRU cache to avoid recomputation of identical parameters.
    """
    if T <= 0 or sigma <= 0:
        return 0.0
    
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    
    if option_type == 'C':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:  # Put option
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def blackscholes_vectorized(r, S_array, K, T, sigma_array, option_type='C'):
    """
    Vectorized Black-Scholes computation for arrays of spot prices and volatilities.
    Much faster than loops for matrix calculations.
    """
    if T <= 0:
        return np.zeros((len(S_array), len(sigma_array)))
    
    # Create meshgrids for vectorized computation
    S_grid, sigma_grid = np.meshgrid(S_array, sigma_array, indexing='ij')
    
    # Vectorized calculations
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S_grid/K) + (r + 0.5 * sigma_grid**2) * T) / (sigma_grid * sqrt_T)
    d2 = d1 - sigma_grid * sqrt_T
    
    if option_type == 'C':
        prices = S_grid * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:  # Put option
        prices = K * np.exp(-r * T) * norm.cdf(-d2) - S_grid * norm.cdf(-d1)
    
    return np.round(prices, 2)

def simulate_paths_optimized(S0, r, sigma, T, n_paths, n_steps, random_seed=None):
    """
    Optimized Monte Carlo path simulation using vectorized operations.
    
    Parameters:
    - S0: Initial stock price
    - r: Risk-free rate
    - sigma: Volatility
    - T: Time to expiration
    - n_paths: Number of simulation paths
    - n_steps: Number of time steps
    - random_seed: Random seed for reproducibility
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)
    
    # Pre-allocate arrays for better memory usage
    paths = np.zeros((n_steps + 1, n_paths))
    paths[0] = S0
    
    # Vectorized random number generation
    randoms = np.random.normal(0, 1, (n_steps, n_paths))
    
    # Vectorized path calculation
    drift = (r - 0.5 * sigma**2) * dt
    diffusion = sigma * sqrt_dt * randoms
    
    # Calculate cumulative returns and apply to paths
    log_returns = drift + diffusion
    log_prices = np.cumsum(log_returns, axis=0)
    paths[1:] = S0 * np.exp(log_prices)
    
    return paths

def parallel_option_pricing(spot_prices, volatilities, K, r, T, option_type='C'):
    """
    Parallel computation of option prices using thread pool.
    Divides computation across available CPU cores.
    """
    def compute_chunk(spot_chunk, vol_array):
        return blackscholes_vectorized(r, spot_chunk, K, T, vol_array, option_type)
    
    # Split spot prices into chunks for parallel processing
    n_chunks = min(OPTIMAL_THREADS, len(spot_prices))
    chunk_size = len(spot_prices) // n_chunks
    chunks = [spot_prices[i:i + chunk_size] for i in range(0, len(spot_prices), chunk_size)]
    
    results = []
    with ThreadPoolExecutor(max_workers=OPTIMAL_THREADS) as executor:
        futures = [executor.submit(compute_chunk, chunk, volatilities) for chunk in chunks]
        
        for future in as_completed(futures):
            results.append(future.result())
    
    return np.vstack(results)

@st.cache_data(ttl=300)  # Cache for 5 minutes
def compute_heatmap_matrix(spot_prices, volatilities, strike, interest_rate, days_to_exp, option_type='C'):
    """
    Cached heatmap matrix computation with automatic cache invalidation.
    """
    T = days_to_exp / 365
    
    if len(spot_prices) * len(volatilities) > 400:  # Use parallel for large matrices
        return parallel_option_pricing(spot_prices, volatilities, strike, interest_rate, T, option_type)
    else:
        return blackscholes_vectorized(interest_rate, spot_prices, strike, T, volatilities, option_type)

@st.cache_data(ttl=300, max_entries=10)
def generate_monte_carlo_simulation(underlying_price, risk_free_rate, volatility, days_to_maturity, 
                                  selected_strike, trade_type, n_simulations, timeshot_ratio):
    """
    Cached Monte Carlo simulation with optimized computation.
    """
    T = days_to_maturity / 365
    timeshot_T = timeshot_ratio * T
    n_steps = max(int(days_to_maturity), 50)  # Minimum 50 steps for accuracy
    
    # Generate paths using optimized function
    paths = simulate_paths_optimized(underlying_price, risk_free_rate, volatility, 
                                   T, n_simulations, n_steps, random_seed=42)
    
    # Calculate time index for timeshot
    time_index = int(timeshot_ratio * n_steps)
    final_prices = paths[time_index]
    
    # Vectorized option payoff calculation
    if trade_type == 'Call':
        option_payoffs = np.maximum(final_prices - selected_strike, 0)
    else:  # Put
        option_payoffs = np.maximum(selected_strike - final_prices, 0)
    
    return final_prices, option_payoffs

#### Sidebar parameters (Optimized with better organization) ###############################################
st.sidebar.header('📊 Option Parameters')

# Market Data Section
with st.sidebar.expander("🏢 Market Data", expanded=True):
    underlying_price = st.number_input('Spot Price ($)', value=100.0, min_value=0.1, step=1.0)
    trade_type = st.segmented_control("Contract Type", ['Call', 'Put'], default='Call')
    selected_strike = st.number_input('Strike/Exercise Price ($)', value=80.0, min_value=0.1, step=1.0)

# Time and Risk Section
with st.sidebar.expander("⏱️ Time & Risk Parameters", expanded=True):
    days_to_maturity = st.number_input('Days to Maturity', value=365, min_value=1, max_value=3650, step=1)
    risk_free_rate = st.number_input('Risk-Free Rate (%)', value=10.0, min_value=0.0, max_value=100.0, step=0.1) / 100
    volatility = st.number_input('Annualized Volatility (%)', value=20.0, min_value=0.1, max_value=200.0, step=0.1) / 100

# Trading Parameters
with st.sidebar.expander("💰 Trading Parameters"):
    option_purchase_price = st.number_input("Option's Purchase Price ($)", value=0.0, min_value=0.0, step=0.01)
    transaction_cost = st.number_input("Transaction Cost ($)", value=0.0, min_value=0.0, step=0.01)

# Heatmap Configuration
with st.sidebar.expander("🎯 Heatmap Configuration"):
    col1, col2 = st.columns(2)
    with col1:
        min_spot_price = st.number_input('Min Spot ($)', value=50.0, min_value=0.1)
        min_vol = st.slider('Min Vol (%)', 1, 100, 5) / 100
    with col2:
        max_spot_price = st.number_input('Max Spot ($)', value=110.0, min_value=min_spot_price + 1)
        max_vol = st.slider('Max Vol (%)', 1, 200, 100) / 100
    
    grid_size = st.slider('Grid Resolution', 5, 25, 10, 
                         help="Higher resolution = better detail but slower computation")

# Monte Carlo Configuration
with st.sidebar.expander("🎲 Monte Carlo Parameters"):
    n_simulations = st.slider('Simulations', 100, 20000, 1000, step=100,
                             help="More simulations = higher accuracy but slower computation")
    time_interval = st.radio('Time Interval Display', ['Days', 'Hours', 'Minutes'], 
                           horizontal=True, help='Visual display unit only')
    timeshot = st.slider("Analysis Timepoint", 0.0, 1.0, 1.0, step=0.01,
                        help="Fraction of time to maturity (0=now, 1=expiration)")

# Performance indicator
st.sidebar.info(f"🚀 Using {OPTIMAL_THREADS} threads for optimal performance")

#### Main Application ########################################################
st.header('⚡ Optimized Black-Scholes Options Analytics')
st.write("High-performance option pricing with real-time analysis and Monte Carlo simulations.")

# Pre-compute arrays
spot_prices_space = np.linspace(min_spot_price, max_spot_price, grid_size)
volatilities_space = np.linspace(min_vol, max_vol, grid_size)

# Calculate current option values
with st.spinner('Computing option values...'):
    call_price = blackscholes_cached(risk_free_rate, underlying_price, selected_strike, 
                                   days_to_maturity / 365, volatility, 'C')
    put_price = blackscholes_cached(risk_free_rate, underlying_price, selected_strike, 
                                  days_to_maturity / 365, volatility, 'P')

# Display current values with better formatting
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Call Option Value", f"${call_price:.3f}")
with col2:
    st.metric("Put Option Value", f"${put_price:.3f}")
with col3:
    selected_price = call_price if trade_type == 'Call' else put_price
    pl_current = selected_price - option_purchase_price - 2 * transaction_cost
    st.metric("Current P&L", f"${pl_current:.2f}", 
             delta=f"${pl_current:.2f}", delta_color="normal")

# Tabs for different analyses
tab1, tab2, tab3 = st.tabs(["🎯 Option Pricing Heatmap", "📈 P&L Analysis", "🎲 Monte Carlo Simulation"])

with tab1:
    st.subheader("Option Value Sensitivity Analysis")
    st.write("Explore how option values change with spot price and volatility variations.")
    
    # Only compute the needed matrix based on trade type
    with st.spinner('Computing heatmap matrices...'):
        if trade_type == 'Call':
            output_matrix = compute_heatmap_matrix(spot_prices_space, volatilities_space, 
                                                 selected_strike, risk_free_rate, 
                                                 days_to_maturity, 'C')
            title = 'Call Option'
            color_label = 'Call Value ($)'
        else:
            output_matrix = compute_heatmap_matrix(spot_prices_space, volatilities_space, 
                                                 selected_strike, risk_free_rate, 
                                                 days_to_maturity, 'P')
            title = 'Put Option'
            color_label = 'Put Value ($)'
    
    # Create optimized heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    
    sns.heatmap(output_matrix.T, annot=True, fmt='.1f',
                xticklabels=[f"{x:.1f}" for x in spot_prices_space],
                yticklabels=[f"{y:.2f}" for y in volatilities_space],
                cbar_kws={'label': color_label}, ax=ax)
    
    ax.set_title(f'{title} Value Heatmap', fontsize=16, fontweight='bold')
    ax.set_xlabel('Spot Price ($)', fontsize=12)
    ax.set_ylabel('Annualized Volatility', fontsize=12)
    
    st.pyplot(fig)
    plt.close(fig)  # Explicit cleanup

with tab2:
    st.subheader("Profit & Loss Analysis")
    st.write("Expected P&L analysis across different market scenarios.")
    
    # Calculate P&L matrix
    with st.spinner('Computing P&L analysis...'):
        if trade_type == 'Call':
            option_matrix = compute_heatmap_matrix(spot_prices_space, volatilities_space, 
                                                 selected_strike, risk_free_rate, 
                                                 days_to_maturity, 'C')
        else:
            option_matrix = compute_heatmap_matrix(spot_prices_space, volatilities_space, 
                                                 selected_strike, risk_free_rate, 
                                                 days_to_maturity, 'P')
        
        pl_matrix = option_matrix.T - option_purchase_price - 2 * transaction_cost
    
    # Display expected P&L for current parameters
    st.success(f"Expected P&L at current parameters: **${pl_current:.2f}**")
    
    # Create P&L heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Use diverging colormap centered at zero
    colormap = sns.diverging_palette(10, 150, as_cmap=True)
    
    sns.heatmap(pl_matrix, annot=True, fmt='.1f',
                xticklabels=[f"{x:.1f}" for x in spot_prices_space],
                yticklabels=[f"{y:.2f}" for y in volatilities_space],
                cmap=colormap, center=0, ax=ax,
                cbar_kws={'label': 'Expected P&L ($)'})
    
    ax.set_title(f'{trade_type} Expected P&L Heatmap', fontsize=16, fontweight='bold')
    ax.set_xlabel('Spot Price ($)', fontsize=12)
    ax.set_ylabel('Annualized Volatility', fontsize=12)
    
    st.pyplot(fig)
    plt.close(fig)

with tab3:
    st.subheader("Monte Carlo Simulation Analysis")
    st.write('Statistical analysis of expected outcomes using Monte Carlo methods.')
    
    with st.expander("📚 Methodology", expanded=False):
        st.write('The distribution is obtained by simulating **N** times the underlying asset price as a geometric Brownian motion:')
        st.latex(r'S(t) = S(0) \cdot e^{(\mu - \sigma^2 / 2)t + \sigma W(t)}')
        st.write('Where μ is the risk-free rate, σ is the annualized volatility, and W(t) is a Wiener process.')
    
    # Run Monte Carlo simulation
    with st.spinner('Running Monte Carlo simulation...'):
        final_prices, option_payoffs = generate_monte_carlo_simulation(
            underlying_price, risk_free_rate, volatility, days_to_maturity,
            selected_strike, trade_type, n_simulations, timeshot
        )
        
        pl_results = option_payoffs - option_purchase_price - 2 * transaction_cost
    
    # Calculate probabilities
    otm_probability = np.mean(option_payoffs == 0)
    itm_probability = 1 - otm_probability
    positive_pl_probability = np.mean(pl_results > 0)
    
    # Display key metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("In-the money probability", f"{itm_probability:.2f}")
    col2.metric("Out-the money probability", f"{otm_probability:.2f}")
    col3.metric("Positive P&L probability", f"{positive_pl_probability:.2f}")
    col4.metric("Average P&L", f"${np.mean(pl_results):.2f}")
    
    # Create distribution plots
    col1, col2 = st.columns(2)
    
    with col1:
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        sns.histplot(final_prices, kde=True, stat='probability', ax=ax1, alpha=0.7)
        ax1.axvline(selected_strike, color='red', linestyle='--', 
                   label=f'Strike: ${selected_strike}', linewidth=2)
        ax1.set_xlabel('Stock Price ($)')
        ax1.set_ylabel('Probability')
        ax1.set_title(f'Price Distribution at t={timeshot:.2f}T')
        ax1.legend()
        st.pyplot(fig1)
        plt.close(fig1)
    
    with col2:
        fig2, (ax2, ax3) = plt.subplots(2, 1, figsize=(8, 6))
        
        # Option payoff distribution
        sns.histplot(option_payoffs, kde=True, stat='probability', ax=ax2, alpha=0.7)
        ax2.set_xlabel('Option Payoff ($)')
        ax2.set_ylabel('Probability')
        ax2.set_title(f'{trade_type} Payoff Distribution')
        
        # P&L distribution
        sns.histplot(pl_results, kde=True, stat='probability', ax=ax3, alpha=0.7, 
                    color='green' if np.mean(pl_results) > 0 else 'red')
        ax3.axvline(0, color='black', linestyle='--', alpha=0.7)
        ax3.set_xlabel('P&L ($)')
        ax3.set_ylabel('Probability')
        ax3.set_title('P&L Distribution')
        
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

# Footer with performance info
st.divider()
st.caption(f"⚡ Optimized for performance • Using {OPTIMAL_THREADS} CPU threads • Cached computations • Vectorized operations")
