import streamlit as st
st.set_page_config(layout="wide")

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
import os
from typing import Tuple, Union, Literal

# Set NumPy to use all available cores for BLAS operations
os.environ['OMP_NUM_THREADS'] = str(os.cpu_count())
os.environ['MKL_NUM_THREADS'] = str(os.cpu_count())

# Configure matplotlib for better memory management
plt.rcParams['figure.max_open_warning'] = 0

### Optimized Core Functions ######################################################

@st.cache_data(ttl=300)  # Cache for 5 minutes
def black_scholes_vectorized(r: float, S: Union[float, np.ndarray], K: float, 
                           T: float, sigma: Union[float, np.ndarray], 
                           option_type: Literal['C', 'P'] = 'C') -> Union[float, np.ndarray]:
    """
    Vectorized Black-Scholes option pricing formula.
    
    Parameters:
    r : Interest Rate
    S : Spot Price(s) - can be array
    K : Strike Price
    T : Time to expiration in years
    sigma : Annualized Volatility - can be array
    option_type : 'C' for Call, 'P' for Put
    """
    # Convert inputs to numpy arrays for vectorization
    S = np.asarray(S)
    sigma = np.asarray(sigma)
    
    # Add small epsilon to avoid division by zero
    epsilon = 1e-8
    sigma = np.maximum(sigma, epsilon)
    T = max(T, epsilon)
    
    # Vectorized calculations
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    
    if option_type == 'C':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:  # Put option
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    return price

def compute_heatmap_chunk(args: Tuple) -> Tuple[int, int, float]:
    """Compute a single cell of the heatmap matrix."""
    i, j, spot_price, volatility, strike, interest_rate, T, option_type = args
    price = black_scholes_vectorized(interest_rate, spot_price, strike, T, volatility, option_type)
    return i, j, round(float(price), 2)

@st.cache_data(ttl=300)
def heatmap_matrix_parallel(spot_prices: np.ndarray, volatilities: np.ndarray, 
                          strike: float, interest_rate: float, days_to_exp: int, 
                          option_type: Literal['C', 'P'] = 'C') -> np.ndarray:
    """
    Compute heatmap matrix using parallel processing and vectorization.
    """
    T = days_to_exp / 365
    n_spots = len(spot_prices)
    n_vols = len(volatilities)
    
    # For small matrices, use vectorization without threading overhead
    if n_spots * n_vols <= 100:
        S_mesh, sigma_mesh = np.meshgrid(spot_prices, volatilities, indexing='ij')
        result = black_scholes_vectorized(interest_rate, S_mesh, strike, T, sigma_mesh, option_type)
        return np.round(result, 2)
    
    # For larger matrices, use parallel processing
    matrix = np.zeros((n_spots, n_vols))
    
    # Prepare arguments for parallel processing
    args_list = [
        (i, j, spot_prices[i], volatilities[j], strike, interest_rate, T, option_type)
        for i in range(n_spots) for j in range(n_vols)
    ]
    
    # Use optimal number of threads (usually CPU count)
    max_workers = min(os.cpu_count(), len(args_list))
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_args = {executor.submit(compute_heatmap_chunk, args): args for args in args_list}
        
        # Collect results
        for future in as_completed(future_to_args):
            try:
                i, j, price = future.result()
                matrix[i, j] = price
            except Exception as exc:
                st.error(f'Computation generated an exception: {exc}')
                # Use fallback calculation
                args = future_to_args[future]
                i, j = args[0], args[1]
                matrix[i, j] = 0.0
    
    return matrix

@st.cache_data(ttl=300)
def monte_carlo_simulation_vectorized(underlying_price: float, risk_free_rate: float, 
                                    volatility: float, days_to_maturity: int, 
                                    n_simulations: int, step_size: int) -> np.ndarray:
    """
    Vectorized Monte Carlo simulation for asset price paths.
    """
    dt = (days_to_maturity / 365) / step_size
    sqrt_dt = np.sqrt(dt)
    
    # Generate all random numbers at once
    Z = np.random.normal(0, sqrt_dt, (step_size, n_simulations))
    
    # Vectorized path calculation
    drift = (risk_free_rate - 0.5 * volatility**2) * dt
    diffusion = volatility * Z
    
    # Cumulative sum for path generation
    log_returns = drift + diffusion
    log_prices = np.cumsum(log_returns, axis=0)
    
    # Add initial price
    initial_log_price = np.log(underlying_price)
    paths = np.vstack([np.full(n_simulations, initial_log_price), 
                      initial_log_price + log_returns])
    
    return np.exp(paths)

@st.cache_data(ttl=300)
def calculate_option_payoffs(strike: float, final_prices: np.ndarray, 
                           option_type: Literal['Call', 'Put']) -> np.ndarray:
    """
    Calculate option payoffs at expiration.
    """
    if option_type == 'Call':
        return np.maximum(final_prices - strike, 0)
    else:  # Put
        return np.maximum(strike - final_prices, 0)

### Streamlit App Configuration ###############################################

# Sidebar parameters with better organization
st.sidebar.header('📊 Option Parameters')

# Market parameters
with st.sidebar.expander("Market Parameters", expanded=True):
    underlying_price = st.number_input('Spot Price ($)', value=100.0, min_value=0.1, step=0.1)
    trade_type = st.selectbox("Contract Type", ['Call', 'Put'], index=0)
    selected_strike = st.number_input('Strike/Exercise Price ($)', value=80.0, min_value=0.1, step=0.1)
    days_to_maturity = st.number_input('Time to Maturity (days)', value=365, min_value=1, max_value=3650)
    risk_free_rate = st.number_input('Risk-Free Interest Rate', value=0.1, min_value=0.0, max_value=1.0, step=0.01)
    volatility = st.number_input('Annualized Volatility', value=0.2, min_value=0.01, max_value=2.0, step=0.01)

# P&L parameters
with st.sidebar.expander("P&L Parameters"):
    option_purchase_price = st.number_input("Option's Purchase Price ($)", value=0.0, min_value=0.0, step=0.1)
    transaction_cost = st.number_input("Transaction Cost ($)", value=0.0, min_value=0.0, step=0.01)

# Heatmap parameters
with st.sidebar.expander("Heatmap Parameters"):
    col1, col2 = st.columns(2)
    with col1:
        min_spot_price = st.number_input('Min Spot Price', value=50.0, min_value=0.1)
        min_vol = st.slider('Min Volatility', 0.01, 0.99, 0.05, 0.01)
    with col2:
        max_spot_price = st.number_input('Max Spot Price', value=150.0, min_value=0.1)
        max_vol = st.slider('Max Volatility', 0.02, 2.0, 0.5, 0.01)
    
    grid_size = st.slider('Grid Size (n×n)', 5, 25, 10, 1)

# Ensure logical constraints
if min_spot_price >= max_spot_price:
    st.sidebar.error("Max Spot Price must be greater than Min Spot Price")
if min_vol >= max_vol:
    st.sidebar.error("Max Volatility must be greater than Min Volatility")

### Main Application Logic ####################################################

# Generate parameter spaces
spot_prices_space = np.linspace(min_spot_price, max_spot_price, grid_size)
volatilities_space = np.linspace(min_vol, max_vol, grid_size)

# Header
st.title('📈 Black-Scholes Options Heatmap')
st.markdown("Calculate option premiums using the Black-Scholes model with real-time heatmap visualization.")

# Current option prices
with st.spinner('Calculating option prices...'):
    current_call_price = black_scholes_vectorized(
        risk_free_rate, underlying_price, selected_strike, 
        days_to_maturity / 365, volatility, 'C'
    )
    current_put_price = black_scholes_vectorized(
        risk_free_rate, underlying_price, selected_strike, 
        days_to_maturity / 365, volatility, 'P'
    )

# Display current prices
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("📞 Call Value", f"${current_call_price:.3f}")
with col2:
    st.metric("📉 Put Value", f"${current_put_price:.3f}")
with col3:
    current_option_price = current_call_price if trade_type == 'Call' else current_put_price
    current_pl = current_option_price - option_purchase_price - 2 * transaction_cost
    st.metric("💰 Current P&L", f"${current_pl:.2f}", 
              delta=f"${current_pl:.2f}" if current_pl != 0 else None)
with col4:
    time_value = current_option_price - max(0, 
        underlying_price - selected_strike if trade_type == 'Call' else selected_strike - underlying_price)
    st.metric("⏰ Time Value", f"${time_value:.3f}")

# Tabs
tab1, tab2, tab3 = st.tabs(["🎯 Fair Value Heatmap", "💸 P&L Heatmap", "📊 Monte Carlo Analysis"])

# Tab 1: Fair Value Heatmap
with tab1:
    st.markdown("### Option Values vs Spot Price & Volatility")
    
    with st.spinner('Computing heatmaps...'):
        # Compute matrices in parallel
        call_matrix = heatmap_matrix_parallel(
            spot_prices_space, volatilities_space, selected_strike, 
            risk_free_rate, days_to_maturity, 'C'
        )
        put_matrix = heatmap_matrix_parallel(
            spot_prices_space, volatilities_space, selected_strike, 
            risk_free_rate, days_to_maturity, 'P'
        )
    
    # Create optimized heatmaps
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # Call heatmap
    sns.heatmap(call_matrix.T, annot=True, fmt='.1f',
                xticklabels=[f'{x:.1f}' for x in spot_prices_space], 
                yticklabels=[f'{y:.2f}' for y in volatilities_space], 
                ax=axes[0], cmap='RdYlGn', 
                cbar_kws={'label': 'Call Value ($)'})
    axes[0].set_title('Call Option Values', fontsize=16, fontweight='bold')
    axes[0].set_xlabel('Spot Price ($)', fontsize=12)
    axes[0].set_ylabel('Volatility', fontsize=12)
    
    # Put heatmap
    sns.heatmap(put_matrix.T, annot=True, fmt='.1f',
                xticklabels=[f'{x:.1f}' for x in spot_prices_space], 
                yticklabels=[f'{y:.2f}' for y in volatilities_space], 
                ax=axes[1], cmap='RdYlBu', 
                cbar_kws={'label': 'Put Value ($)'})
    axes[1].set_title('Put Option Values', fontsize=16, fontweight='bold')
    axes[1].set_xlabel('Spot Price ($)', fontsize=12)
    axes[1].set_ylabel('Volatility', fontsize=12)
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# Tab 2: P&L Heatmap
with tab2:
    st.markdown("### Expected P&L Analysis")
    
    # Select appropriate matrix
    selected_matrix = call_matrix if trade_type == 'Call' else put_matrix
    pl_matrix = selected_matrix.T - option_purchase_price - 2 * transaction_cost
    
    # Current P&L info
    st.info(f"Expected P&L for current parameters: **${current_pl:.2f}**")
    
    # P&L Heatmap
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Use diverging colormap centered at zero
    max_abs_pl = np.max(np.abs(pl_matrix))
    
    sns.heatmap(pl_matrix, annot=True, fmt='.1f',
                xticklabels=[f'{x:.1f}' for x in spot_prices_space],
                yticklabels=[f'{y:.2f}' for y in volatilities_space],
                ax=ax, cmap='RdBu_r', center=0,
                vmin=-max_abs_pl, vmax=max_abs_pl,
                cbar_kws={'label': 'P&L ($)'})
    
    ax.set_title(f'{trade_type} Option P&L Heatmap', fontsize=16, fontweight='bold')
    ax.set_xlabel('Spot Price ($)', fontsize=12)
    ax.set_ylabel('Volatility', fontsize=12)
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# Tab 3: Monte Carlo Analysis
with tab3:
    st.markdown("### Monte Carlo Simulation")
    
    with st.expander("📖 Methodology", expanded=False):
        st.markdown("""
        The simulation models the underlying asset price as a **Geometric Brownian Motion**:
        
        $$S(t) = S(0) \\cdot e^{(\\mu - \\frac{\\sigma^2}{2})t + \\sigma W(t)}$$
        
        Where:
        - $\\mu$ = risk-free rate
        - $\\sigma$ = annualized volatility  
        - $W(t)$ = Wiener process (Brownian motion)
        - $S(0)$ = initial spot price
        """)
    
    # Simulation parameters
    sim_col1, sim_col2, sim_col3 = st.columns(3)
    with sim_col1:
        n_simulations = st.slider('Simulations', 500, 20000, 5000, 500)
    with sim_col2:
        time_interval = st.selectbox('Time Interval', ['Days', 'Hours'], index=0)
    with sim_col3:
        time_snapshot = st.slider(
            "Analysis Date (% of expiry)", 0.0, 100.0, 100.0, 5.0
        ) / 100.0
    
    # Calculate step size
    if time_interval == 'Days':
        step_size = max(1, int(days_to_maturity))
    else:  # Hours
        step_size = max(24, int(days_to_maturity * 24))
    
    # Run simulation
    with st.spinner('Running Monte Carlo simulation...'):
        simulation_paths = monte_carlo_simulation_vectorized(
            underlying_price, risk_free_rate, volatility, 
            days_to_maturity, n_simulations, step_size
        )
        
        # Extract prices at specified time
        time_index = max(0, int(step_size * time_snapshot))
        final_prices = simulation_paths[time_index, :]
        
        # Calculate option payoffs and P&L
        option_payoffs = calculate_option_payoffs(selected_strike, final_prices, trade_type)
        pl_results = option_payoffs - option_purchase_price - 2 * transaction_cost
    
    # Calculate probabilities
    otm_probability = np.sum(option_payoffs == 0) / len(option_payoffs)
    itm_probability = 1 - otm_probability
    positive_pl_probability = np.sum(pl_results > 0) / len(pl_results)
    
    # Display results
    st.markdown("### 📊 Simulation Results")
    
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    with metric_col1:
        st.metric("ITM Probability", f"{itm_probability:.1%}")
    with metric_col2:
        st.metric("OTM Probability", f"{otm_probability:.1%}")
    with metric_col3:
        st.metric("Profitable Trades", f"{positive_pl_probability:.1%}")
    with metric_col4:
        expected_pl = np.mean(pl_results)
        st.metric("Expected P&L", f"${expected_pl:.2f}")
    
    # Visualization
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        
        # Asset price distribution
        ax1.hist(final_prices, bins=50, density=True, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axvline(selected_strike, color='red', linestyle='--', linewidth=2, label=f'Strike: ${selected_strike}')
        ax1.axvline(np.mean(final_prices), color='green', linestyle='--', linewidth=2, 
                   label=f'Mean: ${np.mean(final_prices):.2f}')
        
        ax1.set_xlabel('Asset Price ($)')
        ax1.set_ylabel('Probability Density')
        ax1.set_title(f'Simulated Asset Price Distribution\n(Day {int(time_snapshot * days_to_maturity)})')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        st.pyplot(fig1)
        plt.close()
    
    with chart_col2:
        fig2, (ax2, ax3) = plt.subplots(2, 1, figsize=(10, 6))
        
        # Option payoffs
        ax2.hist(option_payoffs, bins=50, density=True, alpha=0.7, color='orange', edgecolor='black')
        ax2.set_xlabel('Option Payoff ($)')
        ax2.set_ylabel('Probability Density')
        ax2.set_title(f'{trade_type} Option Payoffs')
        ax2.grid(True, alpha=0.3)
        
        # P&L distribution
        colors = ['red' if x < 0 else 'green' for x in np.histogram(pl_results, bins=50)[1][:-1]]
        n, bins, patches = ax3.hist(pl_results, bins=50, density=True, alpha=0.7, edgecolor='black')
        
        # Color bars based on profit/loss
        for i, (patch, bin_center) in enumerate(zip(patches, (bins[:-1] + bins[1:]) / 2)):
            if bin_center < 0:
                patch.set_facecolor('red')
            else:
                patch.set_facecolor('green')
        
        ax3.axvline(0, color='black', linestyle='-', linewidth=2, label='Break-even')
        ax3.axvline(expected_pl, color='blue', linestyle='--', linewidth=2, 
                   label=f'Expected: ${expected_pl:.2f}')
        
        ax3.set_xlabel('P&L ($)')
        ax3.set_ylabel('Probability Density')
        ax3.set_title('P&L Distribution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

# Performance metrics footer
with st.expander("⚡ Performance Info"):
    st.markdown(f"""
    **Optimization Features:**
    - ✅ Vectorized NumPy operations for {grid_size}×{grid_size} heatmap
    - ✅ Parallel processing using {os.cpu_count()} CPU cores
    - ✅ Streamlit caching with 5-minute TTL
    - ✅ Memory-efficient matplotlib handling
    - ✅ Optimized Monte Carlo simulation with {n_simulations:,} paths
    """)
