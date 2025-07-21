import streamlit as st
st.set_page_config(layout="wide", page_title="Black-Scholes Options Pricing")

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import os
from functools import partial
import warnings
warnings.filterwarnings('ignore')

# Configure matplotlib for better performance
plt.ioff()  # Turn off interactive mode
sns.set_theme(style="whitegrid", rc={'figure.dpi': 100})

### Optimized App Functions ######################################################

@st.cache_data(show_spinner=False)
def black_scholes_vectorized(r, S, K, T, sigma, option_type='C'):
    """
    Vectorized Black-Scholes formula
    
    Parameters can be scalars or arrays for vectorized computation
    """
    # Ensure inputs are numpy arrays for vectorization
    S = np.asarray(S)
    K = np.asarray(K)
    T = np.asarray(T)
    sigma = np.asarray(sigma)
    r = np.asarray(r)
    
    # Handle edge cases
    T = np.where(T <= 0, 1e-10, T)  # Avoid division by zero
    sigma = np.where(sigma <= 0, 1e-10, sigma)  # Avoid division by zero
    
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'C':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:  # Put option
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    return price

@st.cache_data(show_spinner=False)
def create_heatmap_matrix_vectorized(spot_prices, volatilities, strike, interest_rate, days_to_exp, option_type='C'):
    """
    Vectorized heatmap matrix computation - much faster than nested loops
    """
    T = days_to_exp / 365
    
    # Create meshgrids for vectorized computation
    S_mesh, sigma_mesh = np.meshgrid(spot_prices, volatilities, indexing='ij')
    
    # Vectorized Black-Scholes calculation
    prices = black_scholes_vectorized(
        interest_rate, S_mesh, strike, T, sigma_mesh, option_type
    )
    
    return np.round(prices, 2)

def compute_heatmap_chunk(args):
    """Helper function for parallel heatmap computation"""
    spot_chunk, volatilities, strike, interest_rate, days_to_exp, option_type = args
    return create_heatmap_matrix_vectorized(
        spot_chunk, volatilities, strike, interest_rate, days_to_exp, option_type
    )

@st.cache_data(show_spinner=False)
def create_heatmap_parallel(spot_prices, volatilities, strike, interest_rate, days_to_exp, option_type='C', max_workers=None):
    """
    Parallel heatmap computation for large grids
    """
    if max_workers is None:
        max_workers = min(4, os.cpu_count() or 1)
    
    # For smaller grids, vectorized is faster than parallel overhead
    if len(spot_prices) * len(volatilities) < 400:
        return create_heatmap_matrix_vectorized(
            spot_prices, volatilities, strike, interest_rate, days_to_exp, option_type
        )
    
    # Split spot prices into chunks for parallel processing
    chunk_size = max(1, len(spot_prices) // max_workers)
    chunks = [spot_prices[i:i + chunk_size] for i in range(0, len(spot_prices), chunk_size)]
    
    # Prepare arguments for each chunk
    chunk_args = [(chunk, volatilities, strike, interest_rate, days_to_exp, option_type) 
                  for chunk in chunks]
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(compute_heatmap_chunk, chunk_args))
    
    # Combine results
    return np.vstack(results)

@st.cache_data(show_spinner=False)
def monte_carlo_simulation_vectorized(S0, r, sigma, T, n_simulations, n_steps):
    """
    Vectorized Monte Carlo simulation - much faster than loops
    """
    dt = T / n_steps
    
    # Generate all random numbers at once
    Z = np.random.standard_normal((n_steps, n_simulations))
    
    # Vectorized path calculation using cumulative sum
    drift = (r - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt) * Z
    
    # Calculate log prices, then exponentiate
    log_returns = drift + diffusion
    log_prices = np.cumsum(log_returns, axis=0)
    
    # Add initial price and convert to actual prices
    log_prices = np.vstack([np.zeros(n_simulations), log_prices])
    prices = S0 * np.exp(log_prices)
    
    return prices

def compute_option_payoffs_vectorized(final_prices, strike, option_type):
    """Vectorized option payoff calculation"""
    if option_type == 'Call':
        return np.maximum(final_prices - strike, 0)
    else:  # Put
        return np.maximum(strike - final_prices, 0)

### Input validation functions ###
def validate_inputs(spot, strike, days, rate, vol):
    """Validate user inputs"""
    errors = []
    
    if spot <= 0:
        errors.append("Spot price must be positive")
    if strike <= 0:
        errors.append("Strike price must be positive")
    if days <= 0:
        errors.append("Days to maturity must be positive")
    if rate < -1 or rate > 1:
        errors.append("Interest rate should be between -100% and 100%")
    if vol <= 0 or vol > 5:
        errors.append("Volatility should be between 0% and 500%")
    
    return errors

### UI Components ###
def create_sidebar():
    """Create optimized sidebar with input validation"""
    st.sidebar.header('📊 Option Parameters')
    
    with st.sidebar.expander("Basic Parameters", expanded=True):
        spot_price = st.number_input('Spot Price ($)', 
                                   value=100.0, min_value=0.01, step=0.01,
                                   help="Current price of the underlying asset")
        
        option_type = st.selectbox("Contract Type", 
                                 ['Call', 'Put'], 
                                 help="Type of option contract")
        
        strike = st.number_input('Strike Price ($)', 
                               value=100.0, min_value=0.01, step=0.01,
                               help="Exercise price of the option")
        
        days_to_maturity = st.number_input('Days to Maturity', 
                                         value=30, min_value=1, max_value=3650,
                                         help="Time until option expiration")
    
    with st.sidebar.expander("Market Parameters"):
        risk_free_rate = st.number_input('Risk-Free Rate (%)', 
                                       value=5.0, min_value=-50.0, max_value=50.0, step=0.1,
                                       help="Annualized risk-free interest rate") / 100
        
        volatility = st.number_input('Volatility (%)', 
                                   value=20.0, min_value=0.1, max_value=500.0, step=0.1,
                                   help="Annualized volatility of the underlying") / 100
    
    with st.sidebar.expander("P&L Parameters"):
        purchase_price = st.number_input("Option Purchase Price ($)", 
                                       value=0.0, min_value=0.0,
                                       help="Price paid for the option")
        transaction_cost = st.number_input("Transaction Cost ($)", 
                                         value=0.0, min_value=0.0,
                                         help="Cost per trade (buy/sell)")
    
    with st.sidebar.expander("Heatmap Parameters"):
        col1, col2 = st.columns(2)
        with col1:
            min_spot = st.number_input('Min Spot', value=max(1, spot_price * 0.5))
            min_vol = st.slider('Min Vol (%)', 1, 200, 10) / 100
        with col2:
            max_spot = st.number_input('Max Spot', value=spot_price * 1.5)
            max_vol = st.slider('Max Vol (%)', 1, 200, 50) / 100
        
        grid_size = st.select_slider('Grid Size', 
                                   options=[5, 10, 15, 20, 25], 
                                   value=15,
                                   help="Larger grids take more time to compute")
    
    with st.sidebar.expander("Simulation Parameters"):
        n_simulations = st.select_slider('Simulations', 
                                       options=[100, 500, 1000, 2500, 5000, 10000],
                                       value=1000,
                                       help="More simulations = better accuracy but slower")
        time_selection = st.radio('Time Interval', ['Days', 'Hours'], horizontal=True)
    
    return (spot_price, option_type, strike, days_to_maturity, risk_free_rate, volatility,
            purchase_price, transaction_cost, min_spot, max_spot, min_vol, max_vol, 
            grid_size, n_simulations, time_selection)

### Main App ###
def main():
    st.title('⚡ Black-Scholes Options Pricing Dashboard')
    st.markdown("*Optimized for performance with vectorized calculations and parallel processing*")
    
    # Get inputs from sidebar
    (spot_price, option_type, strike, days_to_maturity, risk_free_rate, volatility,
     purchase_price, transaction_cost, min_spot, max_spot, min_vol, max_vol, 
     grid_size, n_simulations, time_selection) = create_sidebar()
    
    # Validate inputs
    errors = validate_inputs(spot_price, strike, days_to_maturity, risk_free_rate, volatility)
    if errors:
        st.error("❌ Input Validation Errors:")
        for error in errors:
            st.error(f"• {error}")
        return
    
    # Calculate option prices
    with st.spinner("Computing option prices..."):
        T = days_to_maturity / 365
        call_price = black_scholes_vectorized(risk_free_rate, spot_price, strike, T, volatility, 'C')
        put_price = black_scholes_vectorized(risk_free_rate, spot_price, strike, T, volatility, 'P')
    
    # Display current prices
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📈 Call Price", f"${call_price:.3f}")
    with col2:
        st.metric("📉 Put Price", f"${put_price:.3f}")
    with col3:
        current_price = call_price if option_type == 'Call' else put_price
        pl = current_price - purchase_price - 2 * transaction_cost
        st.metric("💰 Current P&L", f"${pl:.2f}", delta=f"{pl:.2f}")
    with col4:
        time_value = current_price - max(0, spot_price - strike if option_type == 'Call' else strike - spot_price)
        st.metric("⏰ Time Value", f"${time_value:.3f}")
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["🌡️ Options Heatmap", "💹 P&L Analysis", "📊 Monte Carlo Simulation"])
    
    with tab1:
        st.subheader("Option Values vs Spot Price & Volatility")
        
        with st.spinner("Generating heatmap..."):
            # Create price and volatility ranges
            spot_range = np.linspace(min_spot, max_spot, grid_size)
            vol_range = np.linspace(min_vol, max_vol, grid_size)
            
            # Compute heatmap using parallel processing for large grids
            heatmap_data = create_heatmap_parallel(
                spot_range, vol_range, strike, risk_free_rate, 
                days_to_maturity, option_type
            )
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 8))
        
        sns.heatmap(heatmap_data.T, 
                    annot=grid_size <= 15,  # Only show annotations for smaller grids
                    fmt='.1f',
                    xticklabels=[f'{x:.0f}' for x in spot_range[::max(1, len(spot_range)//10)]],
                    yticklabels=[f'{y:.0%}' for y in vol_range[::max(1, len(vol_range)//10)]],
                    ax=ax,
                    cmap='viridis',
                    cbar_kws={'label': f'{option_type} Option Value ($)'})
        
        ax.set_title(f'{option_type} Option Heatmap (Strike: ${strike})', fontsize=16, pad=20)
        ax.set_xlabel('Spot Price ($)', fontsize=12)
        ax.set_ylabel('Volatility', fontsize=12)
        
        # Add current position marker
        spot_idx = np.argmin(np.abs(spot_range - spot_price))
        vol_idx = np.argmin(np.abs(vol_range - volatility))
        ax.scatter(spot_idx, vol_idx, color='red', s=100, marker='x', linewidth=3)
        ax.text(spot_idx, vol_idx-0.5, 'Current', ha='center', va='top', color='red', fontweight='bold')
        
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    
    with tab2:
        st.subheader("Profit & Loss Analysis")
        
        if purchase_price == 0:
            st.info("💡 Set an option purchase price in the sidebar to see P&L analysis")
        else:
            with st.spinner("Computing P&L heatmap..."):
                # Reuse heatmap data and compute P&L
                spot_range = np.linspace(min_spot, max_spot, grid_size)
                vol_range = np.linspace(min_vol, max_vol, grid_size)
                
                option_values = create_heatmap_parallel(
                    spot_range, vol_range, strike, risk_free_rate, 
                    days_to_maturity, option_type
                )
                
                pl_data = option_values - purchase_price - 2 * transaction_cost
            
            # P&L metrics
            current_pl = (call_price if option_type == 'Call' else put_price) - purchase_price - 2 * transaction_cost
            max_profit = np.max(pl_data)
            max_loss = np.min(pl_data)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Current P&L", f"${current_pl:.2f}")
            col2.metric("Max Potential Profit", f"${max_profit:.2f}")
            col3.metric("Max Potential Loss", f"${max_loss:.2f}")
            
            # P&L Heatmap
            fig, ax = plt.subplots(figsize=(12, 8))
            
            sns.heatmap(pl_data.T,
                       annot=grid_size <= 15,
                       fmt='.1f',
                       xticklabels=[f'{x:.0f}' for x in spot_range[::max(1, len(spot_range)//10)]],
                       yticklabels=[f'{y:.0%}' for y in vol_range[::max(1, len(vol_range)//10)]],
                       ax=ax,
                       cmap='RdYlGn',
                       center=0,
                       cbar_kws={'label': 'P&L ($)'})
            
            ax.set_title(f'{option_type} Option P&L Analysis', fontsize=16, pad=20)
            ax.set_xlabel('Spot Price ($)', fontsize=12)
            ax.set_ylabel('Volatility', fontsize=12)
            
            # Add current position marker
            spot_idx = np.argmin(np.abs(spot_range - spot_price))
            vol_idx = np.argmin(np.abs(vol_range - volatility))
            ax.scatter(spot_idx, vol_idx, color='blue', s=100, marker='o', linewidth=2)
            
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
    
    with tab3:
        st.subheader("Monte Carlo Simulation")
        
        with st.expander("📖 Methodology"):
            st.markdown("""
            **Geometric Brownian Motion Simulation:**
            
            The simulation models the underlying asset price as:
            
            $$S(t) = S(0) \\cdot e^{(r - \\frac{\\sigma^2}{2})t + \\sigma W(t)}$$
            
            Where:
            - $S(0)$ = Current spot price
            - $r$ = Risk-free rate  
            - $\\sigma$ = Volatility
            - $W(t)$ = Wiener process (Brownian motion)
            """)
        
        # Simulation parameters
        col1, col2, col3 = st.columns(3)
        with col1:
            evaluation_day = st.slider("Evaluation Day", 0, days_to_maturity, days_to_maturity, 
                                     help="Day to evaluate option value")
        with col2:
            confidence_level = st.select_slider("Confidence Level", [90, 95, 99], value=95)
        
        with st.spinner(f"Running {n_simulations:,} Monte Carlo simulations..."):
            # Determine time steps based on selection
            if time_selection == 'Hours':
                n_steps = days_to_maturity * 24
            else:
                n_steps = days_to_maturity
            
            # Run vectorized simulation
            T = days_to_maturity / 365
            price_paths = monte_carlo_simulation_vectorized(
                spot_price, risk_free_rate, volatility, T, n_simulations, n_steps
            )
            
            # Get evaluation day index
            eval_idx = int((evaluation_day / days_to_maturity) * n_steps)
            final_prices = price_paths[eval_idx, :]
            
            # Calculate option payoffs
            option_payoffs = compute_option_payoffs_vectorized(final_prices, strike, option_type)
            
            # Calculate P&L
            pl_results = option_payoffs - purchase_price - 2 * transaction_cost
        
        # Simulation results
        itm_prob = np.mean(option_payoffs > 0)
        positive_pl_prob = np.mean(pl_results > 0) if purchase_price > 0 else 0
        
        # Confidence intervals
        price_percentiles = np.percentile(final_prices, [50-confidence_level/2, 50, 50+confidence_level/2])
        
        # Display metrics
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        metric_col1.metric("ITM Probability", f"{itm_prob:.1%}")
        metric_col2.metric("Positive P&L Prob", f"{positive_pl_prob:.1%}")
        metric_col3.metric("Expected Price", f"${np.mean(final_prices):.2f}")
        metric_col4.metric(f"{confidence_level}% Price Range", 
                          f"${price_percentiles[0]:.0f} - ${price_percentiles[2]:.0f}")
        
        # Create visualizations
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Price distribution
        ax1.hist(final_prices, bins=50, alpha=0.7, density=True, color='skyblue', edgecolor='black')
        ax1.axvline(strike, color='red', linestyle='--', linewidth=2, label=f'Strike: ${strike}')
        ax1.axvline(spot_price, color='green', linestyle='--', linewidth=2, label=f'Current: ${spot_price}')
        ax1.set_title(f'Price Distribution (Day {evaluation_day})')
        ax1.set_xlabel('Price ($)')
        ax1.set_ylabel('Density')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Option payoff distribution
        ax2.hist(option_payoffs, bins=50, alpha=0.7, density=True, color='lightgreen', edgecolor='black')
        ax2.set_title(f'{option_type} Option Payoff Distribution')
        ax2.set_xlabel('Payoff ($)')
        ax2.set_ylabel('Density')
        ax2.grid(True, alpha=0.3)
        
        # P&L distribution (if purchase price is set)
        if purchase_price > 0:
            ax3.hist(pl_results, bins=50, alpha=0.7, density=True, 
                    color=['red' if x < 0 else 'green' for x in np.histogram(pl_results, bins=50)[1][:-1]], 
                    edgecolor='black')
            ax3.axvline(0, color='black', linestyle='-', linewidth=2)
            ax3.set_title('P&L Distribution')
            ax3.set_xlabel('P&L ($)')
            ax3.set_ylabel('Density')
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'Set purchase price\nto see P&L analysis', 
                    ha='center', va='center', transform=ax3.transAxes, fontsize=12)
            ax3.set_title('P&L Distribution')
        
        # Sample price paths
        sample_paths = price_paths[:, :min(100, n_simulations)]  # Show max 100 paths
        time_axis = np.linspace(0, days_to_maturity, n_steps + 1)
        
        ax4.plot(time_axis, sample_paths, alpha=0.1, color='blue')
        ax4.plot(time_axis, np.mean(price_paths, axis=1), color='red', linewidth=2, label='Mean Path')
        ax4.axhline(strike, color='red', linestyle='--', linewidth=2, label=f'Strike: ${strike}')
        ax4.axvline(evaluation_day, color='orange', linestyle=':', linewidth=2, label=f'Eval Day: {evaluation_day}')
        ax4.set_title('Sample Price Paths')
        ax4.set_xlabel('Days')
        ax4.set_ylabel('Price ($)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

if __name__ == "__main__":
    main()
