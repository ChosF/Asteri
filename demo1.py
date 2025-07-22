import streamlit as st
st.set_page_config(layout="wide", page_title="Black-Scholes Calculator")

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Thread
import threading

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
    }
    
    .metric-container {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 12px;
        padding: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.18);
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
    }
    
    .stTabs > div > div > div {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .stSidebar > div {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
        backdrop-filter: blur(10px);
    }
    
    .calculation-status {
        background: rgba(0, 123, 255, 0.1);
        border-left: 4px solid #007bff;
        padding: 0.75rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .success-status {
        background: rgba(40, 167, 69, 0.1);
        border-left: 4px solid #28a745;
        padding: 0.75rem;
        border-radius: 8px;
        margin: 1rem 0;
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
        print('Error in BlackScholes calculation')
    return precio

def calculate_row_batch(args):
    """Calculate a batch of rows for the heatmap matrix"""
    spot_prices, volatilities, strike, interest_rate, T, option_type, start_idx, end_idx = args
    
    batch_results = []
    for i in range(start_idx, end_idx):
        row_results = []
        for j, vol in enumerate(volatilities):
            bs_result = BlackScholes(interest_rate, spot_prices[i], strike, T, vol, option_type)
            row_results.append(round(bs_result, 2))
        batch_results.append(row_results)
    
    return start_idx, batch_results

def HeatMapMatrix_Threaded(spot_prices, volatilities, strike, interest_rate, days_to_exp, option_type='C'):
    """Generate heatmap matrix using multithreading for better performance"""
    T = days_to_exp / 365
    num_spots = len(spot_prices)
    num_threads = min(4, num_spots)  # Use up to 4 threads
    
    # Create thread pool and divide work
    batch_size = max(1, num_spots // num_threads)
    futures = []
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        for i in range(0, num_spots, batch_size):
            end_idx = min(i + batch_size, num_spots)
            args = (spot_prices, volatilities, strike, interest_rate, T, option_type, i, end_idx)
            future = executor.submit(calculate_row_batch, args)
            futures.append(future)
        
        # Collect results
        M = np.zeros((num_spots, len(volatilities)))
        for future in as_completed(futures):
            start_idx, batch_results = future.result()
            for idx, row in enumerate(batch_results):
                M[start_idx + idx] = row
    
    return M

@st.cache_data(ttl=300)  # Cache for 5 minutes
def cached_heatmap_calculation(spot_prices_tuple, volatilities_tuple, strike, interest_rate, days_to_exp, option_type):
    """Cached version of heatmap calculation"""
    spot_prices = np.array(spot_prices_tuple)
    volatilities = np.array(volatilities_tuple)
    return HeatMapMatrix_Threaded(spot_prices, volatilities, strike, interest_rate, days_to_exp, option_type)

###############################################################################################################

# Header with styling
st.markdown("""
<div class="main-header">
    <h1>🔢 Black Scholes Options Heatmap</h1>
    <p>Calculate an option's arbitrage-free premium using the Black Scholes option pricing model.</p>
</div>
""", unsafe_allow_html=True)

#### Sidebar Parameters ###############################################
st.sidebar.header('🎛️ Option Parameters')
with st.sidebar.expander("📊 Core Parameters", expanded=True):
    underlying_price = st.number_input('Spot Price', value=100.0, step=0.1, format="%.2f")
    trade_type = st.segmented_control("Contract type", ['Call', 'Put'], default='Call')
    selected_strike = st.number_input('Strike/Exercise Price', value=80.0, step=0.1, format="%.2f")
    days_to_maturity = st.number_input('Time to Maturity (days)', value=365, min_value=1, step=1)
    risk_free_rate = st.number_input('Risk-Free Interest Rate', value=0.1, step=0.001, format="%.3f")
    volatility = st.number_input('Annualized Volatility', value=0.2, step=0.001, format="%.3f")

st.sidebar.subheader('💰 P&L Parameters')
with st.sidebar.expander("Trading Costs", expanded=True):
    option_purchase_price = st.number_input("Option's Price", value=0.0, step=0.01, format="%.2f")
    transaction_cost = st.number_input("Opening/Closing Cost", value=0.0, step=0.01, format="%.2f")

st.sidebar.subheader('🎨 Heatmap Parameters')
with st.sidebar.expander("Grid Configuration", expanded=True):
    min_spot_price = st.number_input('Min Spot price', value=50.0, step=1.0, format="%.1f")
    max_spot_price = st.number_input('Max Spot price', value=110.0, step=1.0, format="%.1f")
    min_vol = st.slider('Min Volatility', 0.01, 1.00, 0.01, step=0.01)
    max_vol = st.slider('Max Volatility', 0.01, 1.00, 1.00, step=0.01)
    grid_size = st.slider('Grid size (n×n)', 5, 20, 10)

#### Variables ########################################################
spot_prices_space = np.linspace(min_spot_price, max_spot_price, grid_size)
volatilities_space = np.linspace(min_vol, max_vol, grid_size)

# Calculate current option prices
call_price = BlackScholes(risk_free_rate, underlying_price, selected_strike, days_to_maturity / 365, volatility)
put_price = BlackScholes(risk_free_rate, underlying_price, selected_strike, days_to_maturity / 365, volatility, 'P')

# Display current prices with modern styling
col1, col2 = st.columns(2)
with col1:
    st.markdown(f"""
    <div class="metric-container">
        <h3>📞 Call Value</h3>
        <h2 style="color: #28a745;">${round(call_price, 3):.3f}</h2>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-container">
        <h3>📉 Put Value</h3>
        <h2 style="color: #dc3545;">${round(put_price, 3):.3f}</h2>
    </div>
    """, unsafe_allow_html=True)

# Create tabs
tab1, tab2, tab3 = st.tabs(["🔥 Option's fair value heatmap", "💹 Option's P&L heatmap", "📈 Expected underlying distribution"])

with tab1:
    st.write("Explore different contract's values given variations in Spot Prices and Annualized Volatilities")
    
    # Show calculation progress
    if st.button("🔄 Calculate Heatmaps", type="primary"):
        progress_bar = st.progress(0)
        status_placeholder = st.empty()
        
        status_placeholder.markdown('<div class="calculation-status">🔄 Calculating Call options matrix...</div>', unsafe_allow_html=True)
        
        # Calculate matrices with threading
        spot_tuple = tuple(spot_prices_space)
        vol_tuple = tuple(volatilities_space)
        
        progress_bar.progress(25)
        output_matrix_c = cached_heatmap_calculation(spot_tuple, vol_tuple, selected_strike, risk_free_rate, days_to_maturity, 'C')
        
        progress_bar.progress(50)
        status_placeholder.markdown('<div class="calculation-status">🔄 Calculating Put options matrix...</div>', unsafe_allow_html=True)
        
        output_matrix_p = cached_heatmap_calculation(spot_tuple, vol_tuple, selected_strike, risk_free_rate, days_to_maturity, 'P')
        
        progress_bar.progress(75)
        status_placeholder.markdown('<div class="calculation-status">🎨 Generating visualizations...</div>', unsafe_allow_html=True)
        
        # Create visualizations
        fig, axs = plt.subplots(2, 1, figsize=(12, 16))
        
        # Call heatmap
        sns.heatmap(output_matrix_c.T, annot=True, fmt='.1f',
                    xticklabels=[str(round(i, 2)) for i in spot_prices_space],
                    yticklabels=[str(round(i, 2)) for i in volatilities_space], ax=axs[0],
                    cbar_kws={'label': 'Call Value'}, cmap='viridis')
        axs[0].set_title('📞 Call Heatmap', fontsize=16, pad=20)
        axs[0].set_xlabel('Spot Price', fontsize=12)
        axs[0].set_ylabel('Annualized Volatility', fontsize=12)
        
        # Put heatmap
        sns.heatmap(output_matrix_p.T, annot=True, fmt='.1f',
                    xticklabels=[str(round(i, 2)) for i in spot_prices_space],
                    yticklabels=[str(round(i, 2)) for i in volatilities_space], ax=axs[1],
                    cbar_kws={'label': 'Put Value'}, cmap='plasma')
        axs[1].set_title('📉 Put Heatmap', fontsize=16, pad=20)
        axs[1].set_xlabel('Spot Price', fontsize=12)
        axs[1].set_ylabel('Annualized Volatility', fontsize=12)
        
        plt.tight_layout()
        
        progress_bar.progress(100)
        status_placeholder.markdown('<div class="success-status">✅ Calculations completed successfully!</div>', unsafe_allow_html=True)
        
        st.pyplot(fig)
        
        # Store in session state for other tabs
        st.session_state.output_matrix_c = output_matrix_c
        st.session_state.output_matrix_p = output_matrix_p
        st.session_state.spot_prices_space = spot_prices_space
        st.session_state.volatilities_space = volatilities_space

with tab2:
    st.write("Explore different expected P&L's from a specific contract trade given variations in the Spot Price and Annualized Volatility")
    
    if 'output_matrix_c' in st.session_state:
        cal_contract_prices = [call_price, put_price]
        
        # Calculate P&L matrices
        call_pl = st.session_state.output_matrix_c.T - option_purchase_price - 2 * transaction_cost
        put_pl = st.session_state.output_matrix_p.T - option_purchase_price - 2 * transaction_cost
        pl_options = [call_pl, put_pl]
        
        selection = 0 if trade_type == 'Call' else 1
        specific_contract_pl = cal_contract_prices[selection] - option_purchase_price - 2 * transaction_cost
        
        # Display current P&L
        st.markdown(f"""
        <div class="metric-container">
            <h3>💰 Expected P&L (Current Parameters)</h3>
            <h2 style="color: {'#28a745' if specific_contract_pl >= 0 else '#dc3545'};">${round(specific_contract_pl, 2):.2f}</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Create P&L heatmap
        fig, axs = plt.subplots(1, 1, figsize=(12, 8))
        mapping_color = sns.diverging_palette(15, 145, s=60, as_cmap=True)
        
        sns.heatmap(pl_options[selection], annot=True, fmt='.1f',
                    xticklabels=[str(round(i, 2)) for i in st.session_state.spot_prices_space],
                    yticklabels=[str(round(i, 2)) for i in st.session_state.volatilities_space], ax=axs,
                    cmap=mapping_color, center=0, cbar_kws={'label': 'P&L ($)'})
        axs.set_title(f'💹 {trade_type} Expected P&L', fontsize=16, pad=20)
        axs.set_xlabel('Spot Price', fontsize=12)
        axs.set_ylabel('Annualized Volatility', fontsize=12)
        
        st.pyplot(fig)
    else:
        st.info("📊 Please calculate the heatmaps in the first tab to see P&L analysis.")

with tab3:
    st.write('Calculate the expected distribution of the underlying asset price, the option premium and the p&l from trading the option')
    
    with st.expander("See methodology"):
        st.write('The distribution is obtained by simulating $N$ times the underlying asset price as a geometric brownian process during a specified time period.' \
        ' The function $S : [0, \\infty) \\mapsto [0, \\infty) $ will describe the stochastic process as: ')
        st.latex('S(t) = S(0) e^{(\\mu - \\sigma^2 / 2)t + \\sigma W(t)} ')
        st.write('Where $\\mu$ is the risk free rate, $\\sigma$ the annualized volatility of the asset you want to simulate and $S(0)$ the asset price at the beginning (spot price)')
    
    # Simulation parameters
    t3_col1, t3_col2, t3_col3 = st.columns(3)
    with t3_col1:
        ns = st.slider('Number of simulations ($N$)', 100, 10000, 1000, 10)
    with t3_col2:
        s_selection = st.radio('Select time interval', ['Days', 'Hours', 'Minutes'], horizontal=True, help='The time interval each price point will represent. This option is merely for visual purposes.')
    with t3_col3:
        timeshot = st.slider("Select chart's timestamp (days/year)", 0.0, days_to_maturity / 365, days_to_maturity / 365, format="%.3f")
    
    # Calculate steps based on selection
    if s_selection == 'Days':
        step = days_to_maturity
    elif s_selection == 'Hours':
        step = days_to_maturity * 24
    elif s_selection == 'Minutes':
        step = days_to_maturity * 24 * 60
    

    @st.cache_data(ttl=300)
    def simulate_paths(ns_sims, days_mat, steps, vol, rf_rate):
        dt = (days_mat / 365) / steps
        Z = np.random.normal(0, np.sqrt(dt), (steps, ns_sims))
        paths = np.vstack([np.ones(ns_sims), np.exp((rf_rate - 0.5 * vol**2) * dt + vol * Z)]).cumprod(axis=0)
        return paths
    
    if st.button("🎲 Run Monte Carlo Simulation", type="primary"):
        with st.spinner("Running simulation..."):
            simulation_paths = underlying_price * simulate_paths(ns, days_to_maturity, step, volatility, risk_free_rate)
            
            def get_option_price(K, St, option_type='Call'):
                dynamic_index = -int(step - timeshot * 365 * (step/days_to_maturity) + 1)
                try:
                    if option_type == 'Call':
                        expiration_price = np.maximum(St[dynamic_index, :] - K, 0)
                    elif option_type == 'Put':
                        expiration_price = np.maximum(K - St[dynamic_index, :], 0)
                except:
                    print('Error in option price calculation')
                return expiration_price
            
            option_prices = get_option_price(selected_strike, simulation_paths, trade_type)
            pl_results = option_prices - option_purchase_price - 2 * transaction_cost
            
            # Calculate probabilities
            otm_probability = round(sum(option_prices == 0) / len(option_prices), 2)
            itm_probability = round(1 - otm_probability, 2)
            positive_pl_proba = round(sum(pl_results > 0) / len(pl_results), 2)
            
            # Display results
            st.subheader('📊 Simulation Results')
            
            metric_cols = st.columns(3)
            with metric_cols[0]:
                st.metric("💰 In-the-money probability", f"{itm_probability:.2%}")
            with metric_cols[1]:
                st.metric("📉 Out-the-money probability", f"{otm_probability:.2%}")
            with metric_cols[2]:
                st.metric("📈 Positive P&L probability", f"{positive_pl_proba:.2%}")
            
            # Create plots
            plot_cols = st.columns(2)
            
            with plot_cols[0]:
                fig1 = plt.figure(figsize=(10, 6))
                sns.histplot(simulation_paths[-int(step - timeshot * step + 1), :], kde=True, stat='probability', color='skyblue')
                plt.xlabel('Price ($)')
                plt.ylabel('Probability')
                plt.axvline(selected_strike, color='red', linestyle='--', label='Strike price')
                plt.title(f'Expected Underlying Asset Price Distribution\n(Day {int(timeshot * 365)})')
                plt.legend()
                plt.grid(True, alpha=0.3)
                st.pyplot(fig1)
            
            with plot_cols[1]:
                fig2 = plt.figure(figsize=(10, 3))
                sns.histplot(option_prices, kde=True, stat='probability', color='orange')
                plt.xlabel('Premium ($)')
                plt.ylabel('Probability')
                plt.title(f'Expected {trade_type} Premium Distribution\n(Day {int(timeshot * 365)})')
                plt.grid(True, alpha=0.3)
                st.pyplot(fig2)
                
                fig3 = plt.figure(figsize=(10, 3))
                colors = ['red' if x < 0 else 'green' for x in pl_results]
                sns.histplot(pl_results, kde=True, stat='probability', color='purple')
                plt.xlabel('P&L ($)')
                plt.ylabel('Probability')
                plt.title(f'Expected P&L Distribution\n(Day {int(timeshot * 365)})')
                plt.axvline(0, color='black', linestyle='--', alpha=0.7, label='Break-even')
                plt.legend()
                plt.grid(True, alpha=0.3)
                st.pyplot(fig3)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>🚀 Built with Streamlit | 📊 Powered by Black-Scholes Model | ⚡ Multi-threaded Calculations</p>
</div>
""", unsafe_allow_html=True)
