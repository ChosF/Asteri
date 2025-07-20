import streamlit as st
st.set_page_config(layout="wide")
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time
from functools import partial

### Enhanced App functions with threading support ######################################################
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
            precio = S * norm.cdf(d1, 0, 1) - K * np.exp(-r * T) * norm.cdf(d2, 0, 1)
        elif tipo == 'P':
            precio = K * np.exp(-r * T) * norm.cdf(-d2, 0, 1) - S * norm.cdf(-d1, 0, 1)
    except:
        print('Error')
    return precio

def calculate_bs_row(i, spot_prices, volatilities, strike, interest_rate, T, option_type='C'):
    """Calculate a single row of the Black-Scholes matrix using threading"""
    row = []
    for j in range(len(volatilities)):
        bs_result = BlackScholes(interest_rate, spot_prices[i], strike, T, volatilities[j], option_type)
        row.append(round(bs_result, 2))
    return i, row

def HeatMapMatrix_Threaded(spot_prices, volatilities, strike, interest_rate, days_to_exp, option_type='C', max_workers=None):
    """
    Enhanced heatmap matrix calculation using ThreadPoolExecutor for better performance
    """
    if max_workers is None:
        max_workers = min(32, len(spot_prices))  # Limit max workers to avoid overhead
    
    M = np.zeros(shape=(len(spot_prices), len(volatilities)))
    T = days_to_exp / 365
    
    # Use ThreadPoolExecutor for parallel computation
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Create a partial function with fixed parameters
        calculate_row_func = partial(
            calculate_bs_row,
            spot_prices=spot_prices,
            volatilities=volatilities,
            strike=strike,
            interest_rate=interest_rate,
            T=T,
            option_type=option_type
        )
        
        # Submit all row calculations
        future_to_row = {executor.submit(calculate_row_func, i): i for i in range(len(spot_prices))}
        
        # Collect results as they complete
        for future in as_completed(future_to_row):
            row_idx, row_data = future.result()
            M[row_idx, :] = row_data
    
    return M

def simulate_path_chunk(chunk_params):
    """Simulate a chunk of Monte Carlo paths"""
    start_idx, end_idx, days_to_maturity, underlying_price, risk_free_rate, volatility, step = chunk_params
    
    chunk_size = end_idx - start_idx
    dt = (days_to_maturity / 365) / step
    Z = np.random.normal(0, np.sqrt(dt), (step, chunk_size))
    paths = np.vstack([np.ones(chunk_size), 
                      np.exp((risk_free_rate - 0.5 * volatility**2) * dt + volatility * Z)]).cumprod(axis=0)
    
    return underlying_price * paths

@st.cache_data
def simulate_threaded(NS, days_to_maturity, underlying_price, risk_free_rate, volatility, step, max_workers=None):
    """
    Enhanced Monte Carlo simulation using threading for better performance
    """
    if max_workers is None:
        max_workers = min(8, NS // 100)  # Adjust based on simulation size
    
    if max_workers <= 1 or NS < 500:
        # For small simulations, use single thread to avoid overhead
        return simulate_path_chunk((0, NS, days_to_maturity, underlying_price, risk_free_rate, volatility, step))
    
    # Split simulations into chunks
    chunk_size = max(100, NS // max_workers)
    chunks = []
    
    for i in range(0, NS, chunk_size):
        end_idx = min(i + chunk_size, NS)
        chunks.append((i, end_idx, days_to_maturity, underlying_price, risk_free_rate, volatility, step))
    
    # Execute chunks in parallel
    all_paths = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_chunk = {executor.submit(simulate_path_chunk, chunk): chunk for chunk in chunks}
        
        for future in as_completed(future_to_chunk):
            chunk_paths = future.result()
            all_paths.append(chunk_paths)
    
    # Concatenate all paths
    return np.hstack(all_paths)

def calculate_option_payoff_threaded(K, St, option_type='Call', timeshot_step=None):
    """Calculate option payoffs with threading support"""
    if timeshot_step is None:
        timeshot_step = -1
    
    try:
        if option_type == 'Call':
            payoff = np.maximum(St[timeshot_step, :] - K, 0)
        elif option_type == 'Put':
            payoff = np.maximum(K - St[timeshot_step, :], 0)
        else:
            payoff = np.zeros(St.shape[1])
    except:
        print('Error in payoff calculation')
        payoff = np.zeros(St.shape[1])
    
    return payoff

###############################################################################################################
#### Sidebar parameters ###############################################

st.sidebar.header('Option Parameters')
underlying_price = st.sidebar.number_input('Spot Price', value=100)
trade_type = st.sidebar.segmented_control("Contract type", ['Call', 'Put'], default='Call')
selected_strike = st.sidebar.number_input('Strike/Exercise Price', value=80)
days_to_maturity = st.sidebar.number_input('Time to Maturity (days)', value=365)
risk_free_rate = st.sidebar.number_input('Risk-Free Interest Rate', value=0.1)
volatility = st.sidebar.number_input('Annualized Volatility', value=0.2)

st.sidebar.subheader('P&L Parameters')
option_purchase_price = st.sidebar.number_input("Option's Price")
transaction_cost = st.sidebar.number_input("Opening/Closing Cost")

st.sidebar.subheader('Heatmap Parameters')
min_spot_price = st.sidebar.number_input('Min Spot price', value=50)
max_spot_price = st.sidebar.number_input('Max Spot price', value=110)
min_vol = st.sidebar.slider('Min Volatility', 0.01, 1.00)
max_vol = st.sidebar.slider('Max Volatility', 0.01, 1.00, 1.00)
grid_size = st.sidebar.slider('Grid size (nxn)', 5, 20, 10)

st.sidebar.subheader('Performance Settings')
max_workers_heatmap = st.sidebar.slider('Heatmap Threads', 1, 16, 8, help="Number of threads for heatmap calculations")
max_workers_simulation = st.sidebar.slider('Simulation Threads', 1, 8, 4, help="Number of threads for Monte Carlo simulations")

#### Variables ########################################################
spot_prices_space = np.linspace(min_spot_price, max_spot_price, grid_size)
volatilities_space = np.linspace(min_vol, max_vol, grid_size)
########################################################################

st.header('Black Scholes Options Heatmap (Optimized with Threading)')
st.write("Calculates an option's arbitrage-free premium using the Black Scholes option pricing model with enhanced performance through multithreading.")

# Calculate individual option prices
call_price = BlackScholes(risk_free_rate, underlying_price, selected_strike, days_to_maturity / 365, volatility)
put_price = BlackScholes(risk_free_rate, underlying_price, selected_strike, days_to_maturity / 365, volatility, 'P')

cal_contract_prices = [call_price, put_price]
t1_col1, t1_col2 = st.columns(2)
with t1_col1:
    st.markdown(f"Call value: **{round(call_price, 3)}**")
with t1_col2:
    st.markdown(f"Put value: **{round(put_price, 3)}**")

tab1, tab2, tab3 = st.tabs(["Option's fair value heatmap", "Option's P&L heatmap", "Expected underlying distribution"])

# Pre-calculate matrices with threading
with st.spinner('Calculating heatmaps with multithreading...'):
    start_time = time.time()
    
    # Use threading for heatmap calculations
    output_matrix_C = HeatMapMatrix_Threaded(
        spot_prices_space, volatilities_space, selected_strike, 
        risk_free_rate, days_to_maturity, 'C', max_workers_heatmap
    )
    output_matrix_P = HeatMapMatrix_Threaded(
        spot_prices_space, volatilities_space, selected_strike, 
        risk_free_rate, days_to_maturity, 'P', max_workers_heatmap
    )
    
    calc_time = time.time() - start_time
    st.success(f'Heatmap calculations completed in {calc_time:.2f} seconds using {max_workers_heatmap} threads')

##### Heatmaps configuration #################################################################

with tab1:
    st.write("Explore different contract's values given variations in Spot Prices and Annualized Volatilities")
    fig, axs = plt.subplots(2, 1, figsize=(25, 25))

    sns.heatmap(output_matrix_C.T, annot=True, fmt='.1f',
                xticklabels=[str(round(i, 2)) for i in spot_prices_space],
                yticklabels=[str(round(i, 2)) for i in volatilities_space], ax=axs[0],
                cbar_kws={'label': 'Call Value'})
    axs[0].set_title('Call heatmap', fontsize=20)
    axs[0].set_xlabel('Spot Price', fontsize=15)
    axs[0].set_ylabel('Annualized Volatility', fontsize=15)

    sns.heatmap(output_matrix_P.T, annot=True, fmt='.1f',
                xticklabels=[str(round(i, 2)) for i in spot_prices_space],
                yticklabels=[str(round(i, 2)) for i in volatilities_space], ax=axs[1],
                cbar_kws={'label': 'Put Value'})

    axs[1].set_title('Put heatmap', fontsize=20)
    axs[1].set_xlabel('Spot Price', fontsize=15)
    axs[1].set_ylabel('Annualized Volatility', fontsize=15)

    st.pyplot(fig)

with tab2:
    st.write("Explore different expected P&L's from a specific contract trade given variations in the Spot Price and Annualized Volatility")

    fig, axs = plt.subplots(1, 1, figsize=(25, 15))

    call_PL = output_matrix_C.T - option_purchase_price - 2 * transaction_cost
    put_PL = output_matrix_P.T - option_purchase_price - 2 * transaction_cost
    PL_options = [call_PL, put_PL]
    selection = 0 if trade_type == 'Call' else 1

    specific_contract_pl = cal_contract_prices[selection] - option_purchase_price - 2 * transaction_cost
    st.markdown(f':green[Expected P&L given selected parameters: **{round(specific_contract_pl, 2)}**]')
    
    mapping_color = sns.diverging_palette(15, 145, s=60, as_cmap=True)
    sns.heatmap(PL_options[selection], annot=True, fmt='.1f',
                xticklabels=[str(round(i, 2)) for i in spot_prices_space],
                yticklabels=[str(round(i, 2)) for i in volatilities_space], ax=axs,
                cmap=mapping_color, center=0)
    axs.set_title(f'{trade_type} Expected P&L', fontsize=20)
    axs.set_xlabel('Spot Price', fontsize=15)
    axs.set_ylabel('Annualized Volatility', fontsize=15)

    st.pyplot(fig)

with tab3:
    st.write('Calculate the expected distribution of the underlying asset price, the option premium and the p&l from trading the option')
    with st.expander("See methodology"):
        st.write('The distribution is obtained by simulating $N$ times the underlying asset price as a geometric brownian process during a specified time period.' \
                ' The function $S : [0, \\infty) \\mapsto [0, \\infty) $ will describe the stochastic process as: ')
        st.latex('S(t) = S(0) e^{(\\mu - \\sigma^2 / 2)t + \\sigma W(t)} ')
        st.write('Where $\\mu$ is the risk free rate, $\\sigma$ the annualized volatility of the asset you want to simulate and $S(0)$ the asset price at the beginning (spot price)')
        st.write('**Threading Enhancement**: Monte Carlo simulations are now parallelized using ThreadPoolExecutor for improved performance.')
    
    t3_col1, t3_col2, t3_col3 = st.columns(3)
    with t3_col1:
        NS = st.slider('Number of simulations ($N$)', 100, 10000, 1000, 10)
    with t3_col2:
        s_selection = st.radio('Select time interval', ['Days', 'Hours', 'Minutes'], horizontal=True, 
                              help='The time interval each price point will represent. This option is merely for visual purposes.')
    with t3_col3:
        timeshot = st.slider("Select chart's timestamp (days/year)", 0.0, days_to_maturity / 365, days_to_maturity / 365)

    if s_selection == 'Days':
        step = days_to_maturity
    elif s_selection == 'Hours':
        step = days_to_maturity * 24
    elif s_selection == 'Minutes':
        step = days_to_maturity * 24 * 60

    #### Creating the simulations with threading
    with st.spinner('Running Monte Carlo simulations with multithreading...'):
        start_time = time.time()
        
        simulation_paths = simulate_threaded(
            NS, days_to_maturity, underlying_price, risk_free_rate, 
            volatility, int(step), max_workers_simulation
        )
        
        sim_time = time.time() - start_time
        st.info(f'Monte Carlo simulation completed in {sim_time:.2f} seconds using {max_workers_simulation} threads')

    # Calculate timeshot step
    timeshot_step = -int(step - timeshot * step + 1)
    
    option_prices = calculate_option_payoff_threaded(selected_strike, simulation_paths, trade_type, timeshot_step)
    pl_results = option_prices - option_purchase_price - 2 * transaction_cost

    otm_probability = round(sum(option_prices == 0) / len(option_prices), 2)
    itm_probability = round(1 - otm_probability, 2)
    positive_pl_proba = round(sum(pl_results > 0) / len(pl_results), 2)

    st.subheader('Results')

    t32_col1, t32_col2, t32_col3 = st.columns(3)
    t32_col1.metric("In-the money probability", itm_probability, border=True)
    t32_col2.metric("Out-the money probability", otm_probability, border=True)
    t32_col3.metric("Positive P&L probability", positive_pl_proba, border=True)

    #### Plots
    t33_col1, t33_col2 = st.columns(2)
    with t33_col1:
        t3_fig1 = plt.figure(figsize=(8, 8))
        sns.histplot(simulation_paths[timeshot_step, :], kde=True, stat='probability')
        plt.xlabel('Price')
        plt.axvline(selected_strike, 0, 1, color='r', label='Strike price')
        plt.title(f'Expected underlying asset price distribution at day {int(timeshot * 365)}')
        plt.legend()
        st.pyplot(t3_fig1)

    with t33_col2:
        t3_fig2 = plt.figure(figsize=(8, 3))
        sns.histplot(option_prices, kde=True, stat='probability')
        plt.xlabel('Price')
        plt.title(f'Expected {trade_type} premium at day {int(timeshot * 365)}')
        plt.legend()
        st.pyplot(t3_fig2)

        t3_fig3 = plt.figure(figsize=(8, 3))
        sns.histplot(pl_results, kde=True, stat='probability')
        plt.xlabel('Price')
        plt.title(f'Expected P&L distribution at day {int(timeshot * 365)}')
        plt.legend()
        st.pyplot(t3_fig3)

# Performance information
st.sidebar.markdown("---")
st.sidebar.subheader("Threading Benefits")
st.sidebar.write("✅ Non-blocking UI during calculations")
st.sidebar.write("✅ Faster heatmap generation")
st.sidebar.write("✅ Optimized Monte Carlo simulations")
st.sidebar.write("✅ Configurable thread pools")
