import streamlit as st
st.set_page_config(layout="wide")
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp
from functools import partial

### Optimized App functions ######################################################

def BlackScholes(r, S, K, T, sigma, tipo='C'):
    """ 
    Vectorized Black-Scholes function that can handle arrays
    r : Interest Rate
    S : Spot Price (can be array)
    K : Strike Price (can be array) 
    T : Time to expiration (can be array)
    sigma : Annualized Volatility (can be array)
    """
    # Ensure all inputs are numpy arrays for vectorized operations
    S = np.asarray(S)
    K = np.asarray(K)
    T = np.asarray(T)
    sigma = np.asarray(sigma)
    r = np.asarray(r)
    
    # Vectorized Black-Scholes calculation
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if tipo == 'C':
        precio = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif tipo == 'P':
        precio = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    return precio

@st.cache_data
def HeatMapMatrix_Optimized(Spot_Prices, Volatilities, Strike, Interest_Rate, Days_to_Exp, tipo='C'):
    """
    Fully vectorized heatmap matrix generation using NumPy broadcasting
    """
    # Convert to numpy arrays
    Spot_Prices = np.asarray(Spot_Prices)
    Volatilities = np.asarray(Volatilities)
    
    T = Days_to_Exp / 365
    
    # Create meshgrid for broadcasting - this eliminates all loops
    S_grid, Vol_grid = np.meshgrid(Spot_Prices, Volatilities, indexing='ij')
    
    # Vectorized Black-Scholes calculation for entire matrix at once
    BS_results = BlackScholes(Interest_Rate, S_grid, Strike, T, Vol_grid, tipo)
    
    return np.round(BS_results, 2)

@st.cache_data
def simulate_optimized(NS, days_to_maturity, step, volatility, Risk_Free_Rate):
    """
    Optimized Monte Carlo simulation using vectorized operations
    """
    dt = (days_to_maturity / 365) / step
    
    # Pre-allocate arrays for better memory efficiency
    paths = np.empty((step + 1, NS))
    paths[0] = 1.0  # Initial value
    
    # Generate all random numbers at once
    Z = np.random.normal(0, np.sqrt(dt), (step, NS))
    
    # Vectorized calculation of price paths
    drift_term = (Risk_Free_Rate - 0.5 * volatility**2) * dt
    diffusion_term = volatility * Z
    
    # Use cumsum for cumulative product calculation
    log_returns = drift_term + diffusion_term
    paths[1:] = np.exp(np.cumsum(log_returns, axis=0))
    
    return paths

def get_Option_Price_vectorized(K, St, timeshot, days_to_maturity, step, tipo='Call'):
    """
    Vectorized option price calculation
    """
    # Calculate the correct time index
    time_ratio = timeshot * 365 / days_to_maturity
    time_index = int(step * time_ratio)
    time_index = min(time_index, step - 1)  # Ensure we don't exceed array bounds
    
    # Get prices at the specified time
    prices_at_time = St[time_index, :]
    
    if tipo == 'Call':
        option_values = np.maximum(prices_at_time - K, 0)
    elif tipo == 'Put':
        option_values = np.maximum(K - prices_at_time, 0)
    
    return option_values

#### Sidebar parameters (unchanged) ###############################################
st.sidebar.header('Option Parameters')
Underlying_price = st.sidebar.number_input('Spot Price', value=100)
trade_type = st.sidebar.segmented_control("Contract type", ['Call', 'Put'], default='Call')
SelectedStrike = st.sidebar.number_input('Strike/Exercise Price', value=80)
days_to_maturity = st.sidebar.number_input('Time to Maturity (days)', value=365)
Risk_Free_Rate = st.sidebar.number_input('Risk-Free Interest Rate', value=0.1)
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

#### Variables ########################################################
SpotPrices_space = np.linspace(min_spot_price, max_spot_price, grid_size)
Volatilities_space = np.linspace(min_vol, max_vol, grid_size)

########################################################################

st.header('Black Scholes options heatmap')
st.write("Calculates an option's arbitrage-free premium using the Black Scholes option pricing model.")

# Calculate current option prices
call_price = BlackScholes(Risk_Free_Rate, Underlying_price, SelectedStrike, days_to_maturity / 365, volatility)
put_price = BlackScholes(Risk_Free_Rate, Underlying_price, SelectedStrike, days_to_maturity / 365, volatility, 'P')

cal_contract_prices = [call_price, put_price]
t1_col1, t1_col2 = st.columns(2)
with t1_col1:
    st.markdown(f"Call value: **{round(call_price, 3)}**")
with t1_col2:
    st.markdown(f"Put value: **{round(put_price, 3)}**")

tab1, tab2, tab3 = st.tabs(["Option's fair value heatmap", "Option's P&L heatmap", "Expected underlying distribution"])

# Generate optimized matrices
output_matrix_C = HeatMapMatrix_Optimized(SpotPrices_space, Volatilities_space, SelectedStrike, Risk_Free_Rate, days_to_maturity, 'C')
output_matrix_P = HeatMapMatrix_Optimized(SpotPrices_space, Volatilities_space, SelectedStrike, Risk_Free_Rate, days_to_maturity, 'P')

##### Heatmaps configuration ################################################################

with tab1:
    st.write("Explore different contract's values given variations in Spot Prices and Annualized Volatilities")
    fig, axs = plt.subplots(2, 1, figsize=(25, 25))

    sns.heatmap(output_matrix_C.T, annot=True, fmt='.1f',
                xticklabels=[str(round(i, 2)) for i in SpotPrices_space],
                yticklabels=[str(round(i, 2)) for i in Volatilities_space], ax=axs[0],
                cbar_kws={'label': 'Call Value'})
    axs[0].set_title('Call heatmap', fontsize=20)
    axs[0].set_xlabel('Spot Price', fontsize=15)
    axs[0].set_ylabel('Annualized Volatility', fontsize=15)

    sns.heatmap(output_matrix_P.T, annot=True, fmt='.1f',
                xticklabels=[str(round(i, 2)) for i in SpotPrices_space],
                yticklabels=[str(round(i, 2)) for i in Volatilities_space], ax=axs[1],
                cbar_kws={'label': 'Put Value'})
    axs[1].set_title('Put heatmap', fontsize=20)
    axs[1].set_xlabel('Spot Price', fontsize=15)
    axs[1].set_ylabel('Annualized Volatility', fontsize=15)

    st.pyplot(fig)

with tab2:
    st.write("Explore different expected P&L's from a specific contract trade given variations in the Spot Price and Annualized Volatility")

    fig, axs = plt.subplots(1, 1, figsize=(25, 15))

    # Vectorized P&L calculations
    call_PL = output_matrix_C.T - option_purchase_price - 2 * transaction_cost
    put_PL = output_matrix_P.T - option_purchase_price - 2 * transaction_cost
    PL_options = [call_PL, put_PL]
    
    selection = 0 if trade_type == 'Call' else 1
    specific_contract_pl = cal_contract_prices[selection] - option_purchase_price - 2 * transaction_cost
    
    st.markdown(f':green-badge[Expected P&L given selected parameters: **{round(specific_contract_pl, 2)}**]')
    
    mapping_color = sns.diverging_palette(15, 145, s=60, as_cmap=True)
    sns.heatmap(PL_options[selection], annot=True, fmt='.1f',
                xticklabels=[str(round(i, 2)) for i in SpotPrices_space],
                yticklabels=[str(round(i, 2)) for i in Volatilities_space], ax=axs,
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
        st.write('Where $\\mu$ is the risk free rate, $\\sigma$ the annualized volatility of the asset and $S(0)$ the spot price')
    
    t3_col1, t3_col2, t3_col3 = st.columns(3)
    with t3_col1:
        NS = st.slider('Number of simulations ($N$)', 100, 10000, 1000, 10)
    with t3_col2:
        s_selection = st.radio('Select time interval', ['Days', 'Hours', 'Minutes'], horizontal=True)
    with t3_col3:
        timeshot = st.slider("Select chart's timestamp (days/year)", 0.0, days_to_maturity / 365, days_to_maturity / 365)

    if s_selection == 'Days':
        step = int(days_to_maturity)
    elif s_selection == 'Hours':
        step = int(days_to_maturity * 24)
    elif s_selection == 'Minutes':
        step = int(days_to_maturity * 24 * 60)

    # Optimized simulation
    simulation_paths = Underlying_price * simulate_optimized(NS, days_to_maturity, step, volatility, Risk_Free_Rate)

    # Vectorized option price calculation
    option_prices = get_Option_Price_vectorized(SelectedStrike, simulation_paths, timeshot, days_to_maturity, step, trade_type)
    pl_results = option_prices - option_purchase_price - 2 * transaction_cost

    # Vectorized probability calculations
    otm_probability = round(np.mean(option_prices == 0), 2)
    itm_probability = round(1 - otm_probability, 2)
    positive_pl_proba = round(np.mean(pl_results > 0), 2)

    st.subheader('Results')

    t32_col1, t32_col2, t32_col3 = st.columns(3)
    t32_col1.metric("In-the money probability", itm_probability, border=True)
    t32_col2.metric("Out-the money probability", otm_probability, border=True)
    t32_col3.metric("Positive P&L probability", positive_pl_proba, border=True)

    # Plots
    t33_col1, t33_col2 = st.columns(2)
    with t33_col1:
        # Calculate time index for plotting
        time_ratio = timeshot * 365 / days_to_maturity
        time_index = int(step * time_ratio)
        time_index = min(time_index, step - 1)
        
        t3_fig1 = plt.figure(figsize=(8, 8))
        sns.histplot(simulation_paths[time_index, :], kde=True, stat='probability')
        plt.xlabel('Price')
        plt.axvline(SelectedStrike, 0, 1, color='r', label='Strike price')
        plt.title(f'Expected underlying asset price distribution at day {int(timeshot * 365)}')
        plt.legend()
        st.pyplot(t3_fig1)

    with t33_col2:
        t3_fig2 = plt.figure(figsize=(8, 3))
        sns.histplot(option_prices, kde=True, stat='probability')
        plt.xlabel('Price')
        plt.title(f'Expected {trade_type} premium at day {int(timeshot * 365)}')
        st.pyplot(t3_fig2)

        t3_fig3 = plt.figure(figsize=(8, 3))
        sns.histplot(pl_results, kde=True, stat='probability')
        plt.xlabel('Price')
        plt.title(f'Expected P&L distribution at day {int(timeshot * 365)}')
        st.pyplot(t3_fig3)
