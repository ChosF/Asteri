import streamlit as st
st.set_page_config(layout="wide")
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns
from numba import jit, prange
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
from functools import partial

### App functions ######################################################

@jit(nopython=True)
def fast_black_scholes(r, S, K, T, sigma, tipo=0):
    """
    Compiled Black-Scholes formula using Numba JIT
    tipo: 0 for Call, 1 for Put
    """
    d1 = (np.log(S/K) + (r + sigma**2/2)*T)/(sigma* np.sqrt(T)) 
    d2 = d1 - sigma * np.sqrt(T)
    
    sqrt_2pi = np.sqrt(2 * np.pi)
    
    # Fast normal CDF approximation
    def norm_cdf(x):
        return 0.5 * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))
    
    cdf_d1 = norm_cdf(d1)
    cdf_d2 = norm_cdf(d2)
    cdf_neg_d1 = norm_cdf(-d1)
    cdf_neg_d2 = norm_cdf(-d2)
    
    if tipo == 0:  # Call
        return S * cdf_d1 - K * np.exp(-r * T) * cdf_d2
    else:  # Put
        return K * np.exp(-r * T) * cdf_neg_d2 - S * cdf_neg_d1

@jit(nopython=True, parallel=True)
def vectorized_heatmap_matrix(spot_prices, volatilities, strike, interest_rate, T, option_type):
    """
    Vectorized heatmap calculation using broadcasting and parallel processing
    """
    n_spots = len(spot_prices)
    n_vols = len(volatilities)
    result = np.zeros((n_spots, n_vols))
    
    for i in prange(n_spots):
        for j in prange(n_vols):
            result[i, j] = fast_black_scholes(interest_rate, spot_prices[i], 
                                            strike, T, volatilities[j], option_type)
    
    return result

@jit(nopython=True, parallel=True)
def fast_gbm_simulation(n_simulations, n_steps, dt, s0, r, sigma):
    """
    Fast Geometric Brownian Motion simulation using Numba
    """
    paths = np.zeros((n_steps + 1, n_simulations))
    paths[0, :] = s0
    
    drift = (r - 0.5 * sigma**2) * dt
    vol_dt = sigma * np.sqrt(dt)
    
    for i in prange(n_simulations):
        for j in range(n_steps):
            z = np.random.normal(0, 1)
            paths[j + 1, i] = paths[j, i] * np.exp(drift + vol_dt * z)
    
    return paths

@jit(nopython=True, parallel=True)
def fast_option_payoff(strike, spot_prices, option_type):
    """
    Fast option payoff calculation
    """
    n_sims = len(spot_prices)
    payoffs = np.zeros(n_sims)
    
    for i in prange(n_sims):
        if option_type == 0:  # Call
            payoffs[i] = max(spot_prices[i] - strike, 0.0)
        else:  # Put
            payoffs[i] = max(strike - spot_prices[i], 0.0)
    
    return payoffs

def BlackScholes(r, S, K, T, sigma, tipo='C'):
    """ 
    Wrapper function to maintain API compatibility
    """
    option_type = 0 if tipo == 'C' else 1
    return fast_black_scholes(r, S, K, T, sigma, option_type)

def HeatMapMatrix(Spot_Prices, Volatilities, Strike, Interest_Rate, Days_to_Exp, type='C'):
    """
    Vectorized heatmap matrix calculation
    """
    T = Days_to_Exp / 365
    option_type = 0 if type == 'C' else 1
    
    spot_array = np.array(Spot_Prices)
    vol_array = np.array(Volatilities)
    
    matrix = vectorized_heatmap_matrix(spot_array, vol_array, Strike, Interest_Rate, T, option_type)
    return np.round(matrix, 2)

###############################################################################################################
#### Sidebar parameters ###############################################
st.sidebar.header('Option Parameters')
Underlying_price = st.sidebar.number_input('Spot Price', value = 100)
trade_type = st.sidebar.segmented_control("Contract type", ['Call', 'Put'], default= 'Call')
SelectedStrike = st.sidebar.number_input('Strike/Exercise Price', value = 80)
days_to_maturity = st.sidebar.number_input('Time to Maturity (days)', value = 365)
Risk_Free_Rate = st.sidebar.number_input('Risk-Free Interest Rate ', value = 0.1)
volatility = st.sidebar.number_input('Annualized Volatility', value = 0.2)
st.sidebar.subheader('P&L Parameters')
option_purchase_price = st.sidebar.number_input("Option's Price") 
transaction_cost = st.sidebar.number_input("Opening/Closing Cost") 

st.sidebar.subheader('Heatmap Parameters')
min_spot_price = st.sidebar.number_input('Min Spot price',value= 50)
max_spot_price = st.sidebar.number_input('Max Spot price', value = 110)

min_vol = st.sidebar.slider('Min Volatility', 0.01, 1.00)
max_vol = st.sidebar.slider('Max Volatility', 0.01, 1.00, 1.00)
grid_size = st.sidebar.slider('Grid size (nxn)', 5, 20, 10)

#### Variables ########################################################
SpotPrices_space = np.linspace(min_spot_price, max_spot_price, grid_size)
Volatilities_space = np.linspace(min_vol, max_vol, grid_size)
########################################################################

st.header('Black Scholes options heatmap')
st.write("Calculates an option's arbitrage-free premium using the Black Scholes option pricing model.")

# Cache expensive computations
@st.cache_data
def compute_option_prices(underlying_price, strike, days_to_maturity, risk_free_rate, volatility):
    call_price = BlackScholes(risk_free_rate, underlying_price, strike, days_to_maturity / 365, volatility)
    put_price = BlackScholes(risk_free_rate, underlying_price, strike, days_to_maturity / 365, volatility, 'P')
    return call_price, put_price

@st.cache_data
def compute_heatmaps(spot_prices, volatilities, strike, interest_rate, days_to_exp):
    output_matrix_C = HeatMapMatrix(spot_prices, volatilities, strike, interest_rate, days_to_exp, 'C')
    output_matrix_P = HeatMapMatrix(spot_prices, volatilities, strike, interest_rate, days_to_exp, 'P')
    return output_matrix_C, output_matrix_P

call_price, put_price = compute_option_prices(Underlying_price, SelectedStrike, days_to_maturity, Risk_Free_Rate, volatility)

cal_contract_prices = [call_price, put_price]
t1_col1, t1_col2 = st.columns(2)
with t1_col1:
    st.markdown(f"Call value: **{round(call_price,3)}**")
with t1_col2:
    st.markdown(f"Put value: **{round(put_price,3)}**")

tab1, tab2, tab3 = st.tabs(["Option's fair value heatmap", "Option's P&L heatmap", "Expected underlying distribution"])

###### Operations
output_matrix_C, output_matrix_P = compute_heatmaps(SpotPrices_space.tolist(), Volatilities_space.tolist(), 
                                                   SelectedStrike, Risk_Free_Rate, days_to_maturity)

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

    call_PL = output_matrix_C.T - option_purchase_price - 2 * transaction_cost
    put_PL = output_matrix_P.T - option_purchase_price - 2 * transaction_cost
    PL_options = [call_PL, put_PL]
    selection = 0 if trade_type == 'Call' else 1

    specific_contrac_pl = cal_contract_prices[selection] - option_purchase_price - 2 * transaction_cost
    st.markdown(f':green-badge[Expected P&L given selected parameters: **{round(specific_contrac_pl,2)}**]')
    
    maping_color = sns.diverging_palette(15, 145, s=60, as_cmap=True)
    sns.heatmap(PL_options[selection], annot=True, fmt='.1f',
                xticklabels=[str(round(i, 2)) for i in SpotPrices_space], 
                yticklabels=[str(round(i, 2)) for i in Volatilities_space], ax=axs, 
                cmap=maping_color, center=0)
    axs.set_title(f'{trade_type} Expected P&L', fontsize=20)
    axs.set_xlabel('Spot Price', fontsize=15)
    axs.set_ylabel('Annualized Volatility', fontsize=15)

    st.pyplot(fig)

with tab3:
    st.write('Calculate the expected distribution of the underlying asset price, the option premium and the p&l from trading the option')
    with st.expander("See methodology"):
        st.write('The distribution is obtained by simulating \(N\) times the underlying asset price as a geometric brownian process during a specified time period.' \
        ' The function \(S : [0, \\infty) \\mapsto [0, \\infty)\) will describe the stochastic process as: ')
        st.latex('S(t) = S(0) e^{(\mu - \sigma^2 / 2)t + \sigma W(t)} ')
        st.write('Where \(\mu\) is the risk free rate, \(\sigma\) the annualized volatility of the asset you want to simulate and \(S(0)\) the asset price at the beginning (spot price)')
    
    t3_col1, t3_col2, t3_col3 = st.columns(3)
    with t3_col1:
        NS = st.slider('Number of simulations (\(N\))', 100, 10000, 1000, 10)
    with t3_col2:
        s_selection = st.radio('Select time interval', ['Days', 'Hours', 'Minutes'], horizontal=True, 
                              help='The time interval each price point will represent. This option is merely for visual purposes.')
    with t3_col3:
        timeshot = st.slider("Select chart's timestamp (days/year)", 0.0, days_to_maturity / 365, days_to_maturity / 365)

    time_multipliers = {'Days': 1, 'Hours': 24, 'Minutes': 24 * 60}
    step = int(days_to_maturity * time_multipliers[s_selection])
    
    @st.cache_data
    def simulate_paths(ns, days_to_maturity, steps, s, volatility, risk_free_rate):
        dt = (days_to_maturity / 365) / steps
        return fast_gbm_simulation(ns, steps, dt, s, risk_free_rate, volatility)
    
    simulation_paths = simulate_paths(NS, days_to_maturity, step, Underlying_price, volatility, Risk_Free_Rate)

    def get_option_price_fast(strike, simulation_paths, trade_type, timeshot, step, days_to_maturity):
        dynamic_index = int(step - timeshot * 365 * (step / days_to_maturity))
        if dynamic_index >= simulation_paths.shape[0]:
            dynamic_index = -1
        
        spot_prices = simulation_paths[dynamic_index, :]
        option_type = 0 if trade_type == 'Call' else 1
        
        return fast_option_payoff(strike, spot_prices, option_type)

    option_prices = get_option_price_fast(SelectedStrike, simulation_paths, trade_type, timeshot, step, days_to_maturity)
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

    #### Plots
    t33_col1, t33_col2 = st.columns(2)
    with t33_col1:
        dynamic_index = int(step - timeshot * step)
        if dynamic_index >= simulation_paths.shape[0]:
            dynamic_index = -1
            
        t3_fig1 = plt.figure(figsize=(8, 8))
        sns.histplot(simulation_paths[dynamic_index, :], kde=True, stat='probability')
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
        plt.legend()
        st.pyplot(t3_fig2)

        t3_fig3 = plt.figure(figsize=(8, 3))
        sns.histplot(pl_results, kde=True, stat='probability')
        plt.xlabel('Price')
        plt.title(f'Expected P&L distribution at day {int(timeshot * 365)}')
        plt.legend()
        st.pyplot(t3_fig3)
