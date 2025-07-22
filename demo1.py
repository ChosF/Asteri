import streamlit as st
st.set_page_config(layout="wide")
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns
from numba import njit, prange
from concurrent.futures import ProcessPoolExecutor
import functools

### App functions ######################################################
@njit(fastmath=True, cache=True)
def BlackScholes_vectorized(r, S, K, T, sigma, tipo=0):
    """
    Vectorized Black-Scholes calculation using Numba JIT compilation
    tipo: 0 for Call, 1 for Put
    """
    d1 = (np.log(S/K) + (r + sigma**2/2)*T)/(sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    # Use pre-computed normal CDF values for better performance
    norm_d1 = 0.5 * (1.0 + np.sign(d1) * np.sqrt(1 - np.exp(-2 * d1**2 / np.pi)))
    norm_d2 = 0.5 * (1.0 + np.sign(d2) * np.sqrt(1 - np.exp(-2 * d2**2 / np.pi)))
    norm_neg_d1 = 1 - norm_d1
    norm_neg_d2 = 1 - norm_d2
    
    if tipo == 0:  # Call option
        precio = S * norm_d1 - K * np.exp(-r * T) * norm_d2
    else:  # Put option
        precio = K * np.exp(-r * T) * norm_neg_d2 - S * norm_neg_d1
        
    return precio

@njit(parallel=True, fastmath=True, cache=True)
def HeatMapMatrix_parallel(Spot_Prices, Volatilities, Strike, Interest_Rate, T, option_type=0):
    """
    Parallel computation of heatmap matrix using Numba
    """
    M = np.zeros((len(Spot_Prices), len(Volatilities)))
    
    for i in prange(len(Spot_Prices)):
        for j in prange(len(Volatilities)):
            M[i,j] = BlackScholes_vectorized(Interest_Rate, Spot_Prices[i], Strike, T, Volatilities[j], option_type)
    
    return M

@njit(parallel=True, fastmath=True, cache=True)
def simulate_paths_parallel(S0, mu, sigma, dt, steps, num_simulations):
    """
    Parallel simulation of asset paths using geometric Brownian motion
    """
    paths = np.zeros((steps + 1, num_simulations))
    paths[0, :] = S0
    
    for j in prange(num_simulations):
        for i in range(1, steps + 1):
            z = np.random.normal(0, 1)
            paths[i, j] = paths[i-1, j] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
    
    return paths

@njit(parallel=True, fastmath=True, cache=True)
def calculate_option_payoff_parallel(St, K, option_type=0):
    """
    Parallel calculation of option payoffs
    option_type: 0 for Call, 1 for Put
    """
    payoffs = np.zeros(len(St))
    
    for i in prange(len(St)):
        if option_type == 0:  # Call
            payoffs[i] = max(St[i] - K, 0.0)
        else:  # Put
            payoffs[i] = max(K - St[i], 0.0)
    
    return payoffs

###############################################################################################################
#### Sidebar parameters ###############################################
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
T = days_to_maturity / 365
########################################################################

st.header('Black Scholes options heatmap')
st.write("Calculates an option's arbitrage-free premium using the Black Scholes option pricing model.")

# Calculate option prices using vectorized functions
call_price = BlackScholes_vectorized(Risk_Free_Rate, Underlying_price, SelectedStrike, T, volatility, 0)
put_price = BlackScholes_vectorized(Risk_Free_Rate, Underlying_price, SelectedStrike, T, volatility, 1)

cal_contract_prices = [call_price, put_price]
t1_col1, t1_col2 = st.columns(2)
with t1_col1:
    st.markdown(f"Call value: **{round(call_price, 3)}**")
with t1_col2:
    st.markdown(f"Put value: **{round(put_price, 3)}**")

tab1, tab2, tab3 = st.tabs(["Option's fair value heatmap", "Option's P&L heatmap", "Expected underlying distribution"])

###### Operations - Compute both matrices in parallel
output_matrix_C = HeatMapMatrix_parallel(SpotPrices_space, Volatilities_space, SelectedStrike, Risk_Free_Rate, T, 0)
output_matrix_P = HeatMapMatrix_parallel(SpotPrices_space, Volatilities_space, SelectedStrike, Risk_Free_Rate, T, 1)

##### Heatmaps configuration ################################################################
with tab1:
    st.write("Explore different contract's values given variations in Spot Prices and Annualized Volatilities")
    fig, axs = plt.subplots(2, 1, figsize=(25, 25))

    sns.heatmap(output_matrix_C.T, annot=True, fmt='.1f',
                xticklabels=[str(round(i, 2)) for i in SpotPrices_space],
                yticklabels=[str(round(i, 2)) for i in Volatilities_space], ax=axs[0],
                cbar_kws={'label': 'Call Value',})
    axs[0].set_title('Call heatmap', fontsize=20)
    axs[0].set_xlabel('Spot Price', fontsize=15)
    axs[0].set_ylabel('Annualized Volatility', fontsize=15)

    sns.heatmap(output_matrix_P.T, annot=True, fmt='.1f',
                xticklabels=[str(round(i, 2)) for i in SpotPrices_space],
                yticklabels=[str(round(i, 2)) for i in Volatilities_space], ax=axs[1],
                cbar_kws={'label': 'Put Value',})

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
        st.write('Where $\\mu$ is the risk free rate, $\\sigma$ the annualized volatility of the asset you want to simulate and $S(0)$ the asset price at the beginning (spot price)')
    
    t3_col1, t3_col2, t3_col3 = st.columns(3)
    with t3_col1:
        NS = st.slider('Number of simulations ($N$)', 100, 10000, 1000, 10)
    with t3_col2:
        s_selection = st.radio('Select time interval', ['Days', 'Hours', 'Minutes'], horizontal=True, help='The time interval each price point will represent. This option is merely for visual purposes.')
    with t3_col3:
        timeshot = st.slider("Select chart's timestamp (days/year)", 0.0, days_to_maturity / 365, days_to_maturity / 365)

    step_multiplier = {'Days': 1, 'Hours': 24, 'Minutes': 24 * 60}
    step = days_to_maturity * step_multiplier[s_selection]
    
    #### Creating the simulations
    @st.cache_data
    def simulate_paths_cached(NS, days_to_maturity, step, volatility, Risk_Free_Rate, Underlying_price):
        dt = (days_to_maturity / 365) / step
        return simulate_paths_parallel(Underlying_price, Risk_Free_Rate, volatility, dt, step, NS)
    
    simulation_paths = simulate_paths_cached(NS, days_to_maturity, step, volatility, Risk_Free_Rate, Underlying_price)

    def get_option_prices_vectorized(K, St, trade_type):
        dynamic_index = -int(step - timeshot * 365 * (step/days_to_maturity) + 1)
        option_type = 0 if trade_type == 'Call' else 1
        return calculate_option_payoff_parallel(St[dynamic_index, :], K, option_type)

    option_prices = get_option_prices_vectorized(SelectedStrike, simulation_paths, trade_type)
    pl_results = option_prices - option_purchase_price - 2 * transaction_cost

    otm_probability = round(np.sum(option_prices == 0) / len(option_prices), 2)
    itm_probability = round(1 - otm_probability, 2)
    positive_pl_proba = round(np.sum(pl_results > 0) / len(pl_results), 2)

    st.subheader('Results')

    t32_col1, t32_col2, t32_col3 = st.columns(3)
    t32_col1.metric("In-the money probability", itm_probability, border=True)
    t32_col2.metric("Out-the money probability", otm_probability, border=True)
    t32_col3.metric("Positive P&L probability", positive_pl_proba, border=True)
    
    #### Plots
    t33_col1, t33_col2 = st.columns(2)
    with t33_col1:
        t3_fig1 = plt.figure(figsize=(8, 8))
        target_index = - int(step - timeshot * step + 1)
        sns.histplot(simulation_paths[target_index, :], kde=True, stat='probability')
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
