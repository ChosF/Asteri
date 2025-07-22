import streamlit as st
st.set_page_config(layout="wide")
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor
from numba import njit, prange
import multiprocessing as mp

### App functions ######################################################
@njit
def BlackScholes(r, S, K, T, sigma, tipo_num=0):
    ''' 
    r : Interest Rate
    S : Spot Price
    K : Strike Price
    T : Days due expiration / 365
    sigma : Annualized Volatility
    tipo_num: 0 for Call, 1 for Put
    '''
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    # Use approximate normal CDF for numba compatibility
    if tipo_num == 0:  # Call
        precio = S * norm_cdf_approx(d1) - K * np.exp(-r * T) * norm_cdf_approx(
            d2
        )
    else:  # Put
        precio = K * np.exp(-r * T) * norm_cdf_approx(
            -d2
        ) - S * norm_cdf_approx(-d1)

    return precio

@njit
def norm_cdf_approx(x):
    """Fast approximation of normal CDF using error function"""
    return 0.5 * (1.0 + np.tanh(0.7978845608 * (x + 0.044715 * x**3)))

@njit(parallel=True)
def HeatMapMatrix_fast(
    Spot_Prices, Volatilities, Strike, Interest_Rate, T, tipo_num=0
):
    rows, cols = len(Spot_Prices), len(Volatilities)
    M = np.zeros((rows, cols))

    for i in prange(rows):
        for j in range(cols):
            M[i, j] = BlackScholes(
                Interest_Rate,
                Spot_Prices[i],
                Strike,
                T,
                Volatilities[j],
                tipo_num,
            )

    return M

def HeatMapMatrix(
    Spot_Prices, Volatilities, Strike, Interest_Rate, Days_to_Exp, type="C"
):
    T = Days_to_Exp / 365
    tipo_num = 0 if type == "C" else 1
    Spot_arr = np.array(Spot_Prices, dtype=np.float64)
    Vol_arr = np.array(Volatilities, dtype=np.float64)

    M = HeatMapMatrix_fast(Spot_arr, Vol_arr, Strike, Interest_Rate, T, tipo_num)
    return np.round(M, 2)

# Moved this function definition up
def BlackScholes_scipy(r, S, K, T, sigma, tipo="C"):
    d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if tipo == "C":
        precio = S * norm.cdf(d1, 0, 1) - K * np.exp(-r * T) * norm.cdf(d2, 0, 1)
    elif tipo == "P":
        precio = K * np.exp(-r * T) * norm.cdf(-d2, 0, 1) - S * norm.cdf(
            -d1, 0, 1
        )
    return precio

###############################################################################################################
#### Sidebar parameters ###############################################
st.sidebar.header("Option Parameters")
Underlying_price = st.sidebar.number_input("Spot Price", value=100)
trade_type = st.sidebar.segmented_control(
    "Contract type", ["Call", "Put"], default="Call"
)
SelectedStrike = st.sidebar.number_input("Strike/Exercise Price", value=80)
days_to_maturity = st.sidebar.number_input("Time to Maturity (days)", value=365)
Risk_Free_Rate = st.sidebar.number_input("Risk-Free Interest Rate ", value=0.1)
volatility = st.sidebar.number_input("Annualized Volatility", value=0.2)
st.sidebar.subheader("P&L Parameters")
option_purchase_price = st.sidebar.number_input("Option's Price")
transaction_cost = st.sidebar.number_input("Opening/Closing Cost")

st.sidebar.subheader("Heatmap Parameters")
min_spot_price = st.sidebar.number_input("Min Spot price", value=50)
max_spot_price = st.sidebar.number_input("Max Spot price", value=110)

min_vol = st.sidebar.slider("Min Volatility", 0.01, 1.00)
max_vol = st.sidebar.slider("Max Volatility", 0.01, 1.00, 1.00)
grid_size = st.sidebar.slider("Grid size (nxn)", 5, 20, 10)

#### Variables ########################################################
SpotPrices_space = np.linspace(min_spot_price, max_spot_price, grid_size)
Volatilities_space = np.linspace(min_vol, max_vol, grid_size)
########################################################################

st.header("Black Scholes options heatmap")
st.write(
    "Calculates an option's arbitrage-free premium using the Black Scholes option pricing model."
)

# Use original scipy functions for single calculations
# This now works because the function is defined above
call_price = BlackScholes_scipy(
    Risk_Free_Rate,
    Underlying_price,
    SelectedStrike,
    days_to_maturity / 365,
    volatility,
)
put_price = BlackScholes_scipy(
    Risk_Free_Rate,
    Underlying_price,
    SelectedStrike,
    days_to_maturity / 365,
    volatility,
    "P",
)

cal_contract_prices = [call_price, put_price]
t1_col1, t1_col2 = st.columns(2)
with t1_col1:
    st.markdown(f"Call value: **{round(call_price,3)}**")
with t1_col2:
    st.markdown(f"Put value: **{round(put_price,3)}**")

tab1, tab2, tab3 = st.tabs(
    [
        "Option's fair value heatmap",
        "Option's P&L heatmap",
        "Expected underlying distribution",
    ]
)

###### Operations - Cached for performance
@st.cache_data
def compute_heatmaps(spot_prices, volatilities, strike, rate, days):
    output_matrix_C = HeatMapMatrix(
        spot_prices, volatilities, strike, rate, days, type="C"
    )
    output_matrix_P = HeatMapMatrix(
        spot_prices, volatilities, strike, rate, days, type="P"
    )
    return output_matrix_C, output_matrix_P

output_matrix_C, output_matrix_P = compute_heatmaps(
    SpotPrices_space,
    Volatilities_space,
    SelectedStrike,
    Risk_Free_Rate,
    days_to_maturity,
)

##### Heatmaps configuration #################################################################
with tab1:
    st.write(
        "Explore different contract's values given variations in Spot Prices and Annualized Volatilities"
    )
    fig, axs = plt.subplots(2, 1, figsize=(25, 25))

    sns.heatmap(
        output_matrix_C.T,
        annot=True,
        fmt=".1f",
        xticklabels=[str(round(i, 2)) for i in SpotPrices_space],
        yticklabels=[str(round(i, 2)) for i in Volatilities_space],
        ax=axs[0],
        cbar_kws={"label": "Call Value"},
    )
    axs[0].set_title("Call heatmap", fontsize=20)
    axs[0].set_xlabel("Spot Price", fontsize=15)
    axs[0].set_ylabel("Annualized Volatility", fontsize=15)

    sns.heatmap(
        output_matrix_P.T,
        annot=True,
        fmt=".1f",
        xticklabels=[str(round(i, 2)) for i in SpotPrices_space],
        yticklabels=[str(round(i, 2)) for i in Volatilities_space],
        ax=axs[1],
        cbar_kws={"label": "Put Value"},
    )

    axs[1].set_title("Put heatmap", fontsize=20)
    axs[1].set_xlabel("Spot Price", fontsize=15)
    axs[1].set_ylabel("Annualized Volatility", fontsize=15)

    st.pyplot(fig)

with tab2:
    st.write(
        "Explore different expected P&L's from a specific contract trade given variations in the Spot Price and Annualized Volatility"
    )

    fig, axs = plt.subplots(1, 1, figsize=(25, 15))

    call_PL = output_matrix_C.T - option_purchase_price - 2 * transaction_cost
    put_PL = output_matrix_P.T - option_purchase_price - 2 * transaction_cost
    PL_options = [call_PL, put_PL]
    selection = 0 if trade_type == "Call" else 1

    specific_contract_pl = (
        cal_contract_prices[selection]
        - option_purchase_price
        - 2 * transaction_cost
    )
    st.markdown(
        f":green-badge[Expected P&L given selected parameters: **{round(specific_contract_pl,2)}**]"
    )

    mapping_color = sns.diverging_palette(15, 145, s=60, as_cmap=True)
    sns.heatmap(
        PL_options[selection],
        annot=True,
        fmt=".1f",
        xticklabels=[str(round(i, 2)) for i in SpotPrices_space],
        yticklabels=[str(round(i, 2)) for i in Volatilities_space],
        ax=axs,
        cmap=mapping_color,
        center=0,
    )
    axs.set_title(f"{trade_type} Expected P&L", fontsize=20)
    axs.set_xlabel("Spot Price", fontsize=15)
    axs.set_ylabel("Annualized Volatility", fontsize=15)

    st.pyplot(fig)

with tab3:
    st.write(
        "Calculate the expected distribution of the underlying asset price, the option premium and the p&l from trading the option"
    )
    with st.expander("See methodology"):
        st.write(
            "The distribution is obtained by simulating $N$ times the underlying asset price as a geometric brownian process during a specified time period."
            " The function $S : [0, \infty) \mapsto [0, \infty) $ will describe the stochastic process as: "
        )
        st.latex("S(t) = S(0) e^{(\mu - \sigma^2 / 2)t + \sigma W(t)} ")
        st.write(
            "Where $\mu$ is the risk free rate, $\sigma$ the annualized volatility of the asset you want to simulate and $S(0)$ the asset price at the beginning (spot price)"
        )

    t3_col1, t3_col2, t3_col3 = st.columns(3)
    with t3_col1:
        NS = st.slider("Number of simulations ($N$)", 100, 10000, 1000, 10)
    with t3_col2:
        s_selection = st.radio(
            "Select time interval",
            ["Days", "Hours", "Minutes"],
            horizontal=True,
            help="The time interval each price point will represent. This option is merely for visual purposes.",
        )
    with t3_col3:
        timeshot = st.slider(
            "Select chart's timestamp (days/year)",
            0.0,
            days_to_maturity / 365,
            days_to_maturity / 365,
        )

    step_map = {
        "Days": days_to_maturity,
        "Hours": days_to_maturity * 24,
        "Minutes": days_to_maturity * 24 * 60,
    }
    step = step_map[s_selection]

    #### Creating the simulations
    @st.cache_data
    @njit
    def simulate_paths_fast(NS, T, steps, S0, vol, rate):
        dt = T / steps
        sqrt_dt = np.sqrt(dt)
        drift = (rate - 0.5 * vol**2) * dt

        # Pre-allocate array
        paths = np.zeros((steps + 1, NS))
        paths[0, :] = S0

        # Generate all random numbers at once
        Z = np.random.normal(0, 1, (steps, NS)) * vol * sqrt_dt

        # Vectorized path generation
        for i in range(1, steps + 1):
            paths[i, :] = paths[i - 1, :] * np.exp(drift + Z[i - 1, :])

        return paths

    def simulate(NS, days_to_maturity, s, volatility, Risk_Free_Rate):
        T = days_to_maturity / 365
        return simulate_paths_fast(
            NS, T, s, Underlying_price, volatility, Risk_Free_Rate
        )

    simulation_paths = simulate(
        NS, days_to_maturity, step, volatility, Risk_Free_Rate
    )

    @njit
    def get_Option_Price_fast(K, St_final, option_type=0):
        """option_type: 0 for Call, 1 for Put"""
        if option_type == 0:  # Call
            return np.maximum(St_final - K, 0.0)
        else:  # Put
            return np.maximum(K - St_final, 0.0)

    def get_Option_Price(K, St, type="Call"):
        dynamic_index = -int(
            step - timeshot * 365 * (step / days_to_maturity) + 1
        )
        option_type = 0 if type == "Call" else 1
        return get_Option_Price_fast(K, St[dynamic_index, :], option_type)

    option_prices = get_Option_Price(
        SelectedStrike, simulation_paths, trade_type
    )
    pl_results = option_prices - option_purchase_price - 2 * transaction_cost

    otm_probability = round(np.sum(option_prices == 0) / len(option_prices), 2)
    itm_probability = round(1 - otm_probability, 2)
    positive_pl_proba = round(np.sum(pl_results > 0) / len(pl_results), 2)

    st.subheader("Results")

    t32_col1, t32_col2, t32_col3 = st.columns(3)
    t32_col1.metric("In-the money probability", itm_probability)
    t32_col2.metric("Out-the money probability", otm_probability)
    t32_col3.metric("Positive P&L probability", positive_pl_proba)

    #### Plots
    t33_col1, t33_col2 = st.columns(2)
    with t33_col1:
        t3_fig1 = plt.figure(figsize=(8, 8))
        final_prices = simulation_paths[-int(step - timeshot * step + 1), :]
        sns.histplot(final_prices, kde=True, stat="probability")
        plt.xlabel("Price")
        plt.axvline(SelectedStrike, 0, 1, color="r", label="Strike price")
        plt.title(
            f"Expected underlying asset price distribution at day {int(timeshot * 365)}"
        )
        plt.legend()
        st.pyplot(t3_fig1)

    with t33_col2:
        t3_fig2 = plt.figure(figsize=(8, 3))
        sns.histplot(option_prices, kde=True, stat="probability")
        plt.xlabel("Price")
        plt.title(f"Expected {trade_type} premium at day {int(timeshot * 365)}")
        plt.legend()
        st.pyplot(t3_fig2)

        t3_fig3 = plt.figure(figsize=(8, 3))
        sns.histplot(pl_results, kde=True, stat="probability")
        plt.xlabel("Price")
        plt.title(f"Expected P&L distribution at day {int(timeshot * 365)}")
        plt.legend()
        st.pyplot(t3_fig3)
