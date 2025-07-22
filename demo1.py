import streamlit as st
st.set_page_config(
    layout="wide",
    page_title="Black-Scholes Options Pricing",
    page_icon="📈"
)

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from scipy.stats import norm
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Thread
import time

# Custom CSS for modern UX with smaller metric containers and separation
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        backdrop-filter: blur(10px);
        background: rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 8px;
        margin-bottom: 20px;
        margin-top: 30px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 16px;
        margin: 0 4px;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(255, 255, 255, 0.2);
        transform: translateY(-2px);
    }
    
    .metric-container {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 8px;
        padding: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: all 0.3s ease;
        text-align: center;
        margin: 0 8px;
    }
    
    .metric-container h3 {
        font-size: 0.9rem;
        margin-bottom: 8px;
        color: #888;
    }
    
    .metric-container h2 {
        font-size: 1.4rem;
        margin: 0;
    }
    
    .metric-container:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
    }
    
    .sidebar .stNumberInput, .sidebar .stSelectbox, .sidebar .stSlider {
        margin-bottom: 16px;
    }
    
    .sidebar {
        background: rgba(255, 255, 255, 0.02);
        backdrop-filter: blur(20px);
        border-radius: 0 16px 16px 0;
    }
    
    .expander-content {
        background: rgba(255, 255, 255, 0.02);
        border-radius: 8px;
        padding: 16px;
    }
    
    .separator {
        height: 2px;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
        margin: 25px 0;
    }
    
    @media (prefers-color-scheme: dark) {
        .metric-container {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
    }
    
    @media (prefers-color-scheme: light) {
        .metric-container {
            background: rgba(0, 0, 0, 0.02);
            border: 1px solid rgba(0, 0, 0, 0.1);
        }
        .stTabs [data-baseweb="tab-list"] {
            background: rgba(0, 0, 0, 0.05);
        }
    }
</style>
""", unsafe_allow_html=True)

### Core Functions ######################################################
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
            precio = S * norm.cdf(d1, 0, 1) - K * np.exp(-r * T)*norm.cdf(d2, 0, 1)
        elif tipo == 'P': 
            precio = K * np.exp(-r * T)*norm.cdf(-d2, 0, 1) - S * norm.cdf(-d1, 0, 1)
    except: 
        print('Error in BlackScholes calculation')
    return precio

def calculate_row_threaded(args):
    """Calculate a single row of the heatmap matrix using threading"""
    spot_price, volatilities, strike, interest_rate, T, option_type = args
    row = []
    for vol in volatilities:
        bs_result = BlackScholes(interest_rate, spot_price, strike, T, vol, option_type)
        row.append(round(bs_result, 2))
    return row

def HeatMapMatrix(spot_prices, volatilities, strike, interest_rate, days_to_exp, option_type='C'):
    """Create heatmap matrix using concurrent processing"""
    T = days_to_exp / 365
    
    # Prepare arguments for threading
    args_list = [(spot_price, volatilities, strike, interest_rate, T, option_type) 
                 for spot_price in spot_prices]
    
    # Use ThreadPoolExecutor for concurrent processing
    with ThreadPoolExecutor(max_workers=min(len(spot_prices), 8)) as executor:
        # Submit all tasks
        future_to_row = {executor.submit(calculate_row_threaded, args): i 
                        for i, args in enumerate(args_list)}
        
        # Initialize matrix
        matrix = np.zeros((len(spot_prices), len(volatilities)))
        
        # Collect results as they complete
        for future in as_completed(future_to_row):
            row_index = future_to_row[future]
            try:
                row_data = future.result()
                matrix[row_index] = row_data
            except Exception as exc:
                st.error(f'Row {row_index} generated an exception: {exc}')
    
    return matrix

@st.cache_data
def simulate_paths(ns, days_to_maturity, steps, volatility, risk_free_rate, underlying_price):
    """Cached simulation function with threading for path generation"""
    dt = (days_to_maturity / 365) / steps
    
    def generate_path_chunk(chunk_size):
        """Generate a chunk of simulation paths"""
        z = np.random.normal(0, np.sqrt(dt), (steps, chunk_size))
        paths = np.vstack([
            np.ones(chunk_size), 
            np.exp((risk_free_rate - 0.5 * volatility**2) * dt + volatility * z)
        ]).cumprod(axis=0)
        return underlying_price * paths
    
    # Determine chunk size and number of chunks for threading
    chunk_size = max(1, ns // 4)  # Divide into 4 chunks
    chunks = []
    remaining = ns
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        while remaining > 0:
            current_chunk_size = min(chunk_size, remaining)
            futures.append(executor.submit(generate_path_chunk, current_chunk_size))
            remaining -= current_chunk_size
        
        # Collect results
        for future in as_completed(futures):
            chunks.append(future.result())
    
    # Combine all chunks
    return np.hstack(chunks)

###############################################################################################################
#### Sidebar Parameters ###############################################
with st.sidebar:
    st.header('Option Parameters')
    
    col1, col2 = st.columns(2)
    with col1:
        underlying_price = st.number_input('Spot Price', value=100.0, step=1.0)
        selected_strike = st.number_input('Strike/Exercise Price', value=80.0, step=1.0)
        days_to_maturity = st.number_input('Time to Maturity (days)', value=365, step=1)
    
    with col2:
        risk_free_rate = st.number_input('Risk-Free Interest Rate', value=0.1, step=0.01, format="%.3f")
        volatility = st.number_input('Annualized Volatility', value=0.2, step=0.01, format="%.3f")
        trade_type = st.segmented_control("Contract type", ['Call', 'Put'], default='Call')
    
    st.subheader('P&L Parameters')
    col3, col4 = st.columns(2)
    with col3:
        option_purchase_price = st.number_input("Option's Price", step=0.01, format="%.2f") 
    with col4:
        transaction_cost = st.number_input("Opening/Closing Cost", step=0.01, format="%.2f")
    
    st.subheader('Heatmap Parameters')
    col5, col6 = st.columns(2)
    with col5:
        min_spot_price = st.number_input('Min Spot price', value=50.0, step=1.0)
        max_spot_price = st.number_input('Max Spot price', value=110.0, step=1.0)
    with col6:
        min_vol = st.slider('Min Volatility', 0.01, 1.00, 0.01, step=0.01)
        max_vol = st.slider('Max Volatility', 0.01, 1.00, 1.00, step=0.01)
    
    grid_size = st.slider('Grid size (nxn)', 5, 20, 10)

#### Main App Layout ########################################################
st.title('Black Scholes Options Heatmap')
st.write("Calculates an option's arbitrage-free premium using the Black Scholes option pricing model.")

# Calculate current option prices
call_price = BlackScholes(risk_free_rate, underlying_price, selected_strike, days_to_maturity / 365, volatility)
put_price = BlackScholes(risk_free_rate, underlying_price, selected_strike, days_to_maturity / 365, volatility, 'P')

# Display current prices with custom styling
col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    st.markdown(
        f'<div class="metric-container"><h3>Call Value</h3><h2>${call_price:.3f}</h2></div>', 
        unsafe_allow_html=True
    )
with col2:
    st.markdown(
        f'<div class="metric-container"><h3>Put Value</h3><h2>${put_price:.3f}</h2></div>', 
        unsafe_allow_html=True
    )
with col3:
    put_call_parity = call_price - put_price + selected_strike * np.exp(-risk_free_rate * days_to_maturity / 365) - underlying_price
    st.markdown(
        f'<div class="metric-container"><h3>Put-Call Parity</h3><h2>${put_call_parity:.3f}</h2></div>', 
        unsafe_allow_html=True
    )

# Add separator
st.markdown('<div class="separator"></div>', unsafe_allow_html=True)

# Generate spaces for calculations
spot_prices_space = np.linspace(min_spot_price, max_spot_price, grid_size)
volatilities_space = np.linspace(min_vol, max_vol, grid_size)

# Tabs for different views
tab1, tab2, tab3 = st.tabs(["Option's Fair Value Heatmap", "Option's P&L Heatmap", "Expected Underlying Distribution"])

with tab1:
    st.write("Explore different contract's values given variations in Spot Prices and Annualized Volatilities")
    
    # Calculate matrices using threading
    start_time = time.time()
    
    # Use threading for both call and put calculations
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_call = executor.submit(HeatMapMatrix, spot_prices_space, volatilities_space, selected_strike, risk_free_rate, days_to_maturity, 'C')
        future_put = executor.submit(HeatMapMatrix, spot_prices_space, volatilities_space, selected_strike, risk_free_rate, days_to_maturity, 'P')
        
        output_matrix_c = future_call.result()
        output_matrix_p = future_put.result()
    
    # Create interactive heatmaps with Plotly
    spot_labels = [f"{x:.1f}" for x in spot_prices_space]
    vol_labels = [f"{x:.2f}" for x in volatilities_space]
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Call Options Heatmap', 'Put Options Heatmap'),
        vertical_spacing=0.15
    )
    
    # Call heatmap
    fig.add_trace(
        go.Heatmap(
            z=output_matrix_c.T,
            x=spot_labels,
            y=vol_labels,
            text=[[f"{val:.1f}" for val in row] for row in output_matrix_c.T],
            texttemplate="%{text}",
            textfont={"size": 10},
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Call Value", x=1.02, len=0.4, y=0.75)
        ),
        row=1, col=1
    )
    
    # Put heatmap
    fig.add_trace(
        go.Heatmap(
            z=output_matrix_p.T,
            x=spot_labels,
            y=vol_labels,
            text=[[f"{val:.1f}" for val in row] for row in output_matrix_p.T],
            texttemplate="%{text}",
            textfont={"size": 10},
            colorscale='Plasma',
            showscale=True,
            colorbar=dict(title="Put Value", x=1.02, len=0.4, y=0.25)
        ),
        row=2, col=1
    )
    
    fig.update_xaxes(title_text="Spot Price", row=1, col=1)
    fig.update_xaxes(title_text="Spot Price", row=2, col=1)
    fig.update_yaxes(title_text="Annualized Volatility", row=1, col=1)
    fig.update_yaxes(title_text="Annualized Volatility", row=2, col=1)
    
    fig.update_layout(height=800, showlegend=False)
    
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.write("Explore different expected P&L's from a specific contract trade given variations in the Spot Price and Annualized Volatility")
    
    if 'output_matrix_c' in locals() and 'output_matrix_p' in locals():
        call_pl = output_matrix_c.T - option_purchase_price - 2 * transaction_cost
        put_pl = output_matrix_p.T - option_purchase_price - 2 * transaction_cost
        pl_options = [call_pl, put_pl]
        
        selection = 0 if trade_type == 'Call' else 1
        contract_prices = [call_price, put_price]
        
        specific_contract_pl = contract_prices[selection] - option_purchase_price - 2 * transaction_cost
        st.markdown(f'<div class="metric-container"><h3>Expected P&L given selected parameters</h3><h2>${specific_contract_pl:.2f}</h2></div>', unsafe_allow_html=True)
        
        # Create P&L heatmap
        fig_pl = go.Figure(data=go.Heatmap(
            z=pl_options[selection],
            x=spot_labels,
            y=vol_labels,
            text=[[f"{val:.1f}" for val in row] for row in pl_options[selection]],
            texttemplate="%{text}",
            textfont={"size": 10},
            colorscale='RdBu',
            zmid=0,
            showscale=True,
            colorbar=dict(title="P&L")
        ))
        
        fig_pl.update_layout(
            title=f'{trade_type} Expected P&L Heatmap',
            xaxis_title="Spot Price",
            yaxis_title="Annualized Volatility",
            height=600
        )
        
        st.plotly_chart(fig_pl, use_container_width=True)
    else:
        st.info("Please calculate the fair value heatmaps first in Tab 1")

with tab3:
    st.write('Calculate the expected distribution of the underlying asset price, the option premium and the p&l from trading the option')
    
    with st.expander("See methodology"):
        st.markdown('<div class="expander-content">', unsafe_allow_html=True)
        st.write('The distribution is obtained by simulating $N$ times the underlying asset price as a geometric brownian process during a specified time period.' \
        ' The function $S : [0, \\infty) \\mapsto [0, \\infty) $ will describe the stochastic process as: ')
        st.latex('S(t) = S(0) e^{(\\mu - \\sigma^2 / 2)t + \\sigma W(t)} ')
        st.write('Where $\\mu$ is the risk free rate, $\\sigma$ the annualized volatility of the asset you want to simulate and $S(0)$ the asset price at the beginning (spot price)')
        st.markdown('</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        ns = st.slider('Number of simulations ($N$)', 100, 10000, 1000, 10)
    with col2:
        s_selection = st.radio('Select time interval', ['Days', 'Hours', 'Minutes'], horizontal=True, 
                              help='The time interval each price point will represent. This option is merely for visual purposes.')
    with col3:
        timeshot = st.slider("Select chart's timestamp (days/year)", 0.0, days_to_maturity / 365, days_to_maturity / 365, format="%.3f") 

    if s_selection == 'Days':
        step = days_to_maturity 
    elif s_selection == 'Hours':
        step = days_to_maturity * 24 
    elif s_selection == 'Minutes':
        step = days_to_maturity * 24 * 60 
    
    # Generate simulation paths using threading
    simulation_paths = simulate_paths(ns, days_to_maturity, step, volatility, risk_free_rate, underlying_price)
    
    def get_option_price(K, St, option_type='Call'):
        dynamic_index = -int(step - timeshot * 365 * (step/days_to_maturity) + 1)
        try: 
            if option_type == 'Call':
                expiration_price = np.maximum(St[dynamic_index, :] - K, 0)
            elif option_type == 'Put':
                expiration_price = np.maximum(K - St[dynamic_index, :], 0)
        except Exception as e:
            st.error(f'Error in option price calculation: {e}')
            return np.zeros(St.shape[1])
        return expiration_price

    option_prices = get_option_price(selected_strike, simulation_paths, trade_type)
    pl_results = option_prices - option_purchase_price - 2 * transaction_cost

    # Calculate probabilities
    otm_probability = round(sum(option_prices == 0) / len(option_prices), 3)
    itm_probability = round(1 - otm_probability, 3)
    positive_pl_proba = round(sum(pl_results > 0) / len(pl_results), 3)

    st.subheader('Results')

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            f'<div class="metric-container"><h3>In-the-money probability</h3><h2>{itm_probability}</h2></div>', 
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            f'<div class="metric-container"><h3>Out-the-money probability</h3><h2>{otm_probability}</h2></div>', 
            unsafe_allow_html=True
        )
    with col3:
        st.markdown(
            f'<div class="metric-container"><h3>Positive P&L probability</h3><h2>{positive_pl_proba}</h2></div>', 
            unsafe_allow_html=True
        )

    # Create interactive plots with Plotly
    col1, col2 = st.columns(2)
    
    with col1:
        # Underlying asset price distribution
        index_to_use = -int(step - timeshot * step + 1)
        underlying_prices = simulation_paths[index_to_use, :]
        
        fig1 = px.histogram(
            x=underlying_prices, 
            nbins=30,
            title=f'Expected underlying asset price distribution at day {int(timeshot * 365)}',
            labels={'x': 'Price', 'y': 'Frequency'},
            opacity=0.7
        )
        
        # Add strike price line
        fig1.add_vline(x=selected_strike, line_dash="dash", line_color="red", 
                      annotation_text="Strike price", annotation_position="top")
        
        # Add KDE curve
        kde_x = np.linspace(underlying_prices.min(), underlying_prices.max(), 100)
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(underlying_prices)
        kde_y = kde(kde_x)
        
        # Scale KDE to match histogram
        hist_counts, hist_edges = np.histogram(underlying_prices, bins=30)
        kde_scale = hist_counts.max() / kde_y.max()
        kde_y_scaled = kde_y * kde_scale
        
        fig1.add_trace(go.Scatter(
            x=kde_x, y=kde_y_scaled,
            mode='lines',
            name='Density',
            line=dict(color='orange', width=2)
        ))
        
        fig1.update_layout(height=400, showlegend=True)
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        # Option premium distribution
        fig2 = px.histogram(
            x=option_prices,
            nbins=25,
            title=f'Expected {trade_type} premium at day {int(timeshot * 365)}',
            labels={'x': 'Price', 'y': 'Frequency'},
            opacity=0.7,
            color_discrete_sequence=['green']
        )
        
        # Add KDE for option prices
        if len(np.unique(option_prices)) > 1:  # Only if there's variation
            kde_opt = gaussian_kde(option_prices)
            kde_x_opt = np.linspace(option_prices.min(), option_prices.max(), 100)
            kde_y_opt = kde_opt(kde_x_opt)
            
            hist_counts_opt, _ = np.histogram(option_prices, bins=25)
            kde_scale_opt = hist_counts_opt.max() / kde_y_opt.max()
            kde_y_opt_scaled = kde_y_opt * kde_scale_opt
            
            fig2.add_trace(go.Scatter(
                x=kde_x_opt, y=kde_y_opt_scaled,
                mode='lines',
                name='Density',
                line=dict(color='darkgreen', width=2)
            ))
        
        fig2.update_layout(height=200, showlegend=True)
        st.plotly_chart(fig2, use_container_width=True)

        # P&L distribution
        fig3 = px.histogram(
            x=pl_results,
            nbins=25,
            title=f'Expected P&L distribution at day {int(timeshot * 365)}',
            labels={'x': 'P&L', 'y': 'Frequency'},
            opacity=0.7,
            color_discrete_sequence=['purple']
        )
        
        # Add zero line
        fig3.add_vline(x=0, line_dash="dash", line_color="gray", 
                      annotation_text="Break-even", annotation_position="top")
        
        # Add KDE for P&L
        if len(np.unique(pl_results)) > 1:
            kde_pl = gaussian_kde(pl_results)
            kde_x_pl = np.linspace(pl_results.min(), pl_results.max(), 100)
            kde_y_pl = kde_pl(kde_x_pl)
            
            hist_counts_pl, _ = np.histogram(pl_results, bins=25)
            kde_scale_pl = hist_counts_pl.max() / kde_y_pl.max()
            kde_y_pl_scaled = kde_y_pl * kde_scale_pl
            
            fig3.add_trace(go.Scatter(
                x=kde_x_pl, y=kde_y_pl_scaled,
                mode='lines',
                name='Density',
                line=dict(color='indigo', width=2)
            ))
        
        fig3.update_layout(height=200, showlegend=True)
        st.plotly_chart(fig3, use_container_width=True)
