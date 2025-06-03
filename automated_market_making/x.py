import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="AMM Strategy Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .section-header {
        color: #2c3e50;
        font-size: 1.5rem;
        margin: 1rem 0;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and process all data files"""
    try:
        # Load orderbook data
        ob_df = pd.read_csv('orderbook_train.csv')
        
        # Load trades data
        tr_df = pd.read_csv('public_trades_train.csv')
        
        # Load submission results
        submission_df = pd.read_csv('submission.csv')
        
        # Convert timestamps to datetime
        ob_df['datetime'] = pd.to_datetime(ob_df['timestamp'], unit='ms')
        tr_df['datetime'] = pd.to_datetime(tr_df['timestamp'], unit='ms')
        submission_df['datetime'] = pd.to_datetime(submission_df['timestamp'], unit='ms')
        
        return ob_df, tr_df, submission_df
    except FileNotFoundError as e:
        st.error(f"File not found: {e}")
        return None, None, None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None

def calculate_metrics(ob_df, tr_df, submission_df):
    """Calculate key performance metrics"""
    metrics = {}
    
    if ob_df is not None and not ob_df.empty:
        # Orderbook metrics
        metrics['total_timestamps'] = len(ob_df)
        metrics['avg_bid_ask_spread'] = (ob_df['ask_1_price'] - ob_df['bid_1_price']).mean()
        metrics['avg_bid_size'] = ob_df['bid_1_size'].mean()
        metrics['avg_ask_size'] = ob_df['ask_1_size'].mean()
        
        # Price range
        metrics['price_range'] = ob_df['ask_1_price'].max() - ob_df['bid_1_price'].min()
        
        # Mid price calculation
        ob_df_calc = ob_df.copy()
        ob_df_calc['mid_price'] = (ob_df_calc['bid_1_price'] + ob_df_calc['ask_1_price']) / 2
        metrics['price_volatility'] = ob_df_calc['mid_price'].std()
    
    if tr_df is not None and not tr_df.empty:
        # Trading metrics
        metrics['total_trades'] = len(tr_df)
        metrics['total_volume'] = tr_df['size'].sum()
        metrics['avg_trade_size'] = tr_df['size'].mean()
        metrics['buy_trades'] = len(tr_df[tr_df['side'] == 'buy'])
        metrics['sell_trades'] = len(tr_df[tr_df['side'] == 'sell'])
        
        # VWAP calculation
        metrics['vwap'] = (tr_df['price'] * tr_df['size']).sum() / tr_df['size'].sum()
    
    if submission_df is not None and not submission_df.empty:
        # Strategy metrics
        metrics['total_quotes'] = len(submission_df)
        valid_quotes = submission_df.dropna(subset=['bid_price', 'ask_price'])
        if not valid_quotes.empty:
            metrics['avg_quoted_spread'] = (valid_quotes['ask_price'] - valid_quotes['bid_price']).mean()
            metrics['min_quoted_spread'] = (valid_quotes['ask_price'] - valid_quotes['bid_price']).min()
            metrics['max_quoted_spread'] = (valid_quotes['ask_price'] - valid_quotes['bid_price']).max()
    
    return metrics

def plot_orderbook_evolution(ob_df):
    """Plot orderbook price and size evolution"""
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Price Evolution', 'Size Evolution'),
        shared_xaxes=True,
        vertical_spacing=0.1
    )
    
    # Price evolution
    fig.add_trace(
        go.Scatter(x=ob_df['datetime'], y=ob_df['bid_1_price'], 
                  name='Bid Price', line=dict(color='red')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=ob_df['datetime'], y=ob_df['ask_1_price'], 
                  name='Ask Price', line=dict(color='blue')),
        row=1, col=1
    )
    
    # Add mid price
    mid_price = (ob_df['bid_1_price'] + ob_df['ask_1_price']) / 2
    fig.add_trace(
        go.Scatter(x=ob_df['datetime'], y=mid_price, 
                  name='Mid Price', line=dict(color='green', dash='dash')),
        row=1, col=1
    )
    
    # Size evolution
    fig.add_trace(
        go.Scatter(x=ob_df['datetime'], y=ob_df['bid_1_size'], 
                  name='Bid Size', line=dict(color='red'), fill='tonexty'),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=ob_df['datetime'], y=ob_df['ask_1_size'], 
                  name='Ask Size', line=dict(color='blue'), fill='tozeroy'),
        row=2, col=1
    )
    
    fig.update_layout(
        title="Orderbook Evolution Over Time",
        height=600,
        showlegend=True
    )
    
    return fig

def plot_trades_analysis(tr_df):
    """Plot trades analysis"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Trade Prices Over Time', 'Trade Sizes Distribution', 
                       'Buy vs Sell Volume', 'Trade Size Over Time'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"type": "pie"}, {"secondary_y": False}]]
    )
    
    # Trade prices over time
    buy_trades = tr_df[tr_df['side'] == 'buy']
    sell_trades = tr_df[tr_df['side'] == 'sell']
    
    fig.add_trace(
        go.Scatter(x=buy_trades['datetime'], y=buy_trades['price'], 
                  mode='markers', name='Buy Trades', 
                  marker=dict(color='green', size=buy_trades['size']*10)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=sell_trades['datetime'], y=sell_trades['price'], 
                  mode='markers', name='Sell Trades', 
                  marker=dict(color='red', size=sell_trades['size']*10)),
        row=1, col=1
    )
    
    # Trade sizes distribution
    fig.add_trace(
        go.Histogram(x=tr_df['size'], nbinsx=20, name='Trade Size Distribution'),
        row=1, col=2
    )
    
    # Buy vs Sell volume pie chart
    buy_volume = buy_trades['size'].sum()
    sell_volume = sell_trades['size'].sum()
    
    fig.add_trace(
        go.Pie(labels=['Buy Volume', 'Sell Volume'], 
               values=[buy_volume, sell_volume],
               marker_colors=['green', 'red']),
        row=2, col=1
    )
    
    # Trade size over time
    fig.add_trace(
        go.Scatter(x=tr_df['datetime'], y=tr_df['size'], 
                  mode='lines+markers', name='Trade Size',
                  line=dict(color='purple')),
        row=2, col=2
    )
    
    fig.update_layout(
        title="Trading Activity Analysis",
        height=700,
        showlegend=True
    )
    
    return fig

def plot_strategy_performance(ob_df, submission_df):
    """Plot strategy performance"""
    if submission_df.empty:
        return None
        
    # Merge with orderbook data for comparison
    merged = pd.merge(submission_df, ob_df[['timestamp', 'bid_1_price', 'ask_1_price']], 
                     on='timestamp', how='left', suffixes=('_strategy', '_market'))
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Strategy vs Market Quotes', 'Spread Comparison'),
        shared_xaxes=True,
        vertical_spacing=0.1
    )
    
    # Strategy vs Market quotes
    fig.add_trace(
        go.Scatter(x=merged['datetime'], y=merged['bid_price'], 
                  name='Strategy Bid', line=dict(color='green', width=1)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=merged['datetime'], y=merged['ask_price'], 
                  name='Strategy Ask', line=dict(color='orange', width=1)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=merged['datetime'], y=merged['bid_1_price'], 
                  name='Market Bid', line=dict(color='blue', width =1)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=merged['datetime'], y=merged['ask_1_price'], 
                  name='Market Ask', line=dict(color='red', width = 1)),
        row=1, col=1
    )
    
    # Spread comparison
    strategy_spread = merged['ask_price'] - merged['bid_price']
    market_spread = merged['ask_1_price'] - merged['bid_1_price']
    
    fig.add_trace(
        go.Scatter(x=merged['datetime'], y=strategy_spread, 
                  name='Strategy Spread', line=dict(color='purple')),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=merged['datetime'], y=market_spread, 
                  name='Market Spread', line=dict(color='orange')),
        row=2, col=1
    )
    
    fig.update_layout(
        title="AMM Strategy Performance vs Market",
        height=600,
        showlegend=True
    )
    
    return fig

def plot_depth_analysis(ob_df):
    """Plot orderbook depth analysis"""
    # Calculate total depth for each level
    ob_df_calc = ob_df.copy()
    
    # Calculate cumulative sizes
    bid_levels = ['bid_1_size', 'bid_2_size', 'bid_3_size', 'bid_4_size', 'bid_5_size']
    ask_levels = ['ask_1_size', 'ask_2_size', 'ask_3_size', 'ask_4_size', 'ask_5_size']
    
    ob_df_calc['total_bid_depth'] = ob_df_calc[bid_levels].sum(axis=1)
    ob_df_calc['total_ask_depth'] = ob_df_calc[ask_levels].sum(axis=1)
    ob_df_calc['depth_imbalance'] = (ob_df_calc['total_bid_depth'] - ob_df_calc['total_ask_depth']) / (ob_df_calc['total_bid_depth'] + ob_df_calc['total_ask_depth'])
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Total Depth Over Time', 'Depth Imbalance'),
        shared_xaxes=True
    )
    
    # Total depth
    fig.add_trace(
        go.Scatter(x=ob_df_calc['datetime'], y=ob_df_calc['total_bid_depth'], 
                  name='Total Bid Depth', fill='tonexty', 
                  line=dict(color='red')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=ob_df_calc['datetime'], y=ob_df_calc['total_ask_depth'], 
                  name='Total Ask Depth', fill='tozeroy', 
                  line=dict(color='blue')),
        row=1, col=1
    )
    
    # Depth imbalance
    fig.add_trace(
        go.Scatter(x=ob_df_calc['datetime'], y=ob_df_calc['depth_imbalance'], 
                  name='Depth Imbalance', line=dict(color='purple')),
        row=2, col=1
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
    
    fig.update_layout(
        title="Orderbook Depth Analysis",
        height=600,
        showlegend=True
    )
    
    return fig

# Main app
def main():
    st.markdown('<h1 class="main-header">🚀 Automated Market Making Strategy Dashboard</h1>', unsafe_allow_html=True)
    
    # Load data
    with st.spinner('Loading data...'):
        ob_df, tr_df, submission_df = load_data()
    
    if ob_df is None:
        st.error("Failed to load data. Please ensure the CSV files are in the correct location.")
        return
    
    # Sidebar for filters and controls
    st.sidebar.header("📊 Dashboard Controls")
    
    # Data range selection
    if not ob_df.empty:
        max_records = len(ob_df)
        selected_records = st.sidebar.slider(
            "Number of records to analyze", 
            min_value=100, 
            max_value=min(max_records, 5000), 
            value=min(1000, max_records)
        )
        
        ob_df = ob_df.head(selected_records)
        tr_df = tr_df.head(min(len(tr_df), selected_records // 10))
    
    # Calculate metrics
    metrics = calculate_metrics(ob_df, tr_df, submission_df)
    
    # Key Metrics Section
    st.markdown('<div class="section-header">📈 Key Performance Metrics</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Timestamps", f"{metrics.get('total_timestamps', 'N/A'):,}")
        st.metric("Total Trades", f"{metrics.get('total_trades', 'N/A'):,}")
    
    with col2:
        st.metric("Avg Bid-Ask Spread", f"{metrics.get('avg_bid_ask_spread', 0):.4f}")
        st.metric("Total Volume", f"{metrics.get('total_volume', 0):.2f}")
    
    with col3:
        st.metric("Price Volatility", f"{metrics.get('price_volatility', 0):.4f}")
        st.metric("VWAP", f"{metrics.get('vwap', 0):.2f}")
    
    with col4:
        st.metric("Strategy Quotes", f"{metrics.get('total_quotes', 'N/A'):,}")
        st.metric("Avg Strategy Spread", f"{metrics.get('avg_quoted_spread', 0):.4f}")
    
    # Charts Section
    st.markdown('<div class="section-header">📊 Market Data Analysis</div>', unsafe_allow_html=True)
    
    # Orderbook Evolution
    if not ob_df.empty:
        fig_ob = plot_orderbook_evolution(ob_df)
        st.plotly_chart(fig_ob, use_container_width=True)
    
    # Trades Analysis
    if tr_df is not None and not tr_df.empty:
        fig_trades = plot_trades_analysis(tr_df)
        st.plotly_chart(fig_trades, use_container_width=True)
    else:
        st.warning("No trade data available for analysis.")
    
    # Depth Analysis
    if not ob_df.empty:
        fig_depth = plot_depth_analysis(ob_df)
        st.plotly_chart(fig_depth, use_container_width=True)
    
    # Strategy Performance
    st.markdown('<div class="section-header">🎯 Strategy Performance</div>', unsafe_allow_html=True)
    
    if submission_df is not None and not submission_df.empty:
        fig_strategy = plot_strategy_performance(ob_df, submission_df)
        if fig_strategy:
            st.plotly_chart(fig_strategy, use_container_width=True)
        
        # Strategy Statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Strategy Summary")
            valid_quotes = submission_df.dropna(subset=['bid_price', 'ask_price'])
            st.write(f"**Total Quotes Generated:** {len(submission_df):,}")
            st.write(f"**Valid Quotes:** {len(valid_quotes):,}")
            st.write(f"**Quote Success Rate:** {len(valid_quotes)/len(submission_df)*100:.1f}%")
            
            if not valid_quotes.empty:
                spreads = valid_quotes['ask_price'] - valid_quotes['bid_price']
                st.write(f"**Average Spread:** {spreads.mean():.4f}")
                st.write(f"**Spread Std Dev:** {spreads.std():.4f}")
        
        with col2:
            st.subheader("Recent Strategy Quotes")
            st.dataframe(submission_df.tail(10), use_container_width=True)
    else:
        st.warning("No strategy results available for analysis.")
    
    # Raw Data Section
    with st.expander("📋 Raw Data Preview"):
        tab1, tab2, tab3 = st.tabs(["Orderbook", "Trades", "Strategy Results"])
        
        with tab1:
            st.dataframe(ob_df.head(100), use_container_width=True)
        
        with tab2:
            if tr_df is not None and not tr_df.empty:
                st.dataframe(tr_df.head(100), use_container_width=True)
            else:
                st.write("No trade data available")
        
        with tab3:
            if submission_df is not None and not submission_df.empty:
                st.dataframe(submission_df.head(100), use_container_width=True)
            else:
                st.write("No strategy results available")

if __name__ == "__main__":
    main()