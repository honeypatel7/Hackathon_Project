import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import sys
import os

# Add the directory where ingenium_codemates.py is located
sys.path.append(os.path.abspath("D:/Hackathon/ingenium_codemates_dashboard.py"))

from ingenium_codemates_dashboard import get_tdqn_rewards  # Import function from your model code

# Load datasets
def load_data():
    files = {
        "TSLA": "D:/Hackathon/TSLA.csv",
        "TCS": "D:/Hackathon/TCS.csv",
        "HDFCBANK": "D:/Hackathon/HDFCBANK.csv",
    }
    data = {}
    for name, path in files.items():
        try:
            df = pd.read_csv(path)
            df.columns = df.columns.str.strip()  # Remove spaces from column names
            if "Date" not in df.columns:
                st.error(f"Missing 'Date' column in {name} dataset")
                continue
            data[name] = df
        except FileNotFoundError:
            st.error(f"File not found: {path}")
    return data

data = load_data()

# Sidebar filters
st.sidebar.title("Stock Data Analysis")
selected_stock = st.sidebar.selectbox("Select Stock", list(data.keys()))

# Load selected dataset
df = data[selected_stock]

# Ensure Date column is in datetime format
df['Date'] = pd.to_datetime(df['Date'])
df.sort_values('Date', inplace=True)

# Date filter
start_date = st.sidebar.date_input("Start Date", df['Date'].min())
end_date = st.sidebar.date_input("End Date", df['Date'].max())
filtered_df = df[(df['Date'] >= pd.Timestamp(start_date)) & (df['Date'] <= pd.Timestamp(end_date))]

# Main dashboard title
st.title(f"Stock Price Analysis - {selected_stock}")

# Stock Price Line Chart
fig_price = go.Figure()
fig_price.add_trace(go.Scatter(x=filtered_df['Date'], y=filtered_df['Close'], mode='lines', name='Stock Price'))
fig_price.update_layout(title=f"Stock Price Over Time - {selected_stock}", xaxis_title='Date', yaxis_title='Price')
st.plotly_chart(fig_price)

# Compute SMA and EMA
filtered_df['SMA_50'] = filtered_df['Close'].rolling(window=50).mean()
filtered_df['EMA_50'] = filtered_df['Close'].ewm(span=50, adjust=False).mean()

# SMA and EMA Chart
fig_sma_ema = go.Figure()
fig_sma_ema.add_trace(go.Scatter(x=filtered_df['Date'], y=filtered_df['SMA_50'], mode='lines', name='SMA (50)'))
fig_sma_ema.add_trace(go.Scatter(x=filtered_df['Date'], y=filtered_df['EMA_50'], mode='lines', name='EMA (50)'))
fig_sma_ema.update_layout(title=f"SMA & EMA - {selected_stock}", xaxis_title='Date', yaxis_title='Price')
st.plotly_chart(fig_sma_ema)

# Candlestick chart
fig_candle = go.Figure()
fig_candle.add_trace(go.Candlestick(x=filtered_df['Date'],
                                    open=filtered_df['Open'],
                                    high=filtered_df['High'],
                                    low=filtered_df['Low'],
                                    close=filtered_df['Close'],
                                    name='Candlestick'))
fig_candle.update_layout(title=f"Candlestick Chart for {selected_stock}", xaxis_title='Date', yaxis_title='Price', xaxis_rangeslider_visible=False)
st.plotly_chart(fig_candle)

# Ensure 'Return' column exists in df before calling get_tdqn_rewards
if "Return" not in df.columns:
    df["Return"] = df["Close"].pct_change().fillna(0)  # Calculate daily returns

# Fetch Cumulative Rewards from TDQN Model
try:
    reward_data = get_tdqn_rewards(selected_stock, df)  # Fetch model rewards
    fig_reward = go.Figure()
    fig_reward.add_trace(go.Scatter(x=reward_data['Date'], y=reward_data['Cumulative Reward'], mode='lines', name='Cumulative Reward'))
    fig_reward.update_layout(title=f"Cumulative Rewards Over Time - {selected_stock}", xaxis_title='Date', yaxis_title='Cumulative Reward')
    st.plotly_chart(fig_reward)
except Exception as e:
    st.warning(f"Could not fetch TDQN rewards: {str(e)}")

# Calculate Portfolio Metrics
returns = filtered_df['Close'].pct_change().dropna()
sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if not returns.empty else np.nan
sortino_ratio = returns.mean() / returns[returns < 0].std() * np.sqrt(252) if not returns[returns < 0].empty else np.nan
roi = (filtered_df['Close'].iloc[-1] - filtered_df['Close'].iloc[0]) / filtered_df['Close'].iloc[0] * 100

st.sidebar.subheader("Portfolio Metrics")
st.sidebar.write(f"Sharpe Ratio: {sharpe_ratio:.2f}")
st.sidebar.write(f"Sortino Ratio: {sortino_ratio:.2f}")
st.sidebar.write(f"ROI: {roi:.2f}%")

# Display data
toggle_data = st.checkbox("Show Data Table")
if toggle_data:
    st.dataframe(filtered_df)

