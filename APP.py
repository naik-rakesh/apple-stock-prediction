import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import plotly.graph_objects as go
from datetime import timedelta
import os

# 1. PAGE CONFIGURATION
st.set_page_config(page_title="AAPL Neural Quant", layout="wide", page_icon="🍎")
st.title("🍎 Apple Stock Intelligence")
st.markdown("### Stock Prediction Using LSTM Forecasting System")

FEATURE_COLS = [
    'RSI_14', 'MACD', 'MACD_signal', 'MACD_histogram', 'OBV',
    'Stoch_%K', 'Stoch_%D', 'Williams_%R', 'BB_Width',
    'Daily_Return', 'Volatility_20', 'Close'
]

# 2. CACHED ASSET LOADER
@st.cache_resource
def load_assets():
    model = load_model('lstm_model.h5', compile=False) if os.path.exists('lstm_model.h5') else None
    scaler = joblib.load('scaler.pkl') if os.path.exists('scaler.pkl') else None
    return model, scaler

# 3. TECHNICAL INDICATORS & FEATURE ENGINEERING ENGINE
def apply_indicators(df):
    df.columns = [c.strip().capitalize() for c in df.columns]

    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
        df = df.dropna(subset=['Date']).sort_values('Date')

    # Clean currency formatting strings in Close/Price boundaries
    if df['Close'].dtype == 'O':
        df['Close'] = df['Close'].replace('[\$,]', '', regex=True).astype(float)
    if 'Low' in df.columns and df['Low'].dtype == 'O':
        df['Low'] = df['Low'].replace('[\$,]', '', regex=True).astype(float)
    if 'High' in df.columns and df['High'].dtype == 'O':
        df['High'] = df['High'].replace('[\$,]', '', regex=True).astype(float)

    # Calculate Technical Indicators
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI_14'] = 100 - (100 / (1 + (gain / (loss + 1e-9))))

    exp1, exp2 = df['Close'].ewm(span=12).mean(), df['Close'].ewm(span=26).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
    df['MACD_histogram'] = df['MACD'] - df['MACD_signal']

    vol = df['Volume'] if 'Volume' in df.columns else np.zeros(len(df))
    df['OBV'] = (np.sign(df['Close'].diff()) * vol).fillna(0).cumsum()

    l14 = df['Low'].rolling(14).min() if 'Low' in df.columns else df['Close'].rolling(14).min()
    h14 = df['High'].rolling(14).max() if 'High' in df.columns else df['Close'].rolling(14).max()

    df['Stoch_%K'] = 100 * ((df['Close'] - l14) / (h14 - l14 + 1e-9))
    df['Stoch_%D'] = df['Stoch_%K'].rolling(3).mean()
    df['Williams_%R'] = -100 * ((h14 - df['Close']) / (h14 - l14 + 1e-9))

    df['BB_Width'] = (df['Close'].rolling(20).std() * 4) / (df['Close'].rolling(20).mean() + 1e-9)
    df['Daily_Return'] = df['Close'].pct_change()
    df['Volatility_20'] = df['Daily_Return'].rolling(20).std()

    # Target sequence differential used during training models
    df['Close_diff'] = df['Close'].diff()

    return df.dropna()

model, scaler = load_assets()

# 4. INTERFACE LAYER
uploaded_file = st.sidebar.file_uploader("Upload AAPL Historical CSV File", type=["csv"])

if model is None or scaler is None:
    st.error("🚨 **Missing Assets!** Ensure 'lstm_model.h5' and 'scaler.pkl' are placed directly in the repository root directory.")
elif uploaded_file:
    raw_df = pd.read_csv(uploaded_file)
    df = apply_indicators(raw_df)

    st.sidebar.success("✅ Financial Datasets Loaded & Processed")

    # Feature Matrix Breakdown Expansion panel
    with st.expander("📊 View Engineered Technical Features & Core Analytics"):
        st.dataframe(df[FEATURE_COLS].tail(5), use_container_width=True)
        corr = df[FEATURE_COLS].corr()[['Close']].sort_values(by='Close', ascending=False)
        st.bar_chart(corr)

    st.sidebar.divider()
    forecast_days = st.sidebar.slider("Forecast Horizon (Target Days)", 1, 60, 7)

    if st.button("Predict Future Market Close Prices", type="primary"):
        with st.spinner("Executing Recursive LSTM Prediction Loop..."):
            all_features = FEATURE_COLS + ["Close_diff"]
            scaled_data = scaler.transform(df[all_features].values)

            # Feature Array Mapping Indices
            close_idx = 11
            diff_idx = 12

            # Gather min-max scaler parameters safely
            c_min, c_max = scaler.data_min_[close_idx], scaler.data_max_[close_idx]
            d_min, d_max = scaler.data_min_[diff_idx], scaler.data_max_[diff_idx]

            # Build initial lookback tracking slice (60-day window)
            current_seq = scaled_data[-60:, :len(FEATURE_COLS)].reshape(1, 60, len(FEATURE_COLS))
            last_price = df['Close'].iloc[-1]

            forecast_results = []

            # Multi-step forecasting timeline generation loop
            for _ in range(forecast_days):
                pred_scaled = model.predict(current_seq, verbose=0)[0, 0]

                # Denormalize variance to absolute delta change coordinates
                delta_price = pred_scaled * (d_max - d_min + 1e-9) + d_min
                new_price = last_price + delta_price
                forecast_results.append(new_price)

                # Re-scale target variable coordinate points
                new_price_scaled = (new_price - c_min) / (c_max - c_min + 1e-9)

                # Update sequence pipeline vectors
                new_row = current_seq[0, -1, :].copy()
                new_row[close_idx] = new_price_scaled

                current_seq = np.roll(current_seq, -1, axis=1)
                current_seq[0, -1, :] = new_row
                last_price = new_price

            # VISUALIZATION MAPPING
            future_dates = pd.date_range(start=df['Date'].iloc[-1] + timedelta(days=1), periods=forecast_days)
            forecast_df = pd.DataFrame({"Date": future_dates, "Predicted Close Price ($)": forecast_results})

            col1, col2 = st.columns([2, 1])

            with col1:
                st.subheader("Price Trajectory Mapping")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df['Date'].tail(40), y=df['Close'].tail(40), name="Actual Historical Price"))
                fig.add_trace(go.Scatter(x=forecast_df['Date'], y=forecast_df['Predicted Close Price ($)'],
                                         name="LSTM Future Forecast", line=dict(color='orange', width=3)))
                fig.update_layout(template="plotly_dark", hovermode="x unified", margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.subheader("Forecast Estimates Summary")
                st.metric("Current Market Close", f"${df['Close'].iloc[-1]:.2f}")
                st.metric("Horizon Terminal Target", f"${forecast_results[-1]:.2f}",
                          delta=f"{((forecast_results[-1]/df['Close'].iloc[-1])-1)*100:.2f}% Expected Change")
                st.dataframe(forecast_df.set_index("Date"), use_container_width=True)
else:
    st.info("💡 Please upload your historical **AAPL.csv** source file in the sidebar menu to begin execution mapping.")