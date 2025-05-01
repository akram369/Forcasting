from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Demand Forecasting", layout="wide")
st.title("📦 Predictive Demand Forecasting Dashboard")

# Upload CSV or Excel
uploaded_file = st.file_uploader("Upload your dataset (.csv or .xlsx)", type=["csv", "xlsx"])

if uploaded_file:
    try:
        # Read full file
        if uploaded_file.name.endswith(".csv"):
            raw_df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(".xlsx"):
            raw_df = pd.read_excel(uploaded_file, sheet_name=0)

        # Only keep InvoiceDate and Quantity
        df = raw_df[['InvoiceDate', 'Quantity']].copy()
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
        st.success("✅ Loaded only 'InvoiceDate' and 'Quantity' columns!")

    except Exception as e:
        st.error(f"❌ File Error: {e}")
        st.stop()

    # Daily aggregation
    daily_demand = df.set_index('InvoiceDate').resample('D')['Quantity'].sum().fillna(0)
    data = daily_demand.to_frame(name='Quantity')
    data['is_promo'] = (data.index.weekday == 4).astype(int)
    data['is_holiday'] = data.index.isin(['2010-12-24', '2010-12-25', '2011-01-01']).astype(int)
    for lag in range(1, 8):
        data[f'lag_{lag}'] = data['Quantity'].shift(lag)
    data.dropna(inplace=True)

    st.subheader("📊 Daily Demand")
    st.line_chart(daily_demand)

    # Select model
    model_choice = st.selectbox("Choose Forecasting Model", ["XGBoost", "ARIMA", "LSTM"])

    if model_choice == "XGBoost":
        X = data.drop("Quantity", axis=1)
        y = data["Quantity"]
        model = xgb.XGBRegressor()
        model.fit(X, y)
        y_pred = model.predict(X)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        st.metric("📉 XGBoost RMSE", f"{rmse:.2f}")

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(data.index, y, label='Actual')
        ax.plot(data.index, y_pred, label='Forecast', color='red')
        ax.legend()
        st.pyplot(fig)

    elif model_choice == "ARIMA":
        train_size = int(len(daily_demand) * 0.8)
        train = daily_demand[:train_size]
        test = daily_demand[train_size:]
        model = ARIMA(train, order=(5, 1, 2))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=len(test))
        forecast.index = test.index
        rmse = np.sqrt(mean_squared_error(test, forecast))
        st.metric("📉 ARIMA RMSE", f"{rmse:.2f}")

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(train.index, train, label='Train')
        ax.plot(test.index, test, label='Actual')
        ax.plot(test.index, forecast, label='Forecast', color='red')
        ax.legend()
        st.pyplot(fig)

    elif model_choice == "LSTM":
        scaler = MinMaxScaler()
        scaled_qty = scaler.fit_transform(data[['Quantity']])

        def create_sequences(data, window_size=30):
            X, y = [], []
            for i in range(len(data) - window_size):
                X.append(data[i:i+window_size])
                y.append(data[i+window_size])
            return np.array(X), np.array(y)

        window = 30
        X_seq, y_seq = create_sequences(scaled_qty, window)
        split = int(0.8 * len(X_seq))
        X_train, X_test = X_seq[:split], X_seq[split:]
        y_train, y_test = y_seq[:split], y_seq[split:]

        model = Sequential()
        model.add(LSTM(64, activation='relu', input_shape=(window, 1)))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_train, y_train, epochs=20, batch_size=16, verbose=0)

        y_pred = model.predict(X_test)
        y_pred_inv = scaler.inverse_transform(y_pred)
        y_test_inv = scaler.inverse_transform(y_test)

        rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
        st.metric("📉 LSTM RMSE", f"{rmse:.2f}")

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(range(len(y_test_inv)), y_test_inv, label='Actual')
        ax.plot(range(len(y_pred_inv)), y_pred_inv, label='Forecast', color='red')
        ax.legend()
        st.pyplot(fig)
