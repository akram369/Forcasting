from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import shap
import seaborn as sns
import joblib
import io
import zipfile
import os
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(page_title="📦 Demand Forecasting", layout="wide")
st.title("📈 Predictive Demand Forecasting Dashboard")

# === Phase 6: Auto Retraining Log Function ===
def log_model_run(model_name, rmse):
    log_path = "model_logs.csv"
    entry = pd.DataFrame([{
        "Model": model_name,
        "RMSE": round(rmse, 2),
        "Timestamp": pd.Timestamp.now()
    }])
    if os.path.exists(log_path):
        existing = pd.read_csv(log_path)
        pd.concat([existing, entry]).to_csv(log_path, index=False)
    else:
        entry.to_csv(log_path, index=False)

# === File Upload ===
uploaded_file = st.file_uploader("Upload CSV or Excel (InvoiceDate & Quantity required)", type=["csv", "xlsx"])
if uploaded_file:
    try:
        if uploaded_file.name.endswith(".csv"):
            df_raw = pd.read_csv(uploaded_file, encoding="ISO-8859-1")
        else:
            df_raw = pd.read_excel(uploaded_file)
        df = df_raw[['InvoiceDate', 'Quantity']].copy()
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
        st.success("✅ Dataset loaded successfully!")
    except Exception as e:
        st.error(f"File Error: {e}")
        st.stop()

    # === Data Preparation ===
    daily_demand = df.set_index('InvoiceDate').resample('D')['Quantity'].sum().fillna(0)
    data = daily_demand.to_frame(name='Quantity')
    data['is_promo'] = (data.index.weekday == 4).astype(int)
    data['is_holiday'] = data.index.isin(['2010-12-24', '2010-12-25', '2011-01-01']).astype(int)
    for lag in range(1, 8):
        data[f'lag_{lag}'] = data['Quantity'].shift(lag)
    data.dropna(inplace=True)

    st.subheader("📊 Daily Demand Overview")
    st.line_chart(daily_demand)

    # === Model Selection ===
    model_choice = st.selectbox("Select Forecasting Model", ["XGBoost", "ARIMA", "LSTM"])
    fig, ax = plt.subplots(figsize=(10, 4))
    forecast_df = pd.DataFrame()

    if model_choice == "XGBoost":
        st.subheader("🚀 XGBoost Forecasting")
        from xgboost import XGBRegressor
        X = data.drop("Quantity", axis=1).copy()
        y = data["Quantity"].copy()

        try:
            model = XGBRegressor()
            model.fit(X, y)
            y_pred = model.predict(X)

            rmse = np.sqrt(mean_squared_error(y, y_pred))
            st.metric("📉 XGBoost RMSE", f"{rmse:.2f}")

            ax.plot(data.index, y, label="Actual")
            ax.plot(data.index, y_pred, label="Forecast", color="red")
            ax.legend()
            st.pyplot(fig)

            # SHAP Summary
            st.subheader("🔍 SHAP Summary Plot")
            explainer = shap.Explainer(model, X)
            shap_values = explainer(X)
            st.set_option('deprecation.showPyplotGlobalUse', False)
            shap.summary_plot(shap_values, X)
            st.pyplot()

            forecast_df = pd.DataFrame({"Date": data.index, "Actual": y, "Predicted": y_pred})
            log_model_run("XGBoost", rmse)

        except Exception as e:
            st.error(f"⚠️ Error during XGBoost modeling: {e}")

    elif model_choice == "ARIMA":
        st.subheader("📈 ARIMA Forecasting")
        train_size = int(len(daily_demand) * 0.8)
        train, test = daily_demand[:train_size], daily_demand[train_size:]
        model = ARIMA(train, order=(5, 1, 2))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=len(test))
        forecast.index = test.index
        rmse = np.sqrt(mean_squared_error(test, forecast))
        st.metric("📉 ARIMA RMSE", f"{rmse:.2f}")
        ax.plot(train.index, train, label="Train")
        ax.plot(test.index, test, label="Test")
        ax.plot(test.index, forecast, label="Forecast", color="red")
        ax.legend()
        st.pyplot(fig)
        forecast_df = pd.DataFrame({"Date": test.index, "Actual": test.values, "Predicted": forecast.values})
        log_model_run("ARIMA", rmse)

    elif model_choice == "LSTM":
        st.subheader("🤖 LSTM Forecasting")
        scaler = MinMaxScaler()
        scaled_qty = scaler.fit_transform(data[['Quantity']])
        def create_sequences(data, window=30):
            X, y = [], []
            for i in range(len(data) - window):
                X.append(data[i:i+window])
                y.append(data[i+window])
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
        ax.plot(range(len(y_test_inv)), y_test_inv, label="Actual")
        ax.plot(range(len(y_pred_inv)), y_pred_inv, label="Forecast", color="red")
        ax.legend()
        st.pyplot(fig)
        forecast_df = pd.DataFrame({
            "Index": list(range(len(y_test_inv))),
            "Actual": y_test_inv.flatten(),
            "Predicted": y_pred_inv.flatten()
        })
        log_model_run("LSTM", rmse)

    # === Forecast Export ===
    if not forecast_df.empty:
        csv_buf = io.StringIO()
        forecast_df.to_csv(csv_buf, index=False)
        img_buf = io.BytesIO()
        fig.savefig(img_buf, format='png')
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w") as z:
            z.writestr("forecast.csv", csv_buf.getvalue())
            z.writestr("forecast_plot.png", img_buf.getvalue())
        st.download_button("📦 Download Forecast ZIP", data=zip_buf.getvalue(),
                           file_name="forecast_bundle.zip", mime="application/zip")

# === Phase 5: Model Log Viewer ===
st.sidebar.title("📂 Model Run History")
if os.path.exists("model_logs.csv"):
    logs = pd.read_csv("model_logs.csv")
    logs['Timestamp'] = pd.to_datetime(logs['Timestamp'])
    st.sidebar.dataframe(logs.sort_values("Timestamp", ascending=False), height=300)
else:
    st.sidebar.info("No model runs logged yet.")
