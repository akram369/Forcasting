from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from model_versioning import save_model_version
from utils.model_monitor import log_model_run
from utils.data_loader import load_data
from utils.forecasting_utils import generate_features
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
import json
import requests
import threading


# Champion model path
CHAMPION_MODEL_PATH = "model_versions/{version}/model.pkl"

def get_champion_prediction(input_features: dict, version: str):
    model_path = CHAMPION_MODEL_PATH.format(version=version)
    if not os.path.exists(model_path):
        return {"error": "Champion model not found."}

    model = joblib.load(model_path)

    input_df = pd.DataFrame([input_features])
    forecast = model.predict(input_df)[0]

    return {
        "forecast": float(forecast),
        "model": "Champion XGBoost",
        "status": "success"
    }



warnings.filterwarnings("ignore")

st.set_page_config(page_title="📦 Demand Forecasting", layout="wide")
st.title("📈 Predictive Demand Forecasting Dashboard")

# Load holidays from config/holidays.json
with open("config/holidays.json") as f:
    holiday_dates = pd.to_datetime(json.load(f)["holidays"])

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
    data['is_holiday'] = data.index.isin(holiday_dates).astype(int)
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
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            shap.summary_plot(shap_values, X, show=False)
            st.pyplot(fig)
            plt.clf()

            forecast_df = pd.DataFrame({"Date": data.index, "Actual": y, "Predicted": y_pred})
            log_model_run("XGBoost", rmse)
            save_model_version(model, "XGBoost", rmse)

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
        save_model_version(model_fit, "ARIMA", rmse)

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
    
        # Reshape for LSTM input
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    
        # For plotting later
        test_dates = data.index[window + split:]
    
        def train_lstm():
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
            ax.plot(test_dates, y_test_inv, label="Actual")
            ax.plot(test_dates, y_pred_inv, label="Forecast", color="red")
            ax.set_title("LSTM Forecast vs Actual")
            ax.set_xlabel("Date")
            ax.set_ylabel("Quantity")
            ax.legend()
            st.pyplot(fig)
    
            forecast_df = pd.DataFrame({
                "Date": test_dates,
                "Actual": y_test_inv.flatten(),
                "Predicted": y_pred_inv.flatten()
            })
    
            log_model_run("LSTM", rmse)
            save_model_version(model, "LSTM", rmse)
    
        train_lstm()



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

# === Phase 7: Model Version Viewer ===
st.sidebar.title("🧠 Saved Model Versions")
version_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_versions")
os.makedirs(version_dir, exist_ok=True)

if os.path.exists(version_dir):
    versions = sorted(os.listdir(version_dir))
    if versions:
        selected_version = st.sidebar.selectbox("Select Version Folder", versions)
        meta_path = os.path.join(version_dir, selected_version, "metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                metadata = json.load(f)
            st.sidebar.markdown("#### Model Metadata")
            st.sidebar.json(metadata)
        else:
            st.sidebar.warning("No metadata found.")
    else:
        st.sidebar.info("No saved versions yet. Train a model to create versions.")
else:
    st.sidebar.info("Model version directory not found. Train a model to create the directory.")

# === Phase 8: Model Comparator & Performance Tracker ===
st.sidebar.title("📊 Model Comparator")

if os.path.exists(version_dir):
    version_list = sorted(os.listdir(version_dir))
    if len(version_list) >= 2:
        v1 = st.sidebar.selectbox("Compare Version 1", version_list, key="v1")
        v2 = st.sidebar.selectbox("Compare Version 2", version_list, key="v2")

        def load_metadata(version):
            path = os.path.join(version_dir, version, "metadata.json")
            if os.path.exists(path):
                with open(path, "r") as f:
                    return json.load(f)
            return None

        meta1 = load_metadata(v1)
        meta2 = load_metadata(v2)

        if meta1 and meta2:
            st.subheader("📊 Model Comparison Dashboard")
            col1, col2 = st.columns(2)

            with col1:
                st.markdown(f"### 🔹 {meta1['model_name']} ({v1})")
                st.write(f"**RMSE**: {meta1['rmse']}")
                st.write(f"**Timestamp**: {meta1['timestamp']}")
                st.json(meta1)

            with col2:
                st.markdown(f"### 🔸 {meta2['model_name']} ({v2})")
                st.write(f"**RMSE**: {meta2['rmse']}")
                st.write(f"**Timestamp**: {meta2['timestamp']}")
                st.json(meta2)

            st.markdown("### 📈 RMSE Comparison")
            comparison_df = pd.DataFrame({
                "Version": [v1, v2],
                "Model": [meta1["model_name"], meta2["model_name"]],
                "RMSE": [meta1["rmse"], meta2["rmse"]]
            })
            fig_cmp, ax_cmp = plt.subplots()
            sns.barplot(data=comparison_df, x="Version", y="RMSE", hue="Model", ax=ax_cmp)
            ax_cmp.set_title("RMSE Comparison of Selected Versions")
            st.pyplot(fig_cmp)

            best_model = v1 if meta1["rmse"] < meta2["rmse"] else v2
            st.success(f"🏆 **Champion Model**: {best_model} with RMSE {min(meta1['rmse'], meta2['rmse'])}")
        else:
            st.warning("⚠️ Metadata missing for one or both selected versions.")
    else:
        st.sidebar.info("Need at least 2 versions for comparison. Train more models.")

# === Phase 9: Champion Deployment & Forecast API ===
st.sidebar.title("🏆 Champion Model")

# Identify the model version with the lowest RMSE
champion_meta = None
if os.path.exists(version_dir):
    best_rmse = float("inf")
    for version in os.listdir(version_dir):
        meta_path = os.path.join(version_dir, version, "metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                meta = json.load(f)
                if meta["rmse"] < best_rmse:
                    best_rmse = meta["rmse"]
                    champion_meta = {
                        "version": version,
                        "model_type": meta["model_name"],
                        "rmse": meta["rmse"]
                    }

if champion_meta:
    st.sidebar.success(f"Champion Model: {champion_meta['model_type']} (v{champion_meta['version']})")
    st.sidebar.info(f"RMSE: {champion_meta['rmse']:.2f}")
else:
    st.sidebar.warning("No champion model found. Train a model to create a champion.")

with st.subheader("🌐 Real-time Forecast with Champion"):
    forecast_days = st.number_input("Forecast how many days ahead?", min_value=1, max_value=30, value=5)

    if st.button("Forecast with Champion"):
        if not champion_meta:
            st.error("No champion model available. Please train a model first.")
        else:
            try:
                recent = daily_demand[-7:].values.tolist()
                today = pd.Timestamp.now().normalize()
                is_promo = 1 if today.weekday() == 4 else 0
                is_holiday = 1 if today.strftime("%Y-%m-%d") in holiday_dates else 0

                input_data = {
                    "is_promo": is_promo,
                    "is_holiday": is_holiday
                }
                for j in range(1, 8):
                    input_data[f"lag_{j}"] = recent[-j]

                result = get_champion_prediction(input_data, champion_meta["version"])

                if result.get("status") == "success":
                    st.success(f"Champion forecast for {forecast_days} days ahead: {result['forecast']:.2f}")
                else:
                    st.error(result.get("error", "Unknown error."))

            except Exception as e:
                st.error(f"Error during forecasting: {e}")

# === Phase 10: Anomaly Detection & Email Alerts ===
st.subheader("🚨 Anomaly Detection & Alerting")

send_email = st.toggle("Enable Email Alerts", value=False)
alert_threshold = st.slider("Anomaly Threshold (%)", 5, 100, 20)
actual_input = st.number_input("Enter Actual Demand (today)", min_value=0.0)

# Load email credentials from Streamlit secrets or environment variables
EMAIL_USER = st.secrets.get("email_user") or os.getenv("EMAIL_USER")
EMAIL_PASS = st.secrets.get("email_pass") or os.getenv("EMAIL_PASS")

if st.button("📊 Check for Anomaly"):
    if champion_meta:
        version_path = os.path.join(version_dir, champion_meta["version"])
        model_type = champion_meta["model_type"]

        try:
            if model_type == "XGBoost":
                model = joblib.load(os.path.join(version_path, "model.pkl"))
                recent = daily_demand[-7:].values.tolist()

                # Build today's input
                today = pd.Timestamp.now().normalize()
                is_promo = 1 if today.weekday() == 4 else 0
                is_holiday = 1 if today.strftime("%Y-%m-%d") in holiday_dates else 0
                input_data = {
                    "is_promo": is_promo,
                    "is_holiday": is_holiday
                }
                for j in range(1, 8):
                    input_data[f"lag_{j}"] = recent[-j]
                pred = model.predict(pd.DataFrame([input_data]))[0]

            elif model_type == "ARIMA":
                model = joblib.load(os.path.join(version_path, "model.pkl"))
                pred = model.forecast(steps=1)[0]

            elif model_type == "LSTM":
                model = joblib.load(os.path.join(version_path, "model.pkl"))
                scaler = joblib.load(os.path.join(version_path, "scaler.pkl"))
                recent = daily_demand[-30:].values.reshape(-1, 1)
                scaled_recent = scaler.transform(recent)
                pred = model.predict(scaled_recent.reshape(1, 30, 1))
                pred = scaler.inverse_transform(pred)[0][0]

            # Calculate deviation and check for anomaly
            deviation = abs(pred - actual_input) / max(actual_input, 1) * 100
            st.metric("Predicted", round(pred, 2))
            st.metric("Deviation", f"{deviation:.2f}%")

            if deviation > alert_threshold:
                st.warning(f"⚠️ Anomaly Detected! Deviation: {deviation:.2f}%")
                if send_email and EMAIL_USER and EMAIL_PASS:
                    st.info("Email alert would be sent here (email sending not implemented)")
            else:
                st.success(f"✅ No anomaly detected. Deviation: {deviation:.2f}%")

        except Exception as e:
            st.error(f"Anomaly check error: {e}")
    else:
        st.error("No champion model available for anomaly detection")
