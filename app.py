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
import shap
import io
import zipfile
import warnings
import datetime

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Demand Forecasting", layout="wide")
st.title("📦 Predictive Demand Forecasting Dashboard")

# Upload CSV or Excel
uploaded_file = st.file_uploader("Upload your dataset (.csv or .xlsx)", type=["csv", "xlsx"])

if uploaded_file:
    try:
        if uploaded_file.name.endswith(".csv"):
            raw_df = pd.read_csv(uploaded_file, encoding="ISO-8859-1")
        elif uploaded_file.name.endswith(".xlsx"):
            raw_df = pd.read_excel(uploaded_file)
        df = raw_df[['InvoiceDate', 'Quantity']].copy()
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
        st.success("✅ Loaded only 'InvoiceDate' and 'Quantity' columns!")
    except Exception as e:
        st.error(f"❌ File Error: {e}")
        st.stop()

    # Prepare data
    daily_demand = df.set_index('InvoiceDate').resample('D')['Quantity'].sum().fillna(0)
    data = daily_demand.to_frame(name='Quantity')
    data['is_promo'] = (data.index.weekday == 4).astype(int)
    data['is_holiday'] = data.index.isin(['2010-12-24', '2010-12-25', '2011-01-01']).astype(int)
    for lag in range(1, 8):
        data[f'lag_{lag}'] = data['Quantity'].shift(lag)
    data.dropna(inplace=True)

    st.subheader("📊 Daily Demand")
    st.line_chart(daily_demand)

    # Model selection
    model_choice = st.selectbox("Choose Forecasting Model", ["XGBoost", "ARIMA", "LSTM"])
    forecast_df = pd.DataFrame()
    fig = plt.figure(figsize=(10, 4))

    if model_choice == "XGBoost":
        X = data.drop("Quantity", axis=1)
        y = data["Quantity"]
        model = xgb.XGBRegressor()
        model.fit(X, y)
        y_pred = model.predict(X)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        st.metric("📉 XGBoost RMSE", f"{rmse:.2f}")

        forecast_df = pd.DataFrame({"Date": data.index, "Actual": y, "Predicted": y_pred})
        plt.plot(forecast_df["Date"], forecast_df["Actual"], label='Actual')
        plt.plot(forecast_df["Date"], forecast_df["Predicted"], label='Forecast', color='red')
        plt.legend()
        st.pyplot(fig)

        # 🔍 Phase 4: SHAP Explainability
        st.subheader("🔍 Feature Importance (SHAP)")
        explainer = shap.Explainer(model)
        shap_values = explainer(X)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        shap.summary_plot(shap_values, X, plot_type="bar")
        st.pyplot(bbox_inches='tight')

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

        forecast_df = pd.DataFrame({"Date": test.index, "Actual": test.values, "Predicted": forecast.values})
        plt.plot(train.index, train, label='Train')
        plt.plot(test.index, test, label='Actual')
        plt.plot(test.index, forecast, label='Forecast', color='red')
        plt.legend()
        st.pyplot(fig)

    elif model_choice == "LSTM":
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

        forecast_df = pd.DataFrame({
            "Index": list(range(len(y_test_inv))),
            "Actual": y_test_inv.flatten(),
            "Predicted": y_pred_inv.flatten()
        })
        plt.plot(forecast_df["Index"], forecast_df["Actual"], label='Actual')
        plt.plot(forecast_df["Index"], forecast_df["Predicted"], label='Forecast', color='red')
        plt.legend()
        st.pyplot(fig)

    # Phase 1: Download CSV & Plot
    if not forecast_df.empty:
        csv_buffer = io.StringIO()
        forecast_df.to_csv(csv_buffer, index=False)
        img_buffer = io.BytesIO()
        fig.savefig(img_buffer, format='png')
        img_buffer.seek(0)
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, mode="w") as zf:
            zf.writestr("forecast.csv", csv_buffer.getvalue())
            zf.writestr("forecast_plot.png", img_buffer.getvalue())
        st.download_button(
            label="📦 Download Forecast Bundle (CSV + Plot)",
            data=zip_buffer.getvalue(),
            file_name="forecast_bundle.zip",
            mime="application/zip"
        )

    # ✅ Phase 5: Simulated Model Monitoring
    st.subheader("🔁 Continuous Monitoring & Retraining")
    if st.button("🔄 Simulate Retraining Now"):
        retrain_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.success(f"Model retrained at {retrain_time}")
        st.info("📈 This simulates a scheduled retraining process.")
