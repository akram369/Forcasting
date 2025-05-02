from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import shap
import io, zipfile, os, datetime, joblib
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Demand Forecasting", layout="wide")
st.title("📦 Predictive Demand Forecasting Dashboard")

# Tabs
tab1, tab2, tab3 = st.tabs(["📈 Forecasting", "📊 Model History", "🔁 Retrain Model"])

# Tab 1: Forecasting
with tab1:
    uploaded_file = st.file_uploader("Upload your dataset (.csv or .xlsx)", type=["csv", "xlsx"])

    if uploaded_file:
        try:
            if uploaded_file.name.endswith(".csv"):
                raw_df = pd.read_csv(uploaded_file, encoding="ISO-8859-1")
            else:
                raw_df = pd.read_excel(uploaded_file)

            df = raw_df[['InvoiceDate', 'Quantity']].copy()
            df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
            st.success("✅ Loaded 'InvoiceDate' and 'Quantity' columns!")

        except Exception as e:
            st.error(f"❌ File Error: {e}")
            st.stop()

        daily = df.set_index('InvoiceDate').resample('D')['Quantity'].sum().fillna(0)
        data = daily.to_frame(name='Quantity')
        data['is_promo'] = (data.index.weekday == 4).astype(int)
        data['is_holiday'] = data.index.isin(['2010-12-24', '2010-12-25', '2011-01-01']).astype(int)
        for lag in range(1, 8):
            data[f'lag_{lag}'] = data['Quantity'].shift(lag)
        data.dropna(inplace=True)

        st.subheader("📊 Daily Demand")
        st.line_chart(daily)

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

            plt.plot(data.index, y, label='Actual')
            plt.plot(data.index, y_pred, label='Forecast', color='red')
            plt.legend()
            st.pyplot(fig)

            # SHAP
            explainer = shap.Explainer(model)
            shap_values = explainer(X)
            st.subheader("🔍 SHAP Summary")
            shap.summary_plot(shap_values, X, show=False)
            st.pyplot(bbox_inches='tight')

            forecast_df = pd.DataFrame({"Date": data.index, "Actual": y, "Predicted": y_pred})

        elif model_choice == "ARIMA":
            train_size = int(len(daily) * 0.8)
            train = daily[:train_size]
            test = daily[train_size:]
            model = ARIMA(train, order=(5, 1, 2)).fit()
            forecast = model.forecast(steps=len(test))
            forecast.index = test.index
            rmse = np.sqrt(mean_squared_error(test, forecast))
            st.metric("📉 ARIMA RMSE", f"{rmse:.2f}")

            plt.plot(train.index, train, label='Train')
            plt.plot(test.index, test, label='Actual')
            plt.plot(test.index, forecast, label='Forecast', color='red')
            plt.legend()
            st.pyplot(fig)

            forecast_df = pd.DataFrame({"Date": test.index, "Actual": test.values, "Predicted": forecast.values})

        elif model_choice == "LSTM":
            scaler = MinMaxScaler()
            scaled_qty = scaler.fit_transform(data[['Quantity']])
            def create_seq(data, window=30):
                X, y = [], []
                for i in range(len(data) - window):
                    X.append(data[i:i+window])
                    y.append(data[i+window])
                return np.array(X), np.array(y)

            X_seq, y_seq = create_seq(scaled_qty)
            split = int(0.8 * len(X_seq))
            X_train, X_test = X_seq[:split], X_seq[split:]
            y_train, y_test = y_seq[:split], y_seq[split:]

            model = Sequential([
                LSTM(64, activation='relu', input_shape=(X_train.shape[1], 1)),
                Dense(1)
            ])
            model.compile(optimizer='adam', loss='mse')
            model.fit(X_train, y_train, epochs=20, batch_size=16, verbose=0)

            y_pred = model.predict(X_test)
            y_pred_inv = scaler.inverse_transform(y_pred)
            y_test_inv = scaler.inverse_transform(y_test)
            rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
            st.metric("📉 LSTM RMSE", f"{rmse:.2f}")

            plt.plot(range(len(y_test_inv)), y_test_inv, label='Actual')
            plt.plot(range(len(y_pred_inv)), y_pred_inv, label='Forecast', color='red')
            plt.legend()
            st.pyplot(fig)

            forecast_df = pd.DataFrame({
                "Index": list(range(len(y_test_inv))),
                "Actual": y_test_inv.flatten(),
                "Predicted": y_pred_inv.flatten()
            })

        # Export Forecast + Plot as ZIP
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
                label="📥 Download Forecast Bundle (CSV + Plot)",
                data=zip_buffer.getvalue(),
                file_name="forecast_bundle.zip",
                mime="application/zip"
            )

            # Save logs
            version = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            log_data = pd.DataFrame([{
                "timestamp": version,
                "model": model_choice,
                "rmse": float(rmse)
            }])
            if os.path.exists("model_logs.csv"):
                old_logs = pd.read_csv("model_logs.csv")
                log_data = pd.concat([old_logs, log_data], ignore_index=True)
            log_data.to_csv("model_logs.csv", index=False)
            joblib.dump(model, f"model_{model_choice}_{version}.pkl")


# Tab 2: View Model History
with tab2:
    st.subheader("📚 Model Training History")
    if os.path.exists("model_logs.csv"):
        logs = pd.read_csv("model_logs.csv")
        st.dataframe(logs)
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        logs.plot(x="timestamp", y="rmse", hue="model", ax=ax2, marker='o')
        st.pyplot(fig2)
    else:
        st.info("No training history found.")


# Tab 3: Retrain Button
with tab3:
    st.subheader("🔁 Retrain Models")
    if st.button("📦 Retrain All Models"):
        st.info("Retraining not yet implemented in auto loop.")
        st.success("🔁 You can manually retrain in Tab 1 by re-uploading your data.")
