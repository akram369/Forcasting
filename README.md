# 📈 Predictive Demand Forecasting Dashboard

A powerful Streamlit-based dashboard for demand forecasting using multiple machine learning models.

## 🚀 Features

- **Multiple Model Support**: XGBoost, ARIMA, and LSTM
- **Real-time Forecasting**: Instant predictions with champion model
- **Model Versioning**: Track and compare different model versions
- **Anomaly Detection**: Identify unusual demand patterns
- **SHAP Analysis**: Understand feature importance (XGBoost)
- **Export Capabilities**: Download forecasts and visualizations

## 📋 Requirements

- Python 3.8+
- Streamlit
- XGBoost
- TensorFlow
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- SHAP

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/akram369/predictive-forecasting-dashboard.git
cd predictive-forecasting-dashboard
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🎮 Usage

1. Start the dashboard:
```bash
streamlit run app.py
```

2. Upload your data file (CSV/Excel with InvoiceDate & Quantity columns)
3. Select and train a model
4. Use the champion model for predictions
5. Monitor and compare model performance

## 📊 Dashboard Components

- **Data Upload**: Upload and view your demand data
- **Model Training**: Train XGBoost, ARIMA, or LSTM models
- **Champion Model**: Automatic selection of best performing model
- **Real-time Forecast**: Get instant predictions
- **Anomaly Detection**: Monitor for unusual patterns
- **Model Comparison**: Compare different model versions
- **Export Tools**: Download results and visualizations

## 📝 Project Structure

```
predictive-forecasting-dashboard/
├── app.py                           # Main Streamlit dashboard
├── requirements.txt                 # Python dependencies
├── README.md                        # Project documentation
├── data/
│   └── processed_data.csv           # Preprocessed dataset
├── models/                          # Saved models
├── logs/                            # Model run logs
├── utils/
│   ├── data_loader.py              # Data loading utilities
│   ├── forecasting_utils.py        # Forecasting functions
│   └── model_monitor.py            # Model monitoring
└── .streamlit/
    └── config.toml                 # Streamlit configuration
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
