import numpy as np
import pandas as pd

def generate_features(df, forecast_horizon):
    last_date = df['date'].max()
    future_dates = pd.date_range(start=last_date, periods=forecast_horizon+1, freq='D')[1:]
    return pd.DataFrame({'date': future_dates}) 