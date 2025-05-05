import pandas as pd

def load_data(path='data/processed_data.csv'):
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        return None 