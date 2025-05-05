import csv
from datetime import datetime
import os

LOG_PATH = "logs/model_logs.csv"

def log_model_run(model_name, metric, value):
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([datetime.now(), model_name, metric, value]) 