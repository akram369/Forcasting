import os
import json
from datetime import datetime
import joblib

# Define base paths
MODEL_DIR = "models"
METADATA_PATH = "metadata/version_log.json"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs("metadata", exist_ok=True)

# Load or initialize metadata
def load_version_log():
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, "r") as f:
            return json.load(f)
    return {}

def save_version_log(log):
    with open(METADATA_PATH, "w") as f:
        json.dump(log, f, indent=4)

# Save a new model version
def save_model_version(model, model_type, metrics):
    log = load_version_log()
    model_folder = os.path.join(MODEL_DIR, model_type)
    os.makedirs(model_folder, exist_ok=True)

    version_id = f"v{len(log.get(model_type, [])) + 1}"
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"model_{version_id}_{timestamp}.pkl"
    filepath = os.path.join(model_folder, filename)

    joblib.dump(model, filepath)

    entry = {
        "version": version_id,
        "timestamp": timestamp,
        "filepath": filepath,
        "metrics": metrics
    }
    log.setdefault(model_type, []).append(entry)
    save_version_log(log)
    return version_id, filepath

# Load specific model version
def load_model_version(model_type, version_id):
    log = load_version_log()
    for entry in log.get(model_type, []):
        if entry["version"] == version_id:
            return joblib.load(entry["filepath"])
    return None
