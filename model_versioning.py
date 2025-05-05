import os
import json
from datetime import datetime
import joblib

# Define base paths
VERSION_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_versions")
os.makedirs(VERSION_DIR, exist_ok=True)

def save_model_version(model, model_name, rmse):
    try:
        # Create version directory
        version = f"v{len(os.listdir(VERSION_DIR)) + 1}"
        version_path = os.path.join(VERSION_DIR, version)
        os.makedirs(version_path, exist_ok=True)

        # Save model
        model_path = os.path.join(version_path, "model.pkl")
        joblib.dump(model, model_path)

        # Save metadata
        metadata = {
            "model_name": model_name,
            "rmse": float(rmse),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "version": version
        }
        
        with open(os.path.join(version_path, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=4)

        return version
    except Exception as e:
        print(f"Error saving model version: {e}")
        return None

def load_model_version(version):
    try:
        version_path = os.path.join(VERSION_DIR, version)
        if not os.path.exists(version_path):
            return None
        
        model_path = os.path.join(version_path, "model.pkl")
        if not os.path.exists(model_path):
            return None
        
        return joblib.load(model_path)
    except Exception as e:
        print(f"Error loading model version: {e}")
        return None
