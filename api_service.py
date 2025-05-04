import requests

def get_champion_prediction(payload: dict):
    """
    Sends a POST request to the FastAPI champion model server.

    Args:
        payload (dict): Must contain lag_1 to lag_7, is_promo, is_holiday.

    Returns:
        dict: API response with forecast and status or error.
    """
    try:
        response = requests.post(
            "http://127.0.0.1:8000/predict",  # Adjust if hosted elsewhere
            json=payload
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        return {"error": f"API request failed: {str(e)}"}
