import requests

def get_champion_forecast(days: int):
    try:
        response = requests.post(
            "http://127.0.0.1:8000/champion_forecast",
            json={"days": days}
        )
        response.raise_for_status()
        return response.json()["forecast"]
    except requests.RequestException as e:
        print(f"Request failed: {e}")
        return None
