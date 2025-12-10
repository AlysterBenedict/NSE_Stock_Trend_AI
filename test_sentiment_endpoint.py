import requests
import json

def test_sentiment():
    url = "http://127.0.0.1:5000/get-sentiment"
    payload = {"stock_name": "Infosys"}
    
    try:
        response = requests.post(url, json=payload)
        data = response.json()
        print("DEBUG SAMPLE FOUND:")
        print(json.dumps(data.get('debug_sample', 'No sample found'), indent=2))
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_sentiment()
