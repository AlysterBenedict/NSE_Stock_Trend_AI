import requests
import json

def test_sentiment():
    url = "http://127.0.0.1:5000/get-sentiment"
    payload = {"stock_name": "Infosys"}
    
    try:
        response = requests.post(url, json=payload)
        print(f"Status: {response.status_code}")
        data = response.json()
        print(
            json.dumps(
                {
                    "score": data.get("score"),
                    "label": data.get("label"),
                    "engine": data.get("engine"),
                    "summary": data.get("summary"),
                    "news_count": len(data.get("news", [])),
                },
                indent=2,
            )
        )
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_sentiment()
