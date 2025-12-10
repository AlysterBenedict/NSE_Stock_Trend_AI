import requests
import json

url = 'http://127.0.0.1:5000/get-ai-insights'
payload = {'user_question': 'what is CAGR'}

try:
    response = requests.post(url, json=payload)
    print(f"Status Code: {response.status_code}")
    print("Response JSON:")
    print(json.dumps(response.json(), indent=2))
except Exception as e:
    print(f"Request failed: {e}")
