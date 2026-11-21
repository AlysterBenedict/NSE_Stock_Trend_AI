import requests
import json

try:
    print("Testing /get-market-data endpoint...")
    response = requests.get('http://127.0.0.1:5000/get-market-data')
    
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"Data received: {len(data)} items")
        if len(data) > 0:
            print("First item sample:")
            print(json.dumps(data[0], indent=2))
        else:
            print("Data is empty list []")
    else:
        print("Error response:")
        print(response.text)

except Exception as e:
    print(f"Request failed: {e}")
