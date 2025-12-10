import requests
import json
import time

BASE_URL = "http://127.0.0.1:5000"

def test_query(question):
    print(f"\n--- Testing Query: '{question}' ---")
    payload = {
        "user_question": question
    }
    try:
        response = requests.post(f"{BASE_URL}/get-ai-insights", json=payload)
        if response.status_code == 200:
            print("Response:", response.json().get('insights')[:200] + "...") # Print first 200 chars
        else:
            print("Error:", response.status_code, response.text)
    except Exception as e:
        print("Request Failed:", e)

# 1. PRICE INTENT
test_query("What is the current price of Apple?")

# 2. PREDICT INTENT
test_query("Predict the stock price for Nvidia 30 days from today")

# 3. ADVICE INTENT
test_query("Should I invest in Reliance Industries?")

# 4. ANALYZE INTENT
test_query("Analyze the trend for Tata Motors")

