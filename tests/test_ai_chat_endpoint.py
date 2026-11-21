import requests

BASE_URL = "http://127.0.0.1:5000"


def test_query(question):
    print(f"\n--- Testing Query: '{question}' ---")
    payload = {"user_question": question}
    try:
        response = requests.post(f"{BASE_URL}/get-ai-insights", json=payload)
        if response.status_code == 200:
            print("Response:", response.json().get("insights", "")[:200] + "...")
        else:
            print("Error:", response.status_code, response.text)
    except Exception as e:
        print("Request Failed:", e)


if __name__ == "__main__":
    test_query("What is the current price of Apple?")
    test_query("Predict the stock price for Nvidia 30 days from today")
    test_query("Should I invest in Reliance Industries?")
    test_query("Analyze the trend for Tata Motors")
    test_query("what is CAGR")
