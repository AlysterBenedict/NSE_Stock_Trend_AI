
import requests
import json
from datetime import datetime, timedelta

def test_investment_engine():
    url = "http://127.0.0.1:5000/investment-engine"
    
    # 1 month from now
    future_date = (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")
    
    payload = {
        "principal": 50000,
        "withdrawal_date": future_date
    }
    
    print(f"Testing Investment Engine with: {payload}")
    
    try:
        response = requests.post(url, json=payload)
        
        if response.status_code == 200:
            data = response.json()
            print("\n--- SUCCESS ---")
            print("AI Advice:", data.get('ai_advice'))
            print("\nTop Stocks:")
            for stock in data.get('top_stocks', []):
                print(f"- {stock['name']} ({stock['ticker']})")
                print(f"  Profit: {stock['profit_pct']}%")
                print(f"  Stability: {stock['volatility']}% (Volatility)")
                print(f"  NSE Rank: {stock['nse_rank']}")
                print(f"  Sentiment: {stock['sentiment_score']}")
                if 'trend_data' in stock:
                    print(f"  Trend Data Points: {len(stock['trend_data']['prices'])}")
        else:
            print(f"\n--- FAILED ---")
            print(f"Status: {response.status_code}")
            print(f"Error: {response.text}")
            
    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    test_investment_engine()
