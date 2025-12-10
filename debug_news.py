import yfinance as yf
import json

def test_news():
    tickers = ["INFY.NS", "RELIANCE.NS", "AAPL"]
    for t in tickers:
        print(f"\n--- Checking {t} ---")
        try:
            tick = yf.Ticker(t)
            news = tick.news
            print(f"Type: {type(news)}")
            print(f"Length: {len(news)}")
            if len(news) > 0:
                print(json.dumps(news[0], indent=2))
            else:
                print("No news items found.")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    test_news()
