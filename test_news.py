import yfinance as yf
import json

def check_news():
    try:
        ticker = yf.Ticker("INFY.NS")
        news = ticker.news
        print(json.dumps(news[:3], indent=2))
    except Exception as e:
        print(e)

if __name__ == "__main__":
    check_news()
