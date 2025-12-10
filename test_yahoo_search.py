import requests
import json

def search_yahoo(query):
    print(f"Searching for: {query}")
    url = f"https://query2.finance.yahoo.com/v1/finance/search?q={query}&quotesCount=5&newsCount=0"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        response = requests.get(url, headers=headers)
        data = response.json()
        print(json.dumps(data, indent=2))
        
        if 'quotes' in data and len(data['quotes']) > 0:
            first_result = data['quotes'][0]
            symbol = first_result.get('symbol')
            print(f"\nTop Symbol: {symbol}")
    except Exception as e:
        print(f"Error: {e}")

search_yahoo("Apple")
search_yahoo("Nvidia")
search_yahoo("Tesla")
search_yahoo("Reliance") # Control
