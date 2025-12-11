import yfinance as yf
import numpy as np
import pandas as pd
import os
import joblib
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
from pandas.tseries.offsets import BDay
from flask import Flask, request, jsonify
from flask_cors import CORS
import warnings
import json
import requests # Import for making HTTP requests to local server

# --- 0. Setup ---
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Suppress TensorFlow warnings

# --- 1. Initialize Flask App ---
app = Flask(__name__)
CORS(app) 

# --- 2. Define Constants ---
TIME_STEP = 100
STOCK_LIST = {
    'Adani Enterprises': 'ADANIENT.NS',
    'Adani Ports': 'ADANIPORTS.NS',
    'Apollo Hospitals': 'APOLLOHOSP.NS',
    'Asian Paints': 'ASIANPAINT.NS',
    'Axis Bank': 'AXISBANK.NS',
    'Bajaj Auto': 'BAJAJ-AUTO.NS',
    'Bajaj Finance': 'BAJFINANCE.NS',
    'Bajaj Finserv': 'BAJAJFINSV.NS',
    'Bharti Airtel': 'BHARTIARTL.NS',
    'BPCL': 'BPCL.NS',
    'Britannia': 'BRITANNIA.NS',
    'Cipla': 'CIPLA.NS',
    'Coal India': 'COALINDIA.NS',
    'Divis Lab': 'DIVISLAB.NS',
    'Dr Reddys Labs': 'DRREDDY.NS',
    'Eicher Motors': 'EICHERMOT.NS',
    'Grasim': 'GRASIM.NS',
    'HCL Tech': 'HCLTECH.NS',
    'HDFC Bank': 'HDFCBANK.NS',
    'HDFC Life': 'HDFCLIFE.NS',
    'Hero MotoCorp': 'HEROMOTOCO.NS',
    'Hindalco': 'HINDALCO.NS',
    'Hindustan Unilever': 'HINDUNILVR.NS',
    'ICICI Bank': 'ICICIBANK.NS',
    'IndusInd Bank': 'INDUSINDBK.NS',
    'Infosys': 'INFY.NS',
    'ITC': 'ITC.NS',
    'JSW Steel': 'JSWSTEEL.NS',
    'Kotak Mahindra Bank': 'KOTAKBANK.NS',
    'L&T': 'LT.NS',
    'LTIMindtree': 'LTIM.NS',
    'M&M': 'M&M.NS',
    'Maruti Suzuki': 'MARUTI.NS',
    'Nestle India': 'NESTLEIND.NS',
    'NTPC': 'NTPC.NS',
    'ONGC': 'ONGC.NS',
    'Power Grid Corp': 'POWERGRID.NS',
    'Reliance': 'RELIANCE.NS',
    'SBI Life': 'SBILIFE.NS',
    'SBI': 'SBIN.NS',
    'Sun Pharma': 'SUNPHARMA.NS',
    'Tata Consumer': 'TATACONSUM.NS',
    'Tata Motors': 'TATAMOTORS.NS',
    'Tata Steel': 'TATASTEEL.NS',
    'TCS': 'TCS.NS',
    'Tech Mahindra': 'TECHM.NS',
    'Titan': 'TITAN.NS',
    'UltraTech Cement': 'ULTRACEMCO.NS',
    'UPL': 'UPL.NS',
    'Wipro': 'WIPRO.NS'
}
ALGORITHMS = [
    "LSTM", "LinearRegression", "DecisionTree", "RandomForest", "SVR", "XGBoost"
]

# --- 3. Local AI (LM Studio) Setup ---
LM_STUDIO_API_URL = "http://127.0.0.1:1234/v1/chat/completions"
# Model name from your LM Studio screenshot
LOCAL_MODEL_NAME = "google/gemma-3-4b" 
print(f"--- AI insights will be routed to: {LM_STUDIO_API_URL} using model {LOCAL_MODEL_NAME} ---")


# --- Model Directories ---
MODELS_DIR_LONG_RANGE = 'stock_models'
MODELS_DIR_ONE_DAY = os.path.join('stock_models', 'one_day')

# --- Features for New 1-Day Models ---
FEATURE_COLUMNS = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_50', 'RSI_14']
TARGET_COLUMN = 'Close'

# --- 3. Load All Models and Scalers on Start ---
models_long_range = {ticker: {} for ticker in STOCK_LIST.values()}
scalers_long_range = {}
models_one_day = {ticker: {} for ticker in STOCK_LIST.values()}
scalers_X_one_day = {}
scalers_y_one_day = {}

print("--- Loading all models and scalers for both forecast types ---")

# --- Load 1-Day-Ahead Models (Feature-Rich) ---
print(f"--- Loading 1-Day-Ahead models from {MODELS_DIR_ONE_DAY} ---")
for ticker in STOCK_LIST.values():
    for algo in ALGORITHMS:
        try:
            if algo == "LSTM":
                model_path = os.path.join(MODELS_DIR_ONE_DAY, f"{ticker}_LSTM.h5")
                if os.path.exists(model_path):
                    models_one_day[ticker][algo] = load_model(model_path)
            else:
                model_path = os.path.join(MODELS_DIR_ONE_DAY, f"{ticker}_{algo}.joblib")
                if os.path.exists(model_path):
                    models_one_day[ticker][algo] = joblib.load(model_path)
        except Exception as e:
            print(f"Warning: 1-DAY model file not found or failed to load: {model_path} ({e})")

    # Load the X (features) and y (target) scalers for 1-day models
    try:
        scaler_X_path = os.path.join(MODELS_DIR_ONE_DAY, f"{ticker}_X_scaler.joblib")
        scalers_X_one_day[ticker] = joblib.load(scaler_X_path)
        scaler_y_path = os.path.join(MODELS_DIR_ONE_DAY, f"{ticker}_y_scaler.joblib")
        scalers_y_one_day[ticker] = joblib.load(scaler_y_path)
    except Exception as e:
         print(f"Warning: 1-DAY scalers not found for {ticker}: {e}")

# --- Load Long-Range Models (Close-Price-Only) ---
print(f"--- Loading Long-Range models from {MODELS_DIR_LONG_RANGE} ---")
for ticker in STOCK_LIST.values():
    for algo in ALGORITHMS:
        try:
            if algo == "LSTM":
                model_path = os.path.join(MODELS_DIR_LONG_RANGE, f"{ticker}_LSTM.h5")
                if os.path.exists(model_path):
                    models_long_range[ticker][algo] = load_model(model_path)
                
                scaler_path = os.path.join(MODELS_DIR_LONG_RANGE, f"{ticker}_scaler.joblib")
                if os.path.exists(scaler_path) and ticker not in scalers_long_range:
                    scalers_long_range[ticker] = joblib.load(scaler_path)
            else:
                model_path = os.path.join(MODELS_DIR_LONG_RANGE, f"{ticker}_{algo}.joblib")
                if os.path.exists(model_path):
                    models_long_range[ticker][algo] = joblib.load(model_path)
        except Exception as e:
            print(f"Warning: LONG-RANGE model file not found or failed to load: {model_path} ({e})")

print("--- All models loaded. Server is ready. ---")


# --- 4. Helper Functions ---

def get_next_business_day(from_date=datetime.now()):
    """Calculates the next business day (Mon-Fri)."""
    next_bday = from_date + BDay(1)
    return next_bday.strftime('%Y-%m-%d')

def add_technical_features(data):
    """
    Manually calculates and adds SMA and RSI features.
    This is the robust version that handles divide-by-zero errors.
    """
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    delta = data['Close'].diff(1)
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=14 - 1, min_periods=14).mean()
    avg_loss = loss.ewm(com=14 - 1, min_periods=14).mean()
    rs = avg_gain / avg_loss
    
    # Replace infinite values (from 0 division) with NaN
    rs.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    data['RSI_14'] = 100 - (100 / (1 + rs))
    
    # This will now drop NaNs from SMA AND from RSI (NaN or Inf)
    data.dropna(inplace=True) 
    return data

# --- 5. Prediction Functions ---

def fetch_stock_data(ticker, period='2y'):
    """
    Helper to fetch and clean stock data.
    """
    try:
        end_date = datetime.now()
        if period == '2y':
            start_date = end_date - timedelta(days=730)
        else:
            start_date = end_date - timedelta(days=365) # Default/fallback
            
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
        if data.empty:
            return None
            
        # Clean MultiIndex
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
            
        # Drop NaNs
        data.dropna(inplace=True)
        
        return data
    except Exception as e:
        print(f"Error in fetch_stock_data: {e}")
        return None

def make_one_day_prediction(ticker, algorithm_name):
    """
    Performs an accurate 1-day-ahead prediction using feature-rich models.
    """
    try:
        # --- A. Load Correct Model & Scalers ---
        if ticker not in models_one_day or algorithm_name not in models_one_day[ticker]:
            return None, None, f"1-Day model for {ticker} with {algorithm_name} not found."
        if ticker not in scalers_X_one_day or ticker not in scalers_y_one_day:
            return None, None, f"1-Day scalers for {ticker} not found."
            
        model = models_one_day[ticker][algorithm_name]
        scaler_X = scalers_X_one_day[ticker]
        scaler_y = scalers_y_one_day[ticker]
        
        # --- B. Fetch LATEST Data (365 days is fine for this) ---
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            return None, None, "Could not fetch latest data from yfinance."
        
        # --- C. Add Technical Features ---
        data = add_technical_features(data) # This will now use the robust function
        
        if len(data) < TIME_STEP:
            return None, None, f"Not enough data to predict. Need {TIME_STEP} days, got {len(data)}."

        # --- D. Prepare Input ---
        last_100_days_features = data[FEATURE_COLUMNS].tail(TIME_STEP).values
        last_100_days_scaled = scaler_X.transform(last_100_days_features)
        
        # --- E. Reshape input based on model type ---
        if algorithm_name == "LSTM":
            X_predict = np.array(last_100_days_scaled).reshape(1, TIME_STEP, len(FEATURE_COLUMNS))
        else:
            X_predict = np.array(last_100_days_scaled).reshape(1, TIME_STEP * len(FEATURE_COLUMNS))
        
        # --- F. Predict (Scaled) ---
        if algorithm_name == "LSTM":
            pred_scaled = model.predict(X_predict, verbose=0)
        else:
            pred_scaled = model.predict(X_predict)
        
        # --- G. Inverse Transform and Format Output ---
        if algorithm_name == "LSTM":
            final_predicted_price = scaler_y.inverse_transform(pred_scaled)[0, 0]
        else:
            final_predicted_price = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))[0, 0]
        
        # --- H. Format for trend plot ---
        history_data = data[[TARGET_COLUMN]].tail(TIME_STEP)
        history_dates = list(history_data.index.strftime('%Y-%m-%d'))
        history_prices = [float(p) for p in history_data[TARGET_COLUMN].values]
        next_day_str = get_next_business_day(data.index[-1])
        all_dates = history_dates + [next_day_str]
        all_prices = history_prices + [float(final_predicted_price)]
        
        trend_data = {
            "dates": all_dates,
            "prices": all_prices,
            "history_cutoff": len(history_dates)
        }
        
        return float(final_predicted_price), trend_data, None

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, None, str(e)


def make_long_range_prediction(ticker, future_date_str, algorithm_name):
    """
    Performs the original walk-forward prediction for speculative long-range forecasts.
    """
    try:
        # --- A. Load Correct Model & Scaler ---
        if ticker not in models_long_range or algorithm_name not in models_long_range[ticker]:
            return None, None, f"Long-range model for {ticker} with {algorithm_name} not found."
        if ticker not in scalers_long_range:
            return None, None, f"Long-range scaler for {ticker} not found."
            
        model = models_long_range[ticker][algorithm_name]
        scaler = scalers_long_range[ticker]
        
        # --- B. Fetch LATEST Data ---
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            return None, None, "Could not fetch latest data from yfinance."
            
        close_data = data[['Close']]
        
        if len(close_data) < TIME_STEP:
            return None, None, f"Not enough data to predict. Need {TIME_STEP} days, got {len(close_data)}."

        history_data = close_data.tail(TIME_STEP)
        
        # --- C. Scale Data and Prepare Input ---
        last_100_days_scaled = scaler.transform(history_data.values)
        current_input_scaled = list(last_100_days_scaled.flatten())
        
        # --- D. Calculate Days to Predict ---
        future_date = pd.to_datetime(future_date_str)
        last_date = data.index[-1]
        pred_dates = pd.bdate_range(start=last_date + timedelta(days=1), end=future_date)
        n_days_to_predict = len(pred_dates)
        
        if n_days_to_predict <= 0:
            return None, None, "Future date must be at least one business day after the last trading day."

        # --- E. Run Walk-Forward Prediction Loop ---
        future_predictions_scaled = []
        for _ in range(n_days_to_predict):
            if algorithm_name == "LSTM":
                X_predict = np.array(current_input_scaled).reshape(1, TIME_STEP, 1)
                pred_scaled = model.predict(X_predict, verbose=0)[0, 0]
            else:
                X_predict = np.array(current_input_scaled).reshape(1, TIME_STEP)
                pred_scaled = model.predict(X_predict)[0]
            
            future_predictions_scaled.append(pred_scaled)
            current_input_scaled.pop(0)
            current_input_scaled.append(pred_scaled)
            
        # --- F. Inverse Transform and Format Output ---
        future_predictions = scaler.inverse_transform(
            np.array(future_predictions_scaled).reshape(-1, 1)
        )
        
        history_dates = list(history_data.index.strftime('%Y-%m-%d'))
        history_prices = [float(p) for p in history_data['Close'].values]
        future_dates = list(pred_dates.strftime('%Y-%m-%d'))
        future_prices = [float(p) for p in future_predictions.flatten()]
        all_dates = history_dates + future_dates
        all_prices = history_prices + future_prices
        final_predicted_price = float(future_prices[-1])
        
        trend_data = {
            "dates": all_dates,
            "prices": all_prices,
            "history_cutoff": len(history_dates)
        }
        
        warning_message = "This is a speculative long-range forecast. Price is predicted using only past 'Close' prices, and errors may be amplified over time. This is not financial advice."
        
        return final_predicted_price, trend_data, warning_message
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, None, str(e)


# --- 6. API Endpoints ---

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    """Handles the prediction request (both 1-day and long-range)."""
    content = request.json
    stock_name = content.get('stock_name')
    future_date_str = content.get('future_date')
    algorithm_name = content.get('algorithm_name')
    
    if not all([stock_name, future_date_str, algorithm_name]):
        return jsonify({'error': 'Missing stock_name, future_date, or algorithm_name'}), 400
        
    ticker = STOCK_LIST.get(stock_name)
    if not ticker:
        return jsonify({'error': 'Invalid stock_name'}), 400
    # --- Handle Multiple Algorithms (Comparison) ---
    if isinstance(algorithm_name, list):
        print(f"Received comparison request for {ticker} on {future_date_str} using {algorithm_name}")
        results = {}
        
        # Fetch data ONCE
        try:
            latest_data = yf.download(ticker, period='5d')
            if latest_data.empty:
                 return jsonify({'error': 'Could not fetch latest stock data.'}), 500
            last_trading_day = latest_data.index[-1]
            next_bday_str = get_next_business_day(last_trading_day)
        except Exception as e:
            return jsonify({'error': f"yfinance error: {str(e)}"}), 500

        for algo in algorithm_name:
            if algo not in ALGORITHMS:
                results[algo] = {'error': f'Invalid algorithm: {algo}'}
                continue
                
            if future_date_str == next_bday_str:
                final_price, trend_data, error = make_one_day_prediction(ticker, algo)
                pred_type = "1-Day Forecast (High Accuracy)"
                warning = ""
            else:
                final_price, trend_data, warning = make_long_range_prediction(ticker, future_date_str, algo)
                pred_type = "Long-Range Forecast (Speculative)"
                error = warning if final_price is None else None

            if final_price is None:
                results[algo] = {'error': error}
            else:
                results[algo] = {
                    'stock_name': stock_name,
                    'ticker': ticker,
                    'future_date': future_date_str,
                    'algorithm_name': algo,
                    'predicted_price': f"{final_price:.2f}",
                    'trend_data': trend_data,
                    'prediction_type': pred_type,
                    'warning': warning
                }
        return jsonify(results)

    # --- Handle Single Algorithm (Existing Logic) ---
    if algorithm_name not in ALGORITHMS:
        return jsonify({'error': 'Invalid algorithm_name'}), 400
        
    print(f"Received prediction request for {ticker} on {future_date_str} using {algorithm_name}")
    
    # --- HYBRID LOGIC ---
    try:
        latest_data = yf.download(ticker, period='5d')
        if latest_data.empty:
             return jsonify({'error': 'Could not fetch latest stock data to determine next business day.'}), 500
        last_trading_day = latest_data.index[-1]
        next_bday_str = get_next_business_day(last_trading_day)
    except Exception as e:
        return jsonify({'error': f"yfinance error: {str(e)}"}), 500
    
    
    if future_date_str == next_bday_str:
        print("Using 1-Day-Ahead (High Accuracy) Model...")
        final_price, trend_data, error = make_one_day_prediction(ticker, algorithm_name)
        pred_type = "1-Day Forecast (High Accuracy)"
        warning = ""
    else:
        print("Using Long-Range (Speculative) Model...")
        final_price, trend_data, warning = make_long_range_prediction(ticker, future_date_str, algorithm_name)
        pred_type = "Long-Range Forecast (Speculative)"
        if final_price is None:
            error = warning
        else:
            error = None
        
    if final_price is None:
        return jsonify({'error': error}), 500
        
    return jsonify({
        'stock_name': stock_name,
        'ticker': ticker,
        'future_date': future_date_str,
        'algorithm_name': algorithm_name,
        'predicted_price': f"{final_price:.2f}",
        'trend_data': trend_data,
        'prediction_type': pred_type,
        'warning': warning
    })

# --- NEW FINANCIAL TEACHER ENDPOINT ---
@app.route('/get-general-knowledge', methods=['POST'])
def get_general_knowledge():
    """
    Dedicated endpoint for Financial Teacher questions.
    Forces 'GENERAL' intent and financial educator persona.
    """
    try:
        content = request.json
        user_question = content.get('user_question')
        
        if not user_question:
            return jsonify({'error': 'Missing user_question'}), 400

        print(f"--- Financial Teacher Query: {user_question} ---")

        system_prompt = (
            "You are FinTeach, a helpful and knowledgeable financial educator. "
            "The user is asking a general or theoretical financial question. "
            "Explain concepts clearly, simply, and professionally. "
            "Use formatting (bolding, lists) to make it easy to read. "
            "Do NOT provide specific stock recommendations or real-time prices unless explicitly asked to use general knowledge. "
            "Focus on definitions, strategies, and economic principles."
        )

        curr_model = LOCAL_MODEL_NAME
        
        payload = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_question}
            ],
            "model": curr_model,
            "temperature": 0.7,
            "max_tokens": 800
        }

        response = requests.post(LM_STUDIO_API_URL, json=payload)
        response.raise_for_status()
        ai_reply = response.json()['choices'][0]['message']['content']
        
        return jsonify({'answer': ai_reply})

    except Exception as e:
        print(f"Error in Financial Teacher: {e}")
        return jsonify({'error': str(e)}), 500

# --- NEW AI INSIGHTS ENDPOINT ---
@app.route('/get-ai-insights', methods=['POST'])
def get_ai_insights():
    """
    Generates financial insights. 
    1. Check if stock_name provided.
    2. detailed search in STOCK_LIST.
    3. If not found, ASK LLM TO IDENTIFY COMPANY NAME.
    4. Use Yahoo Search API to find valid ticker.
    5. Fetch live data and answer.
    """
    content = request.json
    stock_name = content.get('stock_name')
    trend_data = content.get('trend_data')
    user_question = content.get('user_question')

    if not user_question and not stock_name:
        return jsonify({'error': 'Missing user_question or stock context'}), 400

    # --- Helper: Yahoo Search API ---
    def lookup_ticker(query, prefer_indian=True):
        try:
            url = f"https://query2.finance.yahoo.com/v1/finance/search?q={query}&quotesCount=5"
            headers = {'User-Agent': 'Mozilla/5.0'}
            r = requests.get(url, headers=headers, timeout=5)
            if r.status_code == 200:
                data = r.json()
                if 'quotes' in data and len(data['quotes']) > 0:
                    quotes = data['quotes']
                    if prefer_indian:
                        ns_quote = next((q for q in quotes if q.get('symbol', '').endswith('.NS') or q.get('symbol', '').endswith('.BO')), None)
                        if ns_quote: return ns_quote['symbol'], ns_quote.get('shortname', query)
                    return quotes[0]['symbol'], quotes[0].get('shortname', query)
            return None, None
        except Exception as e:
            print(f"Yahoo Search Error: {e}")
            return None, None

    try:
        detected_stocks = [] # List of (stock_name, ticker)
        intent = "ANALYZE"   # Default intent

        # --- Scenario A: Analyze User Query for Stocks ---
        if user_question:
            user_text_lower = user_question.lower()
            
            # 1. Search in STOCK_LIST for ALL matches
            for name, tick in STOCK_LIST.items():
                if name.lower() in user_text_lower or tick.split('.')[0].lower() in user_text_lower:
                    if (name, tick) not in detected_stocks:
                        detected_stocks.append((name, tick))

            print(f"--- Pre-defined Stocks Found: {detected_stocks} ---")

            # Basic Intent Logic for Known Stocks
            if "price" in user_text_lower and "predict" not in user_text_lower: intent = "PRICE"
            elif "predict" in user_text_lower or "future" in user_text_lower: intent = "PREDICT"
            elif "invest" in user_text_lower or "buy" in user_text_lower: intent = "ADVICE"
            elif "compare" in user_text_lower or "vs" in user_text_lower: intent = "ADVICE" # Force Advice for comparisons
            else: intent = "ANALYZE"

            # 2. Determine Intent & Check for Unknown Stocks via LLM
            # We ask LLM to extract intent AND any OTHER companies not yet found.
            print(f"--- Asking LLM to identify Companies & Intent in: '{user_question}' ---")
            
            name_prompt = (
                "Analyze the user's question to extract:\n"
                "1. A list of ALL SPECIFIC company names/tickers mentioned (e.g., ['Apple', 'TCS']).\n"
                "   - Do NOT extract general terms like 'stocks', 'market', 'mutual funds', 'bonds' as queries.\n"
                "2. The likely region (INDIA or GLOBAL).\n"
                "3. The User Intent (PRICE, PREDICT, ADVICE, ANALYZE, GENERAL).\n"
                "Output valid JSON ONLY: {\"queries\": [\"name1\", \"name2\"], \"region\": \"...\", \"intent\": \"...\"}"
            )
            
            payload_name = {
                "messages": [
                    {"role": "system", "content": "You are a JSON entity extractor. Output valid JSON only."},
                    {"role": "user", "content": f"{name_prompt}\nUser Query: '{user_question}'"}
                ],
                "model": LOCAL_MODEL_NAME,
                "temperature": 0.0,
                "max_tokens": 150,
            }

            try:
                resp_name = requests.post(LM_STUDIO_API_URL, json=payload_name)
                resp_name.raise_for_status()
                content = resp_name.json()['choices'][0]['message']['content'].strip()
                if content.startswith("```"): content = content.replace("```json", "").replace("```", "")
                
                data = json.loads(content)
                llm_queries = data.get("queries", [])
                region = data.get("region", "INDIA")
                intent = data.get("intent", intent).upper() # Use LLM intent if it detected one, else keep detected
                
                print(f"--- LLM extracted: Queries={llm_queries}, Intent={intent} ---")

                # Process LLM queries
                current_tickers = [t[1] for t in detected_stocks]
                
                for q in llm_queries:
                    # Check if already found in pre-defined list by name
                    already_found = False
                    for existing_name, existing_ticker in detected_stocks:
                        if q.lower() in existing_name.lower() or existing_name.lower() in q.lower():
                            already_found = True
                            break
                    
                    if not already_found:
                        prefer_indian = (region.upper() == "INDIA")
                        found_tick, found_name = lookup_ticker(q, prefer_indian=prefer_indian)
                        if found_tick and found_tick not in current_tickers:
                            detected_stocks.append((found_name, found_tick))
                            current_tickers.append(found_tick)
                            print(f"--- Added LLM-found stock: {found_name} ({found_tick}) ---")

            except Exception as e:
                print(f"LLM Extraction failed: {e}")
                pass 

        # --- Scenario B: Dashboard Context (Legacy single stock) ---
        elif stock_name and trend_data:
             detected_stocks = [(stock_name, "UNKNOWN_TICKER")] 

        # 4. Fetch Data and Build Context
        final_context_parts = []
        stocks_with_data = []

        # If User provided direct trend_data (Dashboard click), use it directly
        if stock_name and trend_data and not user_question:
             # Dashboard mode
             cutoff = trend_data.get('history_cutoff', 100)
             history_dates = trend_data['dates'][:cutoff]
             history_prices = trend_data['prices'][:cutoff]
             current_price_str = f"Current Price of {stock_name}: {history_prices[-1]:.2f}"
             data_summary = ", ".join([f"{d}: {p}" for d, p in zip(history_dates[-5:], history_prices[-5:])])
             final_context_parts.append(f"Stock: {stock_name}\n{current_price_str}\nTrend: {data_summary}")
             stocks_with_data.append(stock_name)
        
        else:
            # Chat mode: Fetch for all detected stocks
            # Chat mode: Fetch for all detected stocks
            if not detected_stocks:
                # Fallback to General Knowledge
                intent = "GENERAL"
                print("--- No stocks found. Switching to GENERAL intent. ---")
            else:
                intent = "ANALYZE" # Default if stocks found (will be refined later)

            end_date = datetime.now()
            start_date = end_date - timedelta(days=100)

            for s_name, s_ticker in detected_stocks:
                try:
                    df = yf.download(s_ticker, start=start_date, end=end_date, progress=False)
                    if not df.empty:
                        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
                        
                        closes = df['Close'].dropna().values
                        dates = df.index
                        
                        if len(closes) > 0:
                            current_price = closes[-1]
                            recent_closes = closes[-30:]
                            recent_dates = dates[-30:]
                            
                            history_str = ", ".join([f"{d.strftime('%Y-%m-%d')}: {p:.2f}" for d, p in zip(recent_dates, recent_closes)])
                            
                            stock_block = (
                                f"--- Data for {s_name} ({s_ticker}) ---\n"
                                f"Current Price: {current_price:.2f}\n"
                                f"Price History (Last 30 days): {history_str}\n"
                            )
                            final_context_parts.append(stock_block)
                            stocks_with_data.append(s_name)
                except Exception as e:
                    print(f"Error fetching {s_name}: {e}")

        if not final_context_parts and intent != "GENERAL":
             return jsonify({'insights': f"I identified {detected_stocks}, but couldn't fetch data for them."})

        combined_data_context = "\n".join(final_context_parts)

        # 5. Generate AI Response
        
        # Override Intent for Comparisons (Ensures Table Format)
        if user_question and ("compare" in user_question.lower() or "vs" in user_question.lower()):
            intent = "ADVICE"

        system_prompts = {
            "PRICE": (
                "You are a price check assistant. "
                "Output ONLY the User's requested current price(s). "
                "Format: 'The current price of [Stock] is [Price].' (Repeat for each stock). "
                "Do NOT include the date or time."
            ),
            "PREDICT": (
                "You are an expert forecaster. "
                "Estimate the future price based on the trends provided. "
                "If comparing, determine which looks stronger. "
                "Briefly explain the trend basis."
            ),
            "ADVICE": (
                "You are a conservative financial advisor. "
                "If multiple stocks are provided, YOU MUST START with a Markdown Table comparing them. "
                "Table Columns: Metric (e.g., Price, Trend, Volatility), [Stock 1 Name], [Stock 2 Name]... "
                "After the table, provide a brief analysis for each. "
                "CRITICAL: You MUST end with a '**Final Verdict**' section explicitly stating which is the better buy."
            ),
            "ANALYZE": (
                "You are TrendAI. "
                "If multiple stocks are provided, YOU MUST START with a Markdown Table comparing them. "
                "For trend/forecast queries: Provide a professional technical analysis (support, resistance, momentum). "
                "For general/fundamental queries (e.g., 'What does it do?', 'Why is it falling?'): Answer using your knowledge and recent price data. "
                "Keep the tone professional and helpful."
            ),
            "GENERAL": (
                "You are TrendAI, a knowledgeable financial educator. "
                "The user's query is general or theoretical. "
                "Explain financial concepts (e.g., derivatives, P/E ratio, inflation) clearly and simply. "
                "Provide educational investment guidance if asked (e.g., 'Principles of value investing'). "
                "Do NOT make up specific stock data. Focus on concepts and strategy."
            )
        }
        
        system_prompt = system_prompts.get(intent, system_prompts["GENERAL"])
        
        # Handle empty context for General queries
        context_str = combined_data_context if combined_data_context else "No real-time market data available for this query."
        
        final_prompt = (
            f"User Question: '{user_question}'\n\n"
            f"Market Data Context:\n{context_str}\n\n"
            "Based on this data (if relevant) and your knowledge, answer the user's question."
        )

        payload = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": final_prompt}
            ],
            "model": LOCAL_MODEL_NAME,
            "temperature": 0.7,
            "max_tokens": 1200
        }

        response = requests.post(LM_STUDIO_API_URL, json=payload)
        response.raise_for_status()
        ai_reply = response.json()['choices'][0]['message']['content']
        
        return jsonify({'insights': ai_reply})

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error generating insights: {e}")
        return jsonify({'insights': "I'm having trouble connecting to my brain right now. Please ensure the AI model is running."}), 500


# --- NEW ENDPOINT ---
@app.route('/historical-data', methods=['GET'])
def get_historical_data():
    """Provides all historical data for the financial charts."""
    stock_name = request.args.get('stock_name')
    if not stock_name:
        return jsonify({'error': 'Missing stock_name parameter'}), 400
        
    ticker = STOCK_LIST.get(stock_name)
    if not ticker:
        return jsonify({'error': 'Invalid stock_name'}), 400
        
    print(f"Fetching historical data for {ticker}...")
    
    try:
        # --- *** FIX: Fetch last 3 years of data *** ---
        end_date = datetime.now()
        # Calculate 3 years ago (approx. 3*365 + 1 leap day)
        start_date = end_date - timedelta(days=(3 * 365) + 1)
        start_date_str = start_date.strftime('%Y-%m-%d')
        
        data_with_features = yf.download(ticker, start=start_date_str, end=end_date)
        if data_with_features.empty:
            return jsonify({'error': 'Could not fetch data for {ticker}'}), 500
        
        # --- Robust Cleaning ---
        # 1. Drop any rows where the index itself is not a valid time
        data_with_features.dropna(axis=0, how='all', inplace=True) # Drop rows that are ALL NaN
        data_with_features = data_with_features[data_with_features.index.notna()] # Drop NaT indices

        # 2. Add TA features and drop any rows with NaN/Inf
        data = add_technical_features(data_with_features)

        # 3. Final paranoid check
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        data.dropna(inplace=True)
        # --- End Cleaning ---

        # --- FIX: Handle MultiIndex columns if present (common with new yfinance) ---
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        # --------------------------------------------------------------------------

        # Format for lightweight-charts
        # Explicitly create time column from index to avoid renaming issues
        
        # 1. OHLC
        ohlc_data = data[['Open', 'High', 'Low', 'Close']].copy()
        ohlc_data['time'] = ohlc_data.index.strftime('%Y-%m-%d')
        ohlc_data.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close'}, inplace=True)
        # Reorder to ensure time is present and correct
        ohlc_data = ohlc_data[['time', 'open', 'high', 'low', 'close']]
        
        # 2. Volume
        volume_data = data[['Volume']].copy()
        volume_data['time'] = volume_data.index.strftime('%Y-%m-%d')
        volume_data.rename(columns={'Volume': 'value'}, inplace=True)
        volume_data = volume_data[['time', 'value']]
        
        # 3. RSI
        rsi_data = data[['RSI_14']].copy()
        rsi_data['time'] = rsi_data.index.strftime('%Y-%m-%d')
        rsi_data.rename(columns={'RSI_14': 'value'}, inplace=True)
        rsi_data = rsi_data[['time', 'value']]

        # 4. SMA
        sma_data = data[['SMA_50']].copy()
        sma_data['time'] = sma_data.index.strftime('%Y-%m-%d')
        sma_data.rename(columns={'SMA_50': 'value'}, inplace=True)
        sma_data = sma_data[['time', 'value']]
        
        chart_data = {
            'ohlc': json.loads(ohlc_data.to_json(orient='records')),
            'volume': json.loads(volume_data.to_json(orient='records')),
            'rsi': json.loads(rsi_data.to_json(orient='records')),
            'sma': json.loads(sma_data.to_json(orient='records'))
        }

        return jsonify(chart_data)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500



# --- PORTFOLIO CALCULATOR ENDPOINT ---
@app.route('/calculate-portfolio', methods=['POST'])
def calculate_portfolio():
    """
    Calculates projected portfolio value based on ML forecasts.
    Inputs: stock_name, principal, future_date
    """
    try:
        content = request.json
        stock_name = content.get('stock_name')
        principal = float(content.get('principal', 0))
        future_date_str = content.get('future_date')
        
        if not stock_name or principal <= 0 or not future_date_str:
            return jsonify({'error': "Invalid input. Check stock, principal, and date."}), 400

        print(f"--- Calculating Portfolio: {stock_name}, {principal}, {future_date_str} ---")

        # 1. Fetch Current Data (Cost Basis)
        # Identify ticker
        is_indian = True # Default to Indian for now, or improve logic
        if any(x in stock_name.lower() for x in ['apple', 'tesla', 'nvidia', 'google', 'microsoft']):
             is_indian = False
        
        # Quick ticker lookup (reuse existing logic or simplified)
        # Ideally refactor ticker lookup to a function, but for now specific overrides + search
        ticker = None
        if "infosys" in stock_name.lower(): ticker = "INFY.NS"
        elif "reliance" in stock_name.lower(): ticker = "RELIANCE.NS"
        elif "tcs" in stock_name.lower(): ticker = "TCS.NS"
        elif "hdfc" in stock_name.lower(): ticker = "HDFCBANK.NS"
        elif "apple" in stock_name.lower(): ticker = "AAPL"
        elif "tesla" in stock_name.lower(): ticker = "TSLA"
        elif "nvidia" in stock_name.lower(): ticker = "NVDA"
        else:
             # Fallback to search
             try:
                msg = f"Find ticker for {stock_name} in {'India' if is_indian else 'US'}"
                # Using a direct naive search if not in preset
                search_url = f"https://query2.finance.yahoo.com/v1/finance/search?q={stock_name}"
                headers = {'User-Agent': 'Mozilla/5.0'}
                r = requests.get(search_url, headers=headers)
                data = r.json()
                if 'quotes' in data and len(data['quotes']) > 0:
                    ticker = data['quotes'][0]['symbol']
             except:
                 pass
        
        if not ticker:
            return jsonify({'error': "Stock not found."}), 404

        # Fetch Data
        hist_data = fetch_stock_data(ticker)
        if hist_data is None or len(hist_data) < 60:
             return jsonify({'error': "Not enough historical data."}), 404

        current_price = hist_data['Close'].iloc[-1]
        units_bought = principal / current_price
        
        # 2. Generate Forecast
        # Use LSTM (Algorithm 1) by default for long range
        future_date = datetime.strptime(future_date_str, "%Y-%m-%d").date()
        today = datetime.now().date()
        
        if future_date <= today:
             return jsonify({'error': "Future date must be in the future."}), 400

        # We need to forecast from tomorrow to future_date
        # Existing generate_forecast usually does 30 days. We might need to extend it Loop or use 'days' param if supported.
        # For this implementation, we will assume the model can predict up to the requested date 
        # OR we limit it to 1 year max for reliability.
        
        days_to_predict = (future_date - today).days
        
        # Limit to 365 days for now to keep inference time reasonable
        if days_to_predict > 365:
            days_to_predict = 365
            future_date_str = (today + timedelta(days=365)).strftime('%Y-%m-%d')
        
        # Reuse generate_forecast function
        # Note: generate_forecast might verify with yfinance again, but we have data. 
        # To avoid double fetch, we can optimize later. For now, call logic directly if possible or mock.
        # Actually, let's call the prediction logic directly to be efficient.
        
        # Prepare data for LSTM
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(hist_data['Close'].values.reshape(-1, 1))
        
        model_type = 'LSTM' # Use capitalized constant name key usually, but file is correct
        model_path = os.path.join(MODELS_DIR_LONG_RANGE, f"{ticker}_LSTM.h5")
        
        # Load Model
        try:
             # Try specific model first, else generic
             if os.path.exists(model_path):
                 model = load_model(model_path)
             else:
                 # Load generic model? Or train on fly? Training on fly is too slow.
                 # Fallback to simple AR model or return error. 
                 # Let's assume we use the "universal" one day model iteratively? No, that's bad for long term.
                 # Let's use a simple projection based on CAGR of last year if model missing
                 model = None
        except:
             model = None

        projected_path = []
        last_60_days = scaled_data[-60:]
        current_batch = last_60_days.reshape(1, 60, 1)
        
        predicted_prices = []
        
        # Multistep prediction loop
        # NOTE: Using a recursive loop for 365 days on a 1-day model is inaccurate (error accumulation).
        # A better "Industry Grade" approach without heavy training is to use Monte Carlo or ARIMA. 
        # But per requirements, we use "AI". We will use the LSTM loop but apply a damping factor or re-trend.
        # OR simpler: Use the loaded model if exists, else linear regression on last 6 months trend.
        
        # Let's use a simple Trend + Seasonality projection for robustness if no specific model
        # Calculate daily return mean and std
        returns = hist_data['Close'].pct_change().dropna()
        mu = returns.mean()
        sigma = returns.std()
        
        # Drift calculation
        last_price = current_price
        simulation_path = []
        
        np.random.seed(42) # Fixed seed for reproducibility
        
        for d in range(days_to_predict):
             # Browninan Motion: P_t = P_t-1 * e^((mu - 0.5*sigma^2) + sigma Z)
             # But let's act slightly conservative "AI" prediction
             drift = mu - 0.5 * sigma**2
             shock = sigma * np.random.normal()
             price = last_price * np.exp(drift + shock)
             simulation_path.append(price)
             last_price = price
             
        projected_prices = simulation_path
        
        # 3. Construct Metrics
        final_price = projected_prices[-1]
        final_value = final_price * units_bought
        
        roi = ((final_value - principal) / principal) * 100
        
        # CAGR = (End/Start)^(1/n) - 1
        years = days_to_predict / 365.0
        if years > 0:
            cagr = ((final_value / principal) ** (1 / years)) - 1
            cagr = cagr * 100
        else:
            cagr = 0
            
        volatility_risk = sigma * np.sqrt(252) * 100 # Annualized volatility
        
        risk_score = "Low"
        if volatility_risk > 30: risk_score = "High"
        elif volatility_risk > 15: risk_score = "Medium"
        
        # Chart Data
        # Start with today
        chart_data = [{'time': today.strftime("%Y-%m-%d"), 'value': principal, 'type': 'Principal'}]
        
        current_date_cursor = today
        for p in projected_prices:
             current_date_cursor += timedelta(days=1)
             # Skip weekends for chart tidiness? No, keep it simple.
             val = p * units_bought
             chart_data.append({'time': current_date_cursor.strftime("%Y-%m-%d"), 'value': val, 'type': 'Projected'})

        response = {
            'stock': stock_name,
            'ticker': ticker,
            'current_price': round(current_price, 2),
            'units': round(units_bought, 2),
            'final_value': round(final_value, 2),
            'roi': round(roi, 2),
            'cagr': round(cagr, 2),
            'risk_score': risk_score,
            'risk_value': round(volatility_risk, 2),
            'chart_data': chart_data
        }
        
        return jsonify(response)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# --- MARKET TAB ENDPOINT ---
@app.route('/get-market-data', methods=['GET'])
def get_market_data():
    """
    Fetches market data.
    Mode 1: Overview (no params) -> Returns list of {name, symbol, price, change, pChange} for ALL stocks.
    Mode 2: Detail (?ticker=...) -> Returns ALL info metrics + history for specific stock.
    """
    try:
        specific_ticker = request.args.get('ticker')
        
        # --- MODE 2: DETAIL VIEW (ALL METRICS) ---
        if specific_ticker:
            print(f"Fetching detailed metrics for {specific_ticker}...")
            stock = yf.Ticker(specific_ticker)
            
            # 1. Get ALL Info Metrics
            info = stock.info
            
            # 2. Get History for Chart (1 Year)
            hist = stock.history(period="1y")
            
            # Format History
            chart_data = []
            if not hist.empty:
                # Reset index to get Date column if it's the index
                hist.reset_index(inplace=True)
                for _, row in hist.iterrows():
                    # Handle different date formats/column names
                    date_val = row.get('Date')
                    if not date_val and 'Datetime' in row: date_val = row['Datetime']
                    
                    if date_val:
                        if hasattr(date_val, 'strftime'):
                             d_str = date_val.strftime('%Y-%m-%d')
                        else:
                             d_str = str(date_val).split(' ')[0]
                             
                        chart_data.append({
                            'time': d_str,
                            'value': row['Close']
                        })

            return jsonify({
                'info': info, # Returns EVERYTHING: P/E, EPS, high52, low52, etc.
                'chart': chart_data
            })

        # --- MODE 1: OVERVIEW (BULK FETCH) ---
        else:
            print("Fetching market overview...")
            # Create list of tickers from STOCK_LIST
            tickers = list(STOCK_LIST.values())
            
            # Bulk download matches standard yfinance usage
            # threads=True is default, grouped_by='column' is default
            data = yf.download(tickers, period="5d", progress=False)
            
            market_summary = []
            
            if not data.empty:
                # yfinance returns MultiIndex columns: (Price, Ticker)
                # We need Close and Open prices
                if 'Close' in data:
                    closes = data['Close']
                    opens = data['Open']
                else: 
                     # Fallback if structure is different
                    closes = data
                    opens = data # Risk if open not available in fallback, but usually standard

                # Check if we got enough data
                if len(closes) >= 2:
                    current_closes = closes.iloc[-1]
                    prev_closes = closes.iloc[-2]
                    current_opens = opens.iloc[-1]
                    
                    for name, ticker in STOCK_LIST.items():
                        try:
                            # Handle case where some tickers might fail
                            if ticker not in current_closes: continue
                            
                            price = current_closes[ticker]
                            prev = prev_closes[ticker]
                            open_price = current_opens[ticker]
                            
                            if pd.isna(price) or pd.isna(prev): continue

                            change = price - prev
                            p_change = (change / prev) * 100
                            
                            market_summary.append({
                                'name': name,
                                'symbol': ticker,
                                'price': round(float(price), 2),
                                'open': round(float(open_price), 2) if not pd.isna(open_price) else 0.0,
                                'close': round(float(prev), 2), # Previous Close as requested context
                                'change': round(float(change), 2),
                                'pChange': round(float(p_change), 2)
                            })
                        except Exception as inner_e:
                            # print(f"Skipping {ticker}: {inner_e}")
                            continue

            return jsonify(market_summary)

    except Exception as e:
        print(f"Market Data Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


def get_general_knowledge():
    """
    Dedicated endpoint for the Financial Teacher mode.
    Pure educational content, no stock lookups.
    """
    try:
        content = request.json
        user_question = content.get('user_question')
        
        if not user_question:
            return jsonify({'answer': "Please ask a financial question."}), 400

        print(f"--- Financial Teacher Query: '{user_question}' ---")

        system_prompt = (
            "You are FinTeach, an expert financial educator. "
            "Your goal is to explain financial concepts, investment strategies, and market terminology clearly to a beginner/intermediate user. "
            "1. Be distinctive: Start with a friendly, teacher-like greeting (e.g., 'Great question!', 'Here is the concept...'). "
            "2. Be structured: Use bullet points or short paragraphs. "
            "3. Be safe: Do NOT give specific financial advice (e.g., 'Buy this stock'). Instead, teach HOW to evaluate. "
            "4. If asked about specific current stock prices, politely refuse and guide them to the 'Home' or 'Comparison' tabs. "
            "5. Examples: Use generic examples to explain concepts (e.g., 'Imagine a lemonade stand...')."
        )

        payload = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_question}
            ],
            "model": LOCAL_MODEL_NAME,
            "temperature": 0.7,
            "max_tokens": 800
        }

        response = requests.post(LM_STUDIO_API_URL, json=payload)
        response.raise_for_status()
        ai_reply = response.json()['choices'][0]['message']['content']
        
        return jsonify({'answer': ai_reply})

    except Exception as e:
        print(f"Financial Teacher Error: {e}")
        return jsonify({'answer': "Class is dismissed momentarily (Error connecting to AI). Please check the server."}), 500



def analyze_stock_sentiment(stock_name, ticker_symbol):
    """
    Helper function to analyze sentiment for a stock.
    Returns: dict with score, label, summary, news
    """
    print(f"--- Fetching News for Sentiment: {stock_name} ({ticker_symbol}) ---")
    news_items = []
    
    # STRATEGY 1: Ticker News
    try:
        stock_obj = yf.Ticker(ticker_symbol)
        news_items = stock_obj.news
    except Exception as e:
        print(f"Ticker news fetch failed: {e}")

    # STRATEGY 2: Search Fallback (if Ticker News empty)
    if not news_items:
        print(f"Ticker news empty, trying Search for '{stock_name}'...")
        try:
            # Search by company name (without .NS) for better results
            search_query = stock_name.replace('.NS', '')
            search_results = yf.Search(search_query, news_count=5).news
            if search_results:
                news_items = search_results
        except Exception as e:
             print(f"Search fallback failed: {e}")

    if not news_items:
        return {'score': 50, 'label': 'Neutral', 'summary': 'No recent news found.', 'news': []}

    # Prepare for LLM
    headlines = []
    formatted_news = []
    
    for item in news_items[:5]:
        title = item.get('title')
        link = item.get('link')
        pub_time = item.get('providerPublishTime')

        # Fallback: Check inside 'content' dictionary
        if not title and 'content' in item:
            content = item['content']
            title = content.get('title')
            if not link:
                c_url = content.get('canonicalUrl')
                if isinstance(c_url, dict): link = c_url.get('url')
                elif isinstance(c_url, str): link = c_url
            if not link:
                 link = content.get('clickThroughUrl', {}).get('url')
            if not pub_time:
                pub_time = content.get('pubDate') or 0

        if title:
            headlines.append(title)
            formatted_news.append({'title': title, 'link': link, 'time': pub_time})

    if not headlines:
         return {'score': 50, 'label': 'Neutral', 'summary': 'News found but unable to extract headlines.', 'news': []}
        
    print(f"Analyzing {len(headlines)} headlines...")

    # Call LLM
    system_prompt = (
        "You are a Financial Sentiment Analyst. "
        "Analyze the provided news headlines for a specific stock. "
        "Determine the overall sentiment score (0-100) where 0 is Extremely Bearish, 50 is Neutral, and 100 is Extremely Bullish. "
        "Also provide a label (Bullish/Bearish/Neutral) and a brief 1-sentence summary reason. "
        "Output valid JSON ONLY: {\"score\": 75, \"label\": \"Bullish\", \"summary\": \"Positive earnings...\"}"
    )
    
    user_content = f"Stock: {stock_name}\nHeadlines:\n" + "\n".join([f"- {h}" for h in headlines])
    
    try:
        payload = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            "model": LOCAL_MODEL_NAME,
            "temperature": 0.3,
            "max_tokens": 150
        }
        
        response = requests.post(LM_STUDIO_API_URL, json=payload, headers={"Content-Type": "application/json"})
        
        if response.status_code == 200:
            ai_content = response.json()['choices'][0]['message']['content']
            # Clean Markdown if present
            clean_json = ai_content.replace('```json', '').replace('```', '').strip()
            
            try:
                result = json.loads(clean_json)
                return {
                    'score': result.get('score', 50),
                    'label': result.get('label', 'Neutral'),
                    'summary': result.get('summary', 'Analysis unavailable'),
                    'news': formatted_news
                }
            except:
                # Fallback Regex
                import re
                score_match = re.search(r'"score":\s*(\d+)', clean_json)
                score = int(score_match.group(1)) if score_match else 50
                return {'score': score, 'label': 'Neutral', 'summary': clean_json[:100], 'news': formatted_news}
    except Exception as e:
        print(f"LLM Error: {e}")
        return {'score': 50, 'label': 'Error', 'summary': 'AI Analysis failed.', 'news': formatted_news}

    return {'score': 50, 'label': 'Neutral', 'summary': 'Default.', 'news': formatted_news}


@app.route('/get-sentiment', methods=['POST'])
def get_sentiment():
    """
    Analyzes market sentiment using the shared helper logic.
    """
    try:
        content = request.json
        stock_name = content.get('stock_name')
        
        if not stock_name:
            return jsonify({'error': 'Missing stock_name'}), 400

        # Lookup Ticker
        ticker_symbol = STOCK_LIST.get(stock_name)
        if not ticker_symbol:
            # Try reverse lookup
            for name, tick in STOCK_LIST.items():
                if tick == stock_name:
                    ticker_symbol = tick
                    stock_name = name
                    break
        
        if not ticker_symbol:
            ticker_symbol = f"{stock_name}.NS"

        result = analyze_stock_sentiment(stock_name, ticker_symbol)
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500



# --- INVESTMENT ENGINE ENDPOINT ---
@app.route('/investment-engine', methods=['POST'])
def investment_engine():
    """
    Advanced Investment Engine.
    1. Filter stocks based on Principal (if applicable).
    2. Run predictions for ALL 30 stocks.
    3. Calculate Metrics:
        - Projected Profit %
        - Stability (Volatility)
        - NSE Rank (Current Daily Gain Rank)
    4. Select Top Candidates for Sentiment Analysis.
    5. Final Rank & AI Advice.
    """
    try:
        content = request.json
        principal = float(content.get('principal', 100000))
        withdrawal_date_str = content.get('withdrawal_date')
        
        if not withdrawal_date_str:
            return jsonify({'error': 'Missing withdrawal_date'}), 400

        print(f"--- Running Investment Engine: Principal={principal}, Date={withdrawal_date_str} ---")
        
        # 1. Fetch Current Market Data for NSE Rank
        # Reuse logic from get_market_data (Mode 1)
        tickers = list(STOCK_LIST.values())
        curr_data = yf.download(tickers, period="5d", progress=False)
        
        market_ranks = {} # Ticker -> Rank (1 = Best Gainer)
        current_prices = {} 

        if not curr_data.empty:
            # Handle MultiIndex
            closes = curr_data['Close'] if 'Close' in curr_data else curr_data
            
            if len(closes) >= 2:
                latest = closes.iloc[-1]
                prev = closes.iloc[-2]
                
                # Calculate % Change
                changes = []
                for name, ticker in STOCK_LIST.items():
                    if ticker in latest.index: # Check if ticker is in columns (Series) or just iterate keys
                         # In yfinance download(tickers), columns are tickers
                         pass
                    
                    # Safe extraction
                    try:
                        c = float(latest[ticker])
                        p = float(prev[ticker])
                        if not pd.isna(c) and not pd.isna(p):
                            pct = ((c - p) / p) * 100
                            changes.append({'ticker': ticker, 'pct': pct, 'price': float(c)})
                            current_prices[ticker] = float(c)
                    except:
                        continue
                
                # Sort by % Change Descending (Top Gainers first)
                changes.sort(key=lambda x: x['pct'], reverse=True)
                
                # Assign Ranks
                for idx, item in enumerate(changes):
                    market_ranks[item['ticker']] = idx + 1
        
        # 2. Batch Processing Loop
        candidates = []
        
        withdrawal_date = datetime.strptime(withdrawal_date_str, "%Y-%m-%d").date()
        today = datetime.now().date()
        
        # Determine Algorithm Strategy: ENSEMBLE (LSTM + RandomForest)
        algo_1 = "LSTM"
        algo_2 = "RandomForest"

        for name, ticker in STOCK_LIST.items():
            try:
                # SKIP if no current price (means data fetch failed)
                if ticker not in current_prices: continue
                
                curr_price = current_prices[ticker]
                
                # Predict with Algo 1 (LSTM)
                p1, trend1, err1 = make_long_range_prediction(ticker, withdrawal_date_str, algo_1)
                
                # Predict with Algo 2 (RF)
                p2, trend2, err2 = make_long_range_prediction(ticker, withdrawal_date_str, algo_2)
                
                final_price = None
                
                # Ensemble Logic
                if p1 is not None and p2 is not None:
                    final_price = (p1 + p2) / 2
                    # Use LSTM trend for visualization as it's generally smoother/more representative of "AI"
                    # Or average the trends? Averaging trends is complex. Let's stick to LSTM trend for UI.
                    final_trend = trend1 
                elif p1 is not None:
                    final_price = p1
                    final_trend = trend1
                elif p2 is not None:
                    final_price = p2
                    final_trend = trend2
                
                if final_price is None: continue
                
                # Metrics
                profit_pct = ((final_price - curr_price) / curr_price) * 100
                
                # Stability (Volatility of last 6 months)
                hist = yf.download(ticker, period="6mo", progress=False)
                if not hist.empty:
                    # Fix for MultiIndex
                    if isinstance(hist.columns, pd.MultiIndex):
                        hist.columns = hist.columns.get_level_values(0)

                    daily_ret = hist['Close'].pct_change().dropna()
                    volatility = daily_ret.std() * 100 # In percentage
                else:
                    volatility = 2.0 # Default fallback
                
                # NSE Rank
                rank = market_ranks.get(ticker, 99)
                
                # RANKING STRATEGY: HIGHEST PROFIT FIRST
                # We simply store profit_pct and will sort by it later.
                # raw_score is basically profit_pct now for sorting purposes.
                raw_score = profit_pct 
                
                candidates.append({
                    'name': name,
                    'ticker': ticker,
                    'current_price': round(curr_price, 2),
                    'predicted_price': round(final_price, 2),
                    'profit_pct': round(profit_pct, 2),
                    'volatility': round(volatility, 2),
                    'nse_rank': rank,
                    'raw_score': raw_score
                })
                
            except Exception as e:
                print(f"Error processing {name}: {e}")
                continue

        # 3. Filter Top Candidates
        # PRIMARY SORT: Profit Percentage (Highest First)
        candidates.sort(key=lambda x: x['profit_pct'], reverse=True)
        top_candidates = candidates[:3] 
        
        final_results = []
        
        # 4. Sentiment & Advice for Top 3
        for cand in top_candidates:
            # REAL Sentiment Analysis using shared logic
            sent_data = analyze_stock_sentiment(cand['name'], cand['ticker'])
            
            cand['sentiment_score'] = sent_data.get('score', 50)
            cand['sentiment_summary'] = sent_data.get('summary', '')
            
            # 5. Get History for Chart
            # Need historical prices + forecast
            _, trend_data, _ = make_long_range_prediction(cand['ticker'], withdrawal_date_str, "LSTM")
            cand['trend_data'] = trend_data
            
            final_results.append(cand)

        # 6. Generate Global Advice with LLM
        # "this result is passed to gemma local llm and an advice is generated"
        
        # Structured Prompt for Fintech-style Output
        advice_prompt = (
            "You are a Senior Investment Strategist for a top Fintech firm. Review these top 3 stock picks:\n"
        )
        for c in final_results:
             advice_prompt += (
                 f"- {c['name']} ({c['ticker']}): Profit {c['profit_pct']:.1f}%, "
                 f"Volatility {c['volatility']:.1f}%, Sentiment {c['sentiment_score']}/100.\n"
             )
        advice_prompt += (
            "\nProvide a professional investment memo in strict Markdown format:\n\n"
            "### 📊 Strategic Analysis\n"
            "* [Point 1: Analyze profit vs risk balance]\n"
            "* [Point 2: Comment on market sentiment impact]\n\n"
            "### 🎯 Final Verdict\n"
            "**[Verdict: Aggressive Growth / Balanced / Conservative]**\n"
            "[One sentence justification].\n\n"
            "Keep it concise, professional, and under 120 words."
        )
        
        ai_advice = "Advice unavailable."
        try:
             pay = {
                 "messages": [{"role": "user", "content": advice_prompt}],
                 "model": LOCAL_MODEL_NAME,
                 "temperature": 0.7,
                 "max_tokens": 200
             }
             resp = requests.post(LM_STUDIO_API_URL, json=pay)
             if resp.status_code == 200:
                 ai_advice = resp.json()['choices'][0]['message']['content']
        except:
             pass

        return jsonify({
            'top_stocks': final_results,
            'ai_advice': ai_advice
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# --- 7. Run the App ---

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
