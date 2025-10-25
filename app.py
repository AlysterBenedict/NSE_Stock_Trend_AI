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
    'Infosys': 'INFY.NS',
    'Yes Bank': 'YESBANK.NS',
    'TCS': 'TCS.NS',
    'HDFC Bank': 'HDFCBANK.NS',
    'ITC': 'ITC.NS'
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

# --- NEW AI INSIGHTS ENDPOINT ---
@app.route('/get-ai-insights', methods=['POST'])
def get_ai_insights():
    """
    Generates financial insights for a stock based on its recent trend data.
    """
    content = request.json
    stock_name = content.get('stock_name')
    trend_data = content.get('trend_data') # Expects {'dates': [...], 'prices': [...], 'history_cutoff': ...}

    if not all([stock_name, trend_data]):
        return jsonify({'error': 'Missing stock_name or trend_data'}), 400

    try:
        # Format the trend data for the prompt
        # We only use the historical data (up to history_cutoff)
        cutoff = trend_data.get('history_cutoff', 100) # Default to 100
        history_dates = trend_data['dates'][:cutoff]
        history_prices = trend_data['prices'][:cutoff]
        
        # Create a simple string of the data
        data_string = ", ".join([f"{date}: ₹{price:.2f}" for date, price in zip(history_dates, history_prices)])
        
        start_date = history_dates[0]
        end_date = history_dates[-1]
        start_price = history_prices[0]
        end_price = history_prices[-1]

        system_prompt = (
            "You are a helpful and highly detailed financial analyst. "
            "Your goal is to provide a comprehensive analysis of a stock's recent performance based *only* on the provided data. "
            "Do NOT provide financial advice or make future predictions. "
            "Analyze the past trend only. "
            "**Your response MUST be formatted using subtopics with point-wise content.** "
            "Do NOT use markdown headings (like #, ##). "
            "Use plain text for subtopics (e.g., 'Overall Trend:') followed by bulleted points (using '*' or '-')."
        )

        user_prompt = (
            f"Please provide a detailed financial analysis for {stock_name}. "
            f"Here is the 'Close' price data for the last {len(history_prices)} trading days, from {start_date} to {end_date}:\n"
            f"Start Price ({start_date}): ₹{start_price:.2f}\n"
            f"End Price ({end_date}): ₹{end_price:.2f}\n"
            f"Full data series (Date: Price): {data_string}\n\n"
            "Based *only* on this data, provide a detailed breakdown of the stock's performance. "
            "Describe the overall trend, identify specific periods of significant price change, and mention any notable peaks or troughs. "
            "Also, comment on the stock's volatility. "
            "**Remember, you must use a 'subtopic: point-wise' format** (e.g., 'Volatility: - The stock showed...'). "
            "Do not use markdown headings and do not give any investment advice."
        )


        print(f"--- Sending request to LM Studio for {stock_name} ---")

        payload = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "model": LOCAL_MODEL_NAME,
            "temperature": 0.7,
            "max_tokens": 10000,
        }

        response = requests.post(LM_STUDIO_API_URL, json=payload)
        response.raise_for_status() # Raise an exception for bad status codes
        
        ai_response = response.json()['choices'][0]['message']['content']
        
        print(f"--- Received response from LM Studio ---")
        
        return jsonify({'insights': ai_response})

    except requests.exceptions.ConnectionError:
        print(f"Error: Could not connect to LM Studio server at {LM_STUDIO_API_URL}")
        return jsonify({'error': 'Failed to connect to local AI server. Is LM Studio running at http://127.0.0.1:1234?'}), 503
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error calling local AI API: {e}")
        # Try to get more error info from response if available
        try:
            error_details = response.json()
            print(f"LM Studio Error Response: {error_details}")
            return jsonify({'error': f'Failed to get AI insights: {str(e)} - {error_details.get("error", "Unknown")}'}), 500
        except:
                return jsonify({'error': f'Failed to get AI insights: {str(e)}'}), 500


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

        # Format for lightweight-charts
        ohlc_data = data[['Open', 'High', 'Low', 'Close']].copy()
        ohlc_data.reset_index(inplace=True)
        ohlc_data.rename(columns={'Date': 'time', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close'}, inplace=True)
        ohlc_data['time'] = ohlc_data['time'].dt.strftime('%Y-%m-%d')
        
        volume_data = data[['Volume']].copy()
        volume_data.reset_index(inplace=True)
        volume_data.rename(columns={'Date': 'time', 'Volume': 'value'}, inplace=True)
        volume_data['time'] = volume_data['time'].dt.strftime('%Y-%m-%d')

        rsi_data = data[['RSI_14']].copy()
        rsi_data.reset_index(inplace=True)
        rsi_data.rename(columns={'Date': 'time', 'RSI_14': 'value'}, inplace=True)
        rsi_data['time'] = rsi_data['time'].dt.strftime('%Y-%m-%d')

        sma_data = data[['SMA_50']].copy()
        sma_data.reset_index(inplace=True)
        sma_data.rename(columns={'Date': 'time', 'SMA_50': 'value'}, inplace=True)
        sma_data['time'] = sma_data['time'].dt.strftime('%Y-%m-%d')
        
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


# --- 7. Run the App ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)

