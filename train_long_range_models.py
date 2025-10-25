# --- train_long_range_models.py ---
# This script trains ALL 6 models for ALL 5 stocks based ONLY on 'Close' price.
# These models are for the long-range, speculative forecast.

import yfinance as yf
import numpy as np
import pandas as pd
import os
import joblib
import warnings
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from datetime import datetime

# --- 0. Setup ---
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Suppress TensorFlow warnings

# Save models to the base 'stock_models' directory
MODELS_DIR = 'stock_models'
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR, exist_ok=True)
    print(f"Created directory: {MODELS_DIR}")

STOCK_LIST = {
    'Infosys': 'INFY.NS',
    'Yes Bank': 'YESBANK.NS',
    'TCS': 'TCS.NS',
    'HDFC Bank': 'HDFCBANK.NS',
    'ITC': 'ITC.NS'
}
TIME_STEP = 100
START_DATE = '2015-01-01' # Use 10+ years of data


# --- 1. Function to Fetch Data (Close Price Only) ---
def fetch_data(ticker, start_date='2015-01-01'):
    end_date = datetime.now().strftime('%Y-%m-%d') 
    
    print(f"Fetching data for {ticker} from {start_date} to {end_date}...")
    
    data = yf.download(ticker, start=start_date, end=end_date)
    
    if data.empty:
        print(f"No data found for ticker {ticker}.")
        return None, None
    
    # Isolate 'Close' prices and scale them
    close_data = data[['Close']]
    dataset = close_data.values
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_dataset = scaler.fit_transform(dataset)
    
    print(f"Data fetched and scaled successfully for {ticker}")
    return scaled_dataset, scaler

# --- 2. Function to Train the LSTM Model ---
def train_and_save_lstm(ticker, scaled_dataset, scaler):
    print(f"\n--- Training LSTM for {ticker} ---")
    try:
        # A. Create 3D sequences for LSTM
        X_train, y_train = [], []
        for i in range(TIME_STEP, len(scaled_dataset)):
            X_train.append(scaled_dataset[i-TIME_STEP:i, 0])
            y_train.append(scaled_dataset[i, 0])
            
        X_train, y_train = np.array(X_train), np.array(y_train)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

        # B. Build Model
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(units=25))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        # C. Train Model
        print(f"Training LSTM for {ticker}...")
        model.fit(X_train, y_train, batch_size=64, epochs=50, verbose=1)
        
        # D. Save Model and Scaler
        model.save(os.path.join(MODELS_DIR, f"{ticker}_LSTM.h5"))
        joblib.dump(scaler, os.path.join(MODELS_DIR, f"{ticker}_scaler.joblib"))
        print(f"LSTM and Scaler for {ticker} saved.")
        
    except Exception as e:
        print(f"Error training LSTM for {ticker}: {e}")

# --- 3. Function to Train ALL Regressor Models ---
def train_and_save_regressors(ticker, scaled_dataset):
    print(f"\n--- Training Regressors for {ticker} ---")
    
    # A. Create 2D "flattened" data for Regressors
    # X shape will be (samples, 100 features), y shape will be (samples,)
    X, y = [], []
    for i in range(TIME_STEP, len(scaled_dataset)):
        X.append(scaled_dataset[i-TIME_STEP:i, 0]) # This is a 1D array of 100
        y.append(scaled_dataset[i, 0])
            
    X, y = np.array(X), np.array(y)
    
    if X.shape[0] == 0:
        print(f"Not enough data to train regressors for {ticker}.")
        return

    # B. Define the models to train
    models_to_train = {
        "LinearRegression": LinearRegression(),
        "DecisionTree": DecisionTreeRegressor(random_state=42),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "SVR": SVR(kernel='rbf', C=1.0, epsilon=0.1),
        "XGBoost": XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42, n_jobs=-1)
    }
    
    # C. Train and Save each model
    for name, model in models_to_train.items():
        try:
            print(f"Training {name} for {ticker}...")
            model.fit(X, y)
            
            # Save the trained model
            model_path = os.path.join(MODELS_DIR, f"{ticker}_{name}.joblib")
            joblib.dump(model, model_path)
            print(f"Saved {name} for {ticker}.")
            
        except Exception as e:
            print(f"Error training {name} for {ticker}: {e}")

# --- 4. Main Execution Loop ---
if __name__ == "__main__":
    print(f"Starting long-range model training pipeline...")
    print(f"Models will be saved in: {MODELS_DIR}")
    
    for stock_name, ticker_symbol in STOCK_LIST.items():
        print(f"\n=========================================")
        print(f"Processing: {stock_name} ({ticker_symbol})")
        print(f"=========================================")
        
        # 1. Fetch and scale data
        scaled_data, scaler = fetch_data(ticker_symbol, START_DATE)
        
        if scaled_data is None:
            continue
            
        # 2. Train and save the LSTM (and the scaler)
        train_and_save_lstm(ticker_symbol, scaled_data, scaler)
        
        # 3. Train and save all 5 regressor models
        train_and_save_regressors(ticker_symbol, scaled_data)

    print(f"\n\n--- ALL LONG-RANGE MODELS HAVE BEEN TRAINED AND SAVED! ---")