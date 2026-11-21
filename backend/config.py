"""Centralized configuration and constants for the NeuroStock Global Panel ML backend."""

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODELS_DIR = os.path.join(BASE_DIR, "models")
MODELS_DIR_LONG_RANGE = os.path.join(MODELS_DIR, "long_range")
MODELS_DIR_ONE_DAY = os.path.join(MODELS_DIR, "one_day")
METRICS_SUMMARY_PATH = os.path.join(MODELS_DIR, "metrics_summary.json")

# Sequence & Training Hyperparameters
LOOKBACK_WINDOW = 60                  # 60 trading days (~3 months) for AttentionNet
START_DATE = "2005-01-01"             # ~20+ years of history across cycles for low-variance global panel
FORECAST_HORIZONS = [1, 5, 10, 21, 63]
LONG_RANGE_HORIZONS = [5, 10, 21, 63]
TARGET_COLUMN = "Close"

# State-of-the-art algorithm lineup
BASE_ALGORITHMS = [
    "LightGBM",
    "CatBoost",
    "XGBoost",
    "AttentionNet",
    "RandomForest",
    "SVR",
    "Ridge",
]
STACKING_BASE_ALGORITHMS = [
    "LightGBM",
    "CatBoost",
    "XGBoost",
    "AttentionNet",
    "RandomForest",
    "SVR",
    "Ridge",
]
ALGORITHMS = ["Ensemble"] + BASE_ALGORITHMS
DEFAULT_ALGORITHM = "Ensemble"

# Macro market regime drivers (Indian same-day close + lagged global cross-asset drivers)
INDIAN_MACRO_TICKERS = {
    "nifty": "^NSEI",
    "vix": "^INDIAVIX",
    "bank_nifty": "^NSEBANK",
    "it_nifty": "^CNXIT",
}
# Global markets close after NSE 3:30 PM IST -> strictly shifted by 1 trading day (zero lookahead!)
GLOBAL_LAGGED_MACRO_TICKERS = {
    "sp500": "^GSPC",
    "us10y": "^TNX",
    "usdinr": "INR=X",
    "brent": "BZ=F",
    "gold": "GC=F",
}
MACRO_TICKERS = {**INDIAN_MACRO_TICKERS, **GLOBAL_LAGGED_MACRO_TICKERS}

# Local LLM (LM Studio) & Financial NLP settings
LM_STUDIO_API_URL = os.environ.get(
    "LM_STUDIO_API_URL", "http://127.0.0.1:1234/v1/chat/completions"
)
LOCAL_MODEL_NAME = os.environ.get("LOCAL_MODEL_NAME", "google/gemma-4-26b-a4b")
FINBERT_MODEL_NAME = os.environ.get("FINBERT_MODEL_NAME", "ProsusAI/finbert")

# Full 58-stock NSE universe (Nifty 50 + extended liquid stocks) used by Global Panel models & APIs
STOCK_LIST = {
    "Adani Enterprises": "ADANIENT.NS",
    "Adani Green": "ADANIGREEN.NS",
    "Adani Ports": "ADANIPORTS.NS",
    "Apollo Hospitals": "APOLLOHOSP.NS",
    "Asian Paints": "ASIANPAINT.NS",
    "Axis Bank": "AXISBANK.NS",
    "Bajaj Auto": "BAJAJ-AUTO.NS",
    "Bajaj Finance": "BAJFINANCE.NS",
    "Bajaj Finserv": "BAJAJFINSV.NS",
    "BEL": "BEL.NS",
    "Bharti Airtel": "BHARTIARTL.NS",
    "BPCL": "BPCL.NS",
    "Britannia": "BRITANNIA.NS",
    "Cipla": "CIPLA.NS",
    "Coal India": "COALINDIA.NS",
    "Divis Lab": "DIVISLAB.NS",
    "Dr Reddys Labs": "DRREDDY.NS",
    "Eicher Motors": "EICHERMOT.NS",
    "Grasim": "GRASIM.NS",
    "HAL": "HAL.NS",
    "HCL Tech": "HCLTECH.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "HDFC Life": "HDFCLIFE.NS",
    "Hero MotoCorp": "HEROMOTOCO.NS",
    "Hindalco": "HINDALCO.NS",
    "Hindustan Unilever": "HINDUNILVR.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "IndusInd Bank": "INDUSINDBK.NS",
    "Infosys": "INFY.NS",
    "ITC": "ITC.NS",
    "JSW Steel": "JSWSTEEL.NS",
    "Kotak Mahindra Bank": "KOTAKBANK.NS",
    "L&T": "LT.NS",
    "LTIMindtree": "LTIM.NS",
    "Lupin": "LUPIN.NS",
    "M&M": "M&M.NS",
    "Maruti Suzuki": "MARUTI.NS",
    "Nestle India": "NESTLEIND.NS",
    "NTPC": "NTPC.NS",
    "ONGC": "ONGC.NS",
    "Power Grid Corp": "POWERGRID.NS",
    "Reliance": "RELIANCE.NS",
    "SBI": "SBIN.NS",
    "SBI Life": "SBILIFE.NS",
    "Sun Pharma": "SUNPHARMA.NS",
    "Tata Consumer": "TATACONSUM.NS",
    "Tata Motors": "TATAMOTORS.NS",
    "Tata Steel": "TATASTEEL.NS",
    "TCS": "TCS.NS",
    "Tech Mahindra": "TECHM.NS",
    "Titan": "TITAN.NS",
    "Trent": "TRENT.NS",
    "UltraTech Cement": "ULTRACEMCO.NS",
    "UPL": "UPL.NS",
    "Varun Beverages": "VBL.NS",
    "Wipro": "WIPRO.NS",
    "Yes Bank": "YESBANK.NS",
    "Zomato": "ZOMATO.NS",
}

# Train the Global Cross-Sectional Panel across ALL 58 NSE stocks (~150,000+ rows)
TRAINABLE_STOCK_LIST = STOCK_LIST

# Sector taxonomy for sector-relative excess return factors & neural sector embeddings
SECTOR_MAP = {
    "ADANIENT.NS": "Commodities",
    "ADANIGREEN.NS": "Energy",
    "ADANIPORTS.NS": "Infra",
    "APOLLOHOSP.NS": "Healthcare",
    "ASIANPAINT.NS": "Consumer",
    "AXISBANK.NS": "Financials",
    "BAJAJ-AUTO.NS": "Auto",
    "BAJFINANCE.NS": "Financials",
    "BAJAJFINSV.NS": "Financials",
    "BEL.NS": "Defence",
    "BHARTIARTL.NS": "Telecom",
    "BPCL.NS": "Energy",
    "BRITANNIA.NS": "Consumer",
    "CIPLA.NS": "Healthcare",
    "COALINDIA.NS": "Energy",
    "DIVISLAB.NS": "Healthcare",
    "DRREDDY.NS": "Healthcare",
    "EICHERMOT.NS": "Auto",
    "GRASIM.NS": "Commodities",
    "HAL.NS": "Defence",
    "HCLTECH.NS": "IT",
    "HDFCBANK.NS": "Financials",
    "HDFCLIFE.NS": "Financials",
    "HEROMOTOCO.NS": "Auto",
    "HINDALCO.NS": "Commodities",
    "HINDUNILVR.NS": "Consumer",
    "ICICIBANK.NS": "Financials",
    "INDUSINDBK.NS": "Financials",
    "INFY.NS": "IT",
    "ITC.NS": "Consumer",
    "JSWSTEEL.NS": "Commodities",
    "KOTAKBANK.NS": "Financials",
    "LT.NS": "Infra",
    "LTIM.NS": "IT",
    "LUPIN.NS": "Healthcare",
    "M&M.NS": "Auto",
    "MARUTI.NS": "Auto",
    "NESTLEIND.NS": "Consumer",
    "NTPC.NS": "Energy",
    "ONGC.NS": "Energy",
    "POWERGRID.NS": "Energy",
    "RELIANCE.NS": "Energy",
    "SBIN.NS": "Financials",
    "SBILIFE.NS": "Financials",
    "SUNPHARMA.NS": "Healthcare",
    "TATACONSUM.NS": "Consumer",
    "TATAMOTORS.NS": "Auto",
    "TATASTEEL.NS": "Commodities",
    "TCS.NS": "IT",
    "TECHM.NS": "IT",
    "TITAN.NS": "Consumer",
    "TRENT.NS": "Consumer",
    "ULTRACEMCO.NS": "Commodities",
    "UPL.NS": "Commodities",
    "VBL.NS": "Consumer",
    "WIPRO.NS": "IT",
    "YESBANK.NS": "Financials",
    "ZOMATO.NS": "Consumer",
}

SECTORS = sorted(set(SECTOR_MAP.values()) | {"Other"})
SECTOR_TO_ID = {sec: idx for idx, sec in enumerate(SECTORS)}
SORTED_TICKERS = sorted(set(STOCK_LIST.values()))
TICKER_TO_ID = {t: idx for idx, t in enumerate(SORTED_TICKERS)}
