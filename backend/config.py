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

# Full 100-stock NSE Nifty 100 universe used by Global Panel models & APIs
STOCK_LIST = {
    "ABB India": "ABB.NS",
    "Adani Enterprises": "ADANIENT.NS",
    "Adani Green": "ADANIGREEN.NS",
    "Adani Ports": "ADANIPORTS.NS",
    "Ambuja Cements": "AMBUJACEM.NS",
    "Apollo Hospitals": "APOLLOHOSP.NS",
    "Asian Paints": "ASIANPAINT.NS",
    "Aurobindo Pharma": "AUROPHARMA.NS",
    "Axis Bank": "AXISBANK.NS",
    "Bajaj Auto": "BAJAJ-AUTO.NS",
    "Bajaj Finance": "BAJFINANCE.NS",
    "Bajaj Finserv": "BAJAJFINSV.NS",
    "Bank of Baroda": "BANKBARODA.NS",
    "BEL": "BEL.NS",
    "Bharat Forge": "BHARATFORG.NS",
    "Bharti Airtel": "BHARTIARTL.NS",
    "BHEL": "BHEL.NS",
    "Bosch": "BOSCHLTD.NS",
    "BPCL": "BPCL.NS",
    "Britannia": "BRITANNIA.NS",
    "Canara Bank": "CANBK.NS",
    "CG Power": "CGPOWER.NS",
    "Cholamandalam Inv": "CHOLAFIN.NS",
    "Cipla": "CIPLA.NS",
    "Coal India": "COALINDIA.NS",
    "Coforge": "COFORGE.NS",
    "Colgate-Palmolive": "COLPAL.NS",
    "Dabur India": "DABUR.NS",
    "Divis Lab": "DIVISLAB.NS",
    "DLF": "DLF.NS",
    "Dr Reddys Labs": "DRREDDY.NS",
    "Eicher Motors": "EICHERMOT.NS",
    "GAIL India": "GAIL.NS",
    "Godrej Consumer": "GODREJCP.NS",
    "Grasim": "GRASIM.NS",
    "HAL": "HAL.NS",
    "Havells India": "HAVELLS.NS",
    "HCL Tech": "HCLTECH.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "HDFC Life": "HDFCLIFE.NS",
    "Hero MotoCorp": "HEROMOTOCO.NS",
    "Hindalco": "HINDALCO.NS",
    "Hindustan Unilever": "HINDUNILVR.NS",
    "Hindustan Zinc": "HINDZINC.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "ICICI Lombard": "ICICIGI.NS",
    "Indian Hotels": "INDHOTEL.NS",
    "Indus Towers": "INDUSTOWER.NS",
    "IndusInd Bank": "INDUSINDBK.NS",
    "Info Edge": "NAUKRI.NS",
    "Infosys": "INFY.NS",
    "IOC": "IOC.NS",
    "ITC": "ITC.NS",
    "Jindal Steel": "JINDALSTEL.NS",
    "JSW Steel": "JSWSTEEL.NS",
    "Kotak Mahindra Bank": "KOTAKBANK.NS",
    "L&T": "LT.NS",
    "LTIMindtree": "LTM.NS",
    "Lupin": "LUPIN.NS",
    "M&M": "M&M.NS",
    "Marico": "MARICO.NS",
    "Maruti Suzuki": "MARUTI.NS",
    "Mphasis": "MPHASIS.NS",
    "Muthoot Finance": "MUTHOOTFIN.NS",
    "Nestle India": "NESTLEIND.NS",
    "NTPC": "NTPC.NS",
    "ONGC": "ONGC.NS",
    "Persistent Systems": "PERSISTENT.NS",
    "PFC": "PFC.NS",
    "Pidilite Industries": "PIDILITIND.NS",
    "PNB": "PNB.NS",
    "Power Grid Corp": "POWERGRID.NS",
    "REC Ltd": "RECLTD.NS",
    "Reliance": "RELIANCE.NS",
    "SBI": "SBIN.NS",
    "SBI Life": "SBILIFE.NS",
    "Shree Cement": "SHREECEM.NS",
    "Shriram Finance": "SHRIRAMFIN.NS",
    "Siemens": "SIEMENS.NS",
    "SRF": "SRF.NS",
    "Sun Pharma": "SUNPHARMA.NS",
    "Tata Communications": "TATACOMM.NS",
    "Tata Consumer": "TATACONSUM.NS",
    "Tata Motors": "TMPV.NS",
    "Tata Power": "TATAPOWER.NS",
    "Tata Steel": "TATASTEEL.NS",
    "TCS": "TCS.NS",
    "Tech Mahindra": "TECHM.NS",
    "Titan": "TITAN.NS",
    "Torrent Pharma": "TORNTPHARM.NS",
    "Trent": "TRENT.NS",
    "TVS Motor": "TVSMOTOR.NS",
    "UltraTech Cement": "ULTRACEMCO.NS",
    "UPL": "UPL.NS",
    "Varun Beverages": "VBL.NS",
    "Vedanta": "VEDL.NS",
    "Wipro": "WIPRO.NS",
    "Yes Bank": "YESBANK.NS",
    "Zomato": "ETERNAL.NS",
    "Zydus Lifesciences": "ZYDUSLIFE.NS",
}

# Train the Global Cross-Sectional Panel across ALL 100 NSE stocks (~400,000+ rows)
TRAINABLE_STOCK_LIST = STOCK_LIST

# Sector taxonomy for sector-relative excess return factors & neural sector embeddings
SECTOR_MAP = {
    "ABB.NS": "Infra",
    "ADANIENT.NS": "Commodities",
    "ADANIGREEN.NS": "Energy",
    "ADANIPORTS.NS": "Infra",
    "AMBUJACEM.NS": "Commodities",
    "APOLLOHOSP.NS": "Healthcare",
    "ASIANPAINT.NS": "Consumer",
    "AUROPHARMA.NS": "Healthcare",
    "AXISBANK.NS": "Financials",
    "BAJAJ-AUTO.NS": "Auto",
    "BAJFINANCE.NS": "Financials",
    "BAJAJFINSV.NS": "Financials",
    "BANKBARODA.NS": "Financials",
    "BEL.NS": "Defence",
    "BHARATFORG.NS": "Auto",
    "BHARTIARTL.NS": "Telecom",
    "BHEL.NS": "Defence",
    "BOSCHLTD.NS": "Auto",
    "BPCL.NS": "Energy",
    "BRITANNIA.NS": "Consumer",
    "CANBK.NS": "Financials",
    "CGPOWER.NS": "Infra",
    "CHOLAFIN.NS": "Financials",
    "CIPLA.NS": "Healthcare",
    "COALINDIA.NS": "Energy",
    "COFORGE.NS": "IT",
    "COLPAL.NS": "Consumer",
    "DABUR.NS": "Consumer",
    "DIVISLAB.NS": "Healthcare",
    "DLF.NS": "Infra",
    "DRREDDY.NS": "Healthcare",
    "EICHERMOT.NS": "Auto",
    "ETERNAL.NS": "Consumer",
    "GAIL.NS": "Energy",
    "GODREJCP.NS": "Consumer",
    "GRASIM.NS": "Commodities",
    "HAL.NS": "Defence",
    "HAVELLS.NS": "Consumer",
    "HCLTECH.NS": "IT",
    "HDFCBANK.NS": "Financials",
    "HDFCLIFE.NS": "Financials",
    "HEROMOTOCO.NS": "Auto",
    "HINDALCO.NS": "Commodities",
    "HINDUNILVR.NS": "Consumer",
    "HINDZINC.NS": "Commodities",
    "ICICIBANK.NS": "Financials",
    "ICICIGI.NS": "Financials",
    "INDHOTEL.NS": "Consumer",
    "INDUSINDBK.NS": "Financials",
    "INDUSTOWER.NS": "Telecom",
    "INFY.NS": "IT",
    "IOC.NS": "Energy",
    "ITC.NS": "Consumer",
    "JINDALSTEL.NS": "Commodities",
    "JSWSTEEL.NS": "Commodities",
    "KOTAKBANK.NS": "Financials",
    "LT.NS": "Infra",
    "LTM.NS": "IT",
    "LUPIN.NS": "Healthcare",
    "M&M": "Auto",
    "M&M.NS": "Auto",
    "MARICO.NS": "Consumer",
    "MARUTI.NS": "Auto",
    "MPHASIS.NS": "IT",
    "MUTHOOTFIN.NS": "Financials",
    "NAUKRI.NS": "IT",
    "NESTLEIND.NS": "Consumer",
    "NTPC.NS": "Energy",
    "ONGC.NS": "Energy",
    "PERSISTENT.NS": "IT",
    "PFC.NS": "Financials",
    "PIDILITIND.NS": "Commodities",
    "PNB.NS": "Financials",
    "POWERGRID.NS": "Energy",
    "RECLTD.NS": "Financials",
    "RELIANCE.NS": "Energy",
    "SBILIFE.NS": "Financials",
    "SBIN.NS": "Financials",
    "SHREECEM.NS": "Commodities",
    "SHRIRAMFIN.NS": "Financials",
    "SIEMENS.NS": "Infra",
    "SRF.NS": "Commodities",
    "SUNPHARMA.NS": "Healthcare",
    "TATACOMM.NS": "Telecom",
    "TATACONSUM.NS": "Consumer",
    "TATAPOWER.NS": "Energy",
    "TATASTEEL.NS": "Commodities",
    "TCS.NS": "IT",
    "TECHM.NS": "IT",
    "TITAN.NS": "Consumer",
    "TMPV.NS": "Auto",
    "TORNTPHARM.NS": "Healthcare",
    "TRENT.NS": "Consumer",
    "TVSMOTOR.NS": "Auto",
    "ULTRACEMCO.NS": "Commodities",
    "UPL.NS": "Commodities",
    "VBL.NS": "Consumer",
    "VEDL.NS": "Commodities",
    "WIPRO.NS": "IT",
    "YESBANK.NS": "Financials",
    "ZYDUSLIFE.NS": "Healthcare",
}

SECTORS = sorted(set(SECTOR_MAP.values()) | {"Other"})
SECTOR_TO_ID = {sec: idx for idx, sec in enumerate(SECTORS)}
SORTED_TICKERS = sorted(set(STOCK_LIST.values()))
TICKER_TO_ID = {t: idx for idx, t in enumerate(SORTED_TICKERS)}
