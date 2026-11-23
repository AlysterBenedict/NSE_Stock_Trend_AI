"""Centralized configuration, NSE universe definitions, and security settings for NeuroStock."""

import os

from backend.security.sanitizers import validate_llm_url

try:
    from dotenv import load_dotenv

    _BACKEND_ENV = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.isfile(_BACKEND_ENV):
        load_dotenv(_BACKEND_ENV, override=False)
except ImportError:
    pass

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODELS_DIR = os.path.join(BASE_DIR, "models")
MODELS_DIR_LONG_RANGE = os.path.join(MODELS_DIR, "long_range")
MODELS_DIR_ONE_DAY = os.path.join(MODELS_DIR, "one_day")
METRICS_SUMMARY_PATH = os.path.join(MODELS_DIR, "metrics_summary.json")

# Sequence & Training Hyperparameters (MUST stay aligned with trained model artifacts)
LOOKBACK_WINDOW = 60
START_DATE = "2005-01-01"
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
# Global markets close after NSE 3:30 PM IST -> strictly shifted by 1 trading day (zero lookahead)
GLOBAL_LAGGED_MACRO_TICKERS = {
    "sp500": "^GSPC",
    "us10y": "^TNX",
    "usdinr": "INR=X",
    "brent": "BZ=F",
    "gold": "GC=F",
}
MACRO_TICKERS = {**INDIAN_MACRO_TICKERS, **GLOBAL_LAGGED_MACRO_TICKERS}

# Dynamic security & runtime environment helpers
_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})


def _parse_bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in _TRUE_VALUES


def get_app_env() -> str:
    return os.environ.get("APP_ENV", "development").strip().lower() or "development"


def get_host() -> str:
    return os.environ.get("HOST", "127.0.0.1").strip() or "127.0.0.1"


def get_port() -> int:
    raw = os.environ.get("PORT", "5000").strip()
    try:
        port = int(raw)
    except ValueError:
        return 5000
    return port if 1 <= port <= 65535 else 5000


def get_log_level() -> str:
    return os.environ.get("LOG_LEVEL", "INFO").strip().upper() or "INFO"


def get_firebase_project_id() -> str:
    return os.environ.get("FIREBASE_PROJECT_ID", "").strip()


def is_auth_required() -> bool:
    raw = os.environ.get("AUTH_REQUIRED")
    if raw is not None and raw.strip() != "":
        return raw.strip().lower() in _TRUE_VALUES
    return get_app_env() == "production" or bool(get_firebase_project_id())


def get_allowed_origins() -> list[str]:
    raw = os.environ.get(
        "ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000"
    )
    origins = [item.strip().rstrip("/") for item in raw.split(",") if item.strip()]
    return origins or ["http://localhost:3000", "http://127.0.0.1:3000"]


def is_api_docs_enabled() -> bool:
    return _parse_bool_env("ENABLE_API_DOCS", False)


def is_hsts_enabled() -> bool:
    return _parse_bool_env("ENABLE_HSTS", False)


def is_model_integrity_required() -> bool:
    return _parse_bool_env("VERIFY_MODEL_INTEGRITY", True)


def is_proxy_trusted() -> bool:
    return _parse_bool_env("TRUST_PROXY_HEADERS", False)


def get_max_request_body_bytes() -> int:
    raw = os.environ.get("MAX_REQUEST_BODY_BYTES", "262144").strip()
    try:
        size = int(raw)
    except ValueError:
        return 262144
    return max(4096, min(size, 4 * 1024 * 1024))


def is_remote_llm_allowed() -> bool:
    return _parse_bool_env("ALLOW_REMOTE_LLM_URL", False)


def get_lm_studio_api_url() -> str:
    raw = os.environ.get("LM_STUDIO_API_URL", "http://127.0.0.1:1234/v1/chat/completions")
    return validate_llm_url(raw, allow_remote=is_remote_llm_allowed())


def validate_runtime_security_config() -> None:
    """Enforce fail-closed security invariants before starting the ASGI application."""
    get_lm_studio_api_url()
    if get_app_env() == "production":
        if not is_auth_required():
            raise RuntimeError("AUTH_REQUIRED cannot be disabled when APP_ENV=production.")
        if not get_firebase_project_id():
            raise RuntimeError("FIREBASE_PROJECT_ID must be set when APP_ENV=production.")
        origins = get_allowed_origins()
        if "*" in origins:
            raise RuntimeError("Wildcard '*' CORS origin is forbidden when APP_ENV=production.")


# Local LLM (LM Studio) & Financial NLP settings
LM_STUDIO_API_URL = get_lm_studio_api_url()
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

TRAINABLE_STOCK_LIST = STOCK_LIST
VALID_TICKERS = frozenset(STOCK_LIST.values())
TICKER_TO_NAME = {ticker: name for name, ticker in STOCK_LIST.items()}

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
