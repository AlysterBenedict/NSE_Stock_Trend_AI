# NeuroStock — NSE Stock Trend Predictor AI

**Author:** Alyster Benedict

## Project Overview
**NeuroStock (NSE Stock Trend Predictor AI)** is a full-stack quantitative forecasting and financial analysis platform for National Stock Exchange (NSE) listed equities. It pairs an **asynchronous FastAPI/Uvicorn backend** running a **GPU-accelerated ML Stack** (stationary log-return GBDTs + PyTorch TCN-BiGRU-Attention + Out-of-Fold Stacking Ensemble + Direct Multi-Horizon Quantile Forecasting + GARCH(1,1) Student-t Monte Carlo + FinBERT sentiment) with a **Local LLM** via LM Studio for conversational market insights and financial education.

## Key Features

* **1-Day Stationary Log-Return Forecast**: Predicts next-day log returns from **133 stationary & cross-sectional alpha features** (multi-horizon returns, candlestick microstructure, EMA/MACD/RSI/ADX/Stochastic, Garman-Klass & ATR volatility, MFI/OBV/VWAP flow, Nifty 50 `^NSEI` & India VIX `^INDIAVIX` macro regime context, lagged global cross-asset drivers, cross-sectional percentile ranks, and cyclical calendar encodings).
* **Direct Multi-Horizon Forecasting + 90% Confidence Cones**: Direct models across horizons $h \in \{1, 5, 10, 21, 63\}$ trading days with LightGBM quantile regression (`q10`, `q50`, `q90`) and Bayesian-shrunk equilibrium drift beyond 63 days.
* **State-of-the-Art Model Lineup** (Purged Walk-Forward CV + Out-of-Sample Benchmarks):
  * **Ensemble (Recommended)**: Non-negative convex stacking meta-learner trained on strictly out-of-fold predictions of `LightGBM`, `CatBoost`, `XGBoost`, `AttentionNet`, `RandomForest`, `SVR`, and `Ridge`.
  * **AttentionNet**: PyTorch **Multi-Scale Dilated TCN + 2-Layer BiGRU + Multi-Head Self-Attention** with FiLM entity conditioning.
  * **GBDT & Classical Suite**: `LightGBM`, `CatBoost`, `XGBoost`, `RandomForest`, `SVR`, and `Ridge`.
* **NSE 100 Alpha Radar & Walk-Forward Backtest Lab**: Vectorized cross-sectional scoring across 100 NSE equities, Sector Rotation Quadrant, Macro Regime indicator, and walk-forward strategy simulator with realistic NSE cost assumptions.
* **Institutional Portfolio Projector & Multi-Stock Optimizer**: Cholesky-correlated **GARCH(1,1) Student-t Monte Carlo** simulation with Black-Litterman / Min-Variance / Risk-Parity optimization, Efficient Frontier, correlation matrix, and historical stress tests.
* **FinBERT News Sentiment & TrendAI Assistant**: Local `ProsusAI/finbert` financial transformer scoring recent headlines with recency decay, plus `TrendAI` and `FinTeach` powered by LM Studio.

## Security & Concurrency Architecture

* **Async ASGI Backend (FastAPI + Uvicorn)**: Non-blocking pooled `httpx.AsyncClient` I/O, bounded CPU/IO worker executors, CUDA VRAM lock coordination, and async singleflight request coalescing (`AsyncTTLCache`).
* **Stateless Firebase RS256 JWT Verification**: Verifies `Authorization: Bearer <ID Token>` against Google's official X.509 public certificate metadata with strict `aud`, `iss`, `exp`, and `sub` validation (fail-closed in production).
* **Cryptographic Model Integrity**: Every `.joblib`, `.pt`, `.cbm`, and benchmark JSON artifact under `models/` is verified via canonical path containment and SHA-256 digest (`models/manifest.sha256.json`) prior to loading (`torch.load(..., weights_only=True)`).
* **Defense-in-Depth Hardening**: Strict Pydantic v2 input schemas, bounded request payloads (`256 KB`), strict CORS origin allowlist, sliding-window per-user/per-IP rate limiting, security headers (`CSP`, `X-Frame-Options: DENY`, `nosniff`), prompt-injection XML delimiters, `http`/`https` link validation, and XSS-safe DOM brief export.

## Tech Stack

### Backend & ML
* **Language**: Python 3.10+ (Conda environment recommended for CUDA GPUs)
* **Framework**: FastAPI + Uvicorn (ASGI), Pydantic v2, HTTPX, PyJWT
* **Deep Learning & Financial NLP**: PyTorch (CUDA AMP FP16/TF32), Hugging Face Transformers (`ProsusAI/finbert`)
* **GBDT & Statistical Libraries**: LightGBM, CatBoost, XGBoost, Scikit-learn, SciPy, `arch` (GARCH)
* **Data Processing**: Pandas, NumPy, yfinance, Joblib

### Frontend
* **Core**: React 18, React DOM
* **HTTP Client**: Axios (`services/apiClient.js`) with Firebase ID token injection & cancellation
* **Visualization**: Lightweight Charts (TradingView)
* **Authentication**: Firebase SDK
* **Testing**: Jest, React Testing Library

## Project Structure

```
Stock_Predictor/
├── backend/
│   ├── app.py                                    # FastAPI ASGI factory, lifespan, middleware & uvicorn entry
│   ├── config.py                                 # Centralized config, NSE universe & security settings
│   ├── schemas.py                                # Strict Pydantic v2 request validation schemas
│   ├── requirements.txt                          # Pinned Python dependencies
│   ├── security/                                 # Auth, integrity, sanitizers, logging & ASGI middleware
│   │   ├── auth.py                               # Stateless Firebase RS256 JWT verification
│   │   ├── integrity.py                          # SHA-256 model manifest & path containment verifier
│   │   ├── logging_utils.py                      # Structured logging & CRLF log sanitization
│   │   ├── middleware.py                         # Security headers, body size cap & sliding-window rate limiter
│   │   └── sanitizers.py                         # Ticker, URL, SSRF & prompt-injection sanitizers
│   ├── routes/                                   # Async FastAPI APIRouters
│   └── services/                                 # Async runtime, ML inference, Monte Carlo, FinBERT & LLM services
├── ml/                                           # Purged walk-forward CV & global panel trainers
├── models/                                       # Trained artifacts + manifest.sha256.json + metrics_summary.json
├── frontend/                                     # React 18 SPA (lazy tabs, ErrorBoundary, XSS-safe export)
├── tests/                                        # Pytest security, concurrency & API contract suite
├── scripts/
│   └── generate_technical_spec.py
├── SECURITY.md
└── README.md
```

## Installation & Setup

### 1. Backend Environment (Conda + CUDA GPU)
```powershell
conda activate stock_predictor_env

# Install PyTorch with CUDA first on NVIDIA GPU machines:
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r backend/requirements.txt
```

Configure backend environment variables:
```powershell
copy backend\.env.example backend\.env
# Set FIREBASE_PROJECT_ID to your Firebase project ID (or AUTH_REQUIRED=false for local offline dev only)
```

### 2. Frontend Setup
```powershell
cd frontend
copy .env.example .env
# Populate REACT_APP_FIREBASE_* with your Firebase Web App config
npm install
cd ..
```

### 3. Retrain Global Cross-Sectional Panel Models (Optional)
Training automatically refreshes `models/manifest.sha256.json` upon completion:
```powershell
python ml/train_one_day_models.py
python ml/train_long_range_models.py --resume
```

## Running the Application

**Terminal 1 — Async Backend (FastAPI / Uvicorn):**
```powershell
conda activate stock_predictor_env
python backend/app.py
```

**Terminal 2 — Frontend:**
```powershell
cd frontend
npm start
```

## Verification & Testing

```powershell
# Backend lint & security/async test suite
ruff check backend ml tests
pytest -v tests/

# Frontend tests & production build
cd frontend
npm test -- --watchAll=false
npm run build
```

## Disclaimer
This tool is for **educational and research purposes only**. Financial markets are inherently stochastic and forecasts do not guarantee future returns. Do not use this application for real financial trading.
