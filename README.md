# NeuroStock — NSE Stock Trend Predictor AI

## 📌 Project Overview
**NeuroStock (NSE Stock Trend Predictor AI)** is a full-stack quantitative forecasting and financial analysis platform for National Stock Exchange (NSE) listed equities. It pairs a **GPU-accelerated State-of-the-Art ML Stack** (stationary log-return GBDTs + PyTorch TCN-BiGRU-Attention + Out-of-Fold Stacking Ensemble + Direct Multi-Horizon Quantile Forecasting + GARCH(1,1) Student-t Monte Carlo + FinBERT sentiment) with a **Local LLM** via LM Studio for conversational market insights and financial education.

## 🚀 Key Features

* **1-Day Stationary Log-Return Forecast**: Predicts next-day log returns from **133 stationary & cross-sectional alpha features** (multi-horizon returns, candlestick microstructure, EMA/MACD/RSI/ADX/Stochastic, Garman-Klass & ATR volatility, MFI/OBV/VWAP flow, Nifty 50 `^NSEI` & India VIX `^INDIAVIX` macro regime context, lagged global cross-asset drivers, cross-sectional percentile ranks, and cyclical calendar encodings) — eliminating raw-price non-stationarity and tree extrapolation failure.
* **Direct Multi-Horizon Forecasting + 90% Confidence Cones**: Trains direct models across horizons $h \in \{1, 5, 10, 21, 63\}$ trading days with LightGBM quantile regression (`q10`, `q50`, `q90`) and Bayesian-shrunk equilibrium drift beyond 63 days — eliminating recursive walk-forward drift explosion.
* **State-of-the-Art Model Lineup** (Purged Walk-Forward CV + Out-of-Sample Benchmarks):
  * **Ensemble (Recommended)**: Non-negative convex stacking meta-learner trained on strictly out-of-fold predictions of `LightGBM`, `CatBoost`, `XGBoost`, `AttentionNet`, `RandomForest`, `SVR`, and `Ridge`.
  * **AttentionNet**: PyTorch **Multi-Scale Dilated TCN + 2-Layer BiGRU + Multi-Head Self-Attention** trained with Huber loss, `AdamW`, Cosine LR schedule, and CUDA AMP mixed precision.
  * **GBDT & Classical Suite**: `LightGBM`, `CatBoost`, `XGBoost`, `RandomForest`, `SVR`, and `Ridge`.
* **Institutional Portfolio Projector**: 10,000-path vectorized **GARCH(1,1) Student-t Monte Carlo** simulation with Bayesian drift shrinkage, returning P10/P50/P90 wealth cones, Sharpe Ratio, Sortino Ratio, Max Drawdown, 95% VaR/CVaR, and Probability of Profit.
* **Multi-Factor AI Investment Engine**: Ranks NSE equities via a 5-factor **Risk-Adjusted Alpha Score** (Forecasted Sharpe 35%, Ensemble Consensus 20%, Downside Stability 15%, Relative Strength vs Nifty 50 15%, and FinBERT Sentiment 15%).
* **FinBERT News Sentiment Analysis**: Local `ProsusAI/finbert` financial transformer scoring up to 12 articles with exponential recency decay, plus optional LM Studio executive synthesis.
* **AI Market Assistant & Financial Tutor**: Conversational `TrendAI` market analyst and `FinTeach` educator powered by LM Studio.

## 🛠️ Tech Stack

### Backend & ML
* **Language**: Python 3.10+ (CUDA-enabled Conda environment recommended for NVIDIA GPUs)
* **Framework**: Flask (Application Factory + Blueprints), Flask-CORS
* **Deep Learning & Financial NLP**: PyTorch (CUDA AMP FP16/TF32), Hugging Face Transformers (`ProsusAI/finbert`)
* **GBDT & Statistical Libraries**: LightGBM, CatBoost, XGBoost, Scikit-learn, SciPy, `arch` (GARCH)
* **Data Processing**: Pandas, NumPy, yfinance, Joblib

### Frontend
* **Core**: React 18, React DOM
* **HTTP Client**: Centralized Axios API client (`services/apiClient.js`)
* **Visualization**: Lightweight Charts (TradingView)
* **Authentication**: Firebase SDK
* **Testing**: Jest, React Testing Library

## 📂 Project Structure

```
Stock_Predictor/
├── backend/
│   ├── app.py                                    # Flask application factory (create_app) & entry point
│   ├── config.py                                 # Centralized constants, SOTA algorithm list & paths
│   ├── requirements.txt                          # Python dependencies
│   ├── routes/                                   # Thin Flask Blueprints
│   │   ├── ai_routes.py                          # /get-ai-insights, /get-general-knowledge
│   │   ├── market_routes.py                      # /get-market-data
│   │   ├── portfolio_routes.py                   # /calculate-portfolio, /investment-engine
│   │   ├── prediction_routes.py                  # /predict, /historical-data
│   │   └── sentiment_routes.py                   # /get-sentiment
│   └── services/                                 # Reusable quantitative & AI services
│       ├── ai_assistant_service.py               # TrendAI & FinTeach orchestration
│       ├── attention_net.py                      # PyTorch TCN + BiGRU + Multi-Head Attention model
│       ├── feature_engineering.py                # 133 stationary alpha, cross-sectional & macro regime features
│       ├── investment_engine_service.py          # 5-factor Risk-Adjusted Alpha Ranker
│       ├── llm_client.py                         # Shared LM Studio client
│       ├── market_data_service.py                # yfinance download, chart formatting & ticker lookup
│       ├── model_registry.py                     # Loader for .pt, .cbm, .joblib & metrics_summary.json
│       ├── portfolio_service.py                  # 10,000-path GARCH(1,1)-t Monte Carlo engine
│       ├── prediction_service.py                 # 1-day & direct multi-horizon inference
│       └── sentiment_service.py                  # ProsusAI/finbert + recency decay scorer
├── ml/                                           # GPU/multi-core training pipelines
│   ├── training_utils.py                         # Purged walk-forward CV, GPU/CPU trainers & stacker
│   ├── train_one_day_models.py                   # 1-day stationary log-return pipeline (--stocks, --resume)
│   └── train_long_range_models.py                # Direct multi-horizon (5/10/21/63d) + quantile pipeline
├── models/                                       # Trained artifacts (.pt, .cbm, .joblib) & metrics_summary.json
│   ├── one_day/
│   └── long_range/
├── frontend/                                     # React 18 SPA (feature-sliced src/)
├── tests/                                        # Endpoint & integration verification scripts
├── scripts/
│   └── generate_technical_spec.py
├── docs/                                         # Technical specs, reports & presentations
└── README.md
```

## ⚙️ Installation & Manual Retraining (NVIDIA GPU / Conda)

### 1. Activate Your Conda Environment & Install GPU Dependencies
Install **PyTorch with CUDA** first so `AttentionNet` and `FinBERT` run natively on your NVIDIA GPU on Windows, then install the remaining packages:
```powershell
conda activate "<your-conda-env-name>"
cd "c:\Users\bened\Documents\Alyster Coding\PROJECTS\Stock_Predictor"

pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r backend/requirements.txt
```
Verify GPU detection:
```powershell
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

### 2. Retrain the Global Cross-Sectional Panel Models
Trains unified multi-seed bagged **Global Cross-Sectional Panel** models across **all 100 NSE stocks (2005–present, ~400,000+ rows)** with stationary + cross-sectional + lagged global macro features, OOF variance calibration, and shrinkage stacking:
```powershell
python ml/train_one_day_models.py
python ml/train_long_range_models.py --resume
```
Out-of-sample benchmark metrics (Directional Accuracy, MAPE, RMSE, Skill vs Naive, Rank IC, ICIR, and Long-Short Sharpe) are saved to `models/metrics_summary.json` under `GLOBAL_PANEL` and per-ticker keys.

### 3. Frontend Setup
```powershell
cd "c:\Users\bened\Documents\Alyster Coding\PROJECTS\Stock_Predictor\frontend"
npm install
```

## 🏃‍♂️ Running the Application

**Terminal 1 — Backend:**
```powershell
conda activate "<your-conda-env-name>"
cd "c:\Users\bened\Documents\Alyster Coding\PROJECTS\Stock_Predictor"
python backend/app.py
```

**Terminal 2 — Frontend:**
```powershell
cd "c:\Users\bened\Documents\Alyster Coding\PROJECTS\Stock_Predictor\frontend"
npm start
```

## ⚠️ Disclaimer
This tool is for **educational and research purposes only**. Financial markets are inherently stochastic and forecasts do not guarantee future returns. Do not use this application for real financial trading.
