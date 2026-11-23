# NeuroStock — NSE Stock Trend Predictor AI

**Author:** Alyster Benedict

## Project Overview
**NeuroStock (NSE Stock Trend Predictor AI)** is a full-stack quantitative forecasting and financial analysis platform for National Stock Exchange (NSE) listed equities. It pairs an **asynchronous FastAPI/Uvicorn backend** running a **GPU-accelerated ML Stack** (stationary log-return GBDTs + PyTorch TCN-BiGRU-Attention + Out-of-Fold Stacking Ensemble + Direct Multi-Horizon Quantile Forecasting + GARCH(1,1) Student-t Monte Carlo + FinBERT sentiment) with **hybrid LLM routing** — querying **Local LLM (LM Studio)** first and automatically failing over to **NVIDIA NIM (`nvidia/nemotron-3-ultra-550b-a55b`)** when local inference is unavailable.

## Key Features

* **1-Day Stationary Log-Return Forecast**: Predicts next-day log returns from **133 stationary & cross-sectional alpha features** (multi-horizon returns, candlestick microstructure, EMA/MACD/RSI/ADX/Stochastic, Garman-Klass & ATR volatility, MFI/OBV/VWAP flow, Nifty 50 `^NSEI` & India VIX `^INDIAVIX` macro regime context, lagged global cross-asset drivers, cross-sectional percentile ranks, and cyclical calendar encodings).
* **Direct Multi-Horizon Forecasting + 90% Confidence Cones**: Direct models across horizons $h \in \{1, 5, 10, 21, 63\}$ trading days with LightGBM quantile regression (`q10`, `q50`, `q90`) and Bayesian-shrunk equilibrium drift beyond 63 days.
* **State-of-the-Art Model Lineup** (Purged Walk-Forward CV + Out-of-Sample Benchmarks):
  * **Ensemble (Recommended)**: Non-negative convex stacking meta-learner trained on strictly out-of-fold predictions of `LightGBM`, `CatBoost`, `XGBoost`, `AttentionNet`, `RandomForest`, `SVR`, and `Ridge`.
  * **AttentionNet**: PyTorch **Multi-Scale Dilated TCN + 2-Layer BiGRU + Multi-Head Self-Attention** with FiLM entity conditioning.
  * **GBDT & Classical Suite**: `LightGBM`, `CatBoost`, `XGBoost`, `RandomForest`, `SVR`, and `Ridge`.
* **NSE 100 Alpha Radar & Walk-Forward Backtest Lab**: Vectorized cross-sectional scoring across 100 NSE equities, Sector Rotation Quadrant, Macro Regime indicator, and walk-forward strategy simulator with realistic NSE cost assumptions.
* **Institutional Portfolio Projector & Multi-Stock Optimizer**: Cholesky-correlated **GARCH(1,1) Student-t Monte Carlo** simulation with Black-Litterman / Min-Variance / Risk-Parity optimization, Efficient Frontier, correlation matrix, and historical stress tests.
* **FinBERT News Sentiment & TrendAI Assistant**: Local `ProsusAI/finbert` financial transformer scoring recent headlines with recency decay, plus `TrendAI` and `FinTeach` powered by **Local LLM** (LM Studio) with automatic NVIDIA NIM (`Nemotron 550B`) fallback using per-account AES-256-GCM encrypted API keys.

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
├── app.py                                    # Hugging Face Spaces entrypoint (Gradio + FastAPI + /gradio_api middleware)
├── requirements.txt                          # Root Python dependencies for Hugging Face Spaces
├── packages.txt                              # System packages for Hugging Face Spaces (libgomp1 for LightGBM)
├── firebase.json                             # Firebase Hosting SPA config (serves frontend/build)
├── .firebaserc                               # Default Firebase project mapping
├── backend/
│   ├── app.py                                # FastAPI ASGI factory, lifespan, middleware & uvicorn entry
│   ├── config.py                             # Centralized config, NSE universe & security settings
│   ├── schemas.py                            # Strict Pydantic v2 request validation schemas
│   ├── requirements.txt                      # Backend Python dependencies
│   ├── .env.example                          # Backend environment template
│   ├── security/                             # Auth, integrity, sanitizers, logging & ASGI middleware
│   │   ├── auth.py                           # Stateless Firebase RS256 JWT verification
│   │   ├── integrity.py                      # SHA-256 model manifest & path containment verifier
│   │   ├── key_vault.py                      # Per-user AES-256-GCM + HKDF-SHA256 encrypted API key vault
│   │   ├── logging_utils.py                  # Structured logging & CRLF log sanitization
│   │   ├── middleware.py                     # Security headers, body size cap & sliding-window rate limiter
│   │   └── sanitizers.py                     # Ticker, URL, SSRF & prompt-injection sanitizers
│   ├── routes/                               # Async FastAPI APIRouters
│   └── services/                             # Async runtime, ML inference, Monte Carlo, FinBERT & LLM services
├── ml/                                       # Purged walk-forward CV & global panel trainers
├── models/                                   # Trained artifacts + manifest.sha256.json + metrics_summary.json
├── frontend/                                 # React 18 SPA (lazy tabs, ErrorBoundary, XSS-safe export)
│   ├── .env.example                          # Frontend Firebase & API base URL template
│   └── package.json                          # CRA scripts & local dev proxy (http://127.0.0.1:5000)
├── tests/                                    # Pytest security, concurrency & API contract suite
├── scripts/
│   └── generate_technical_spec.py
├── SECURITY.md
└── README.md
```

## Prerequisites

* **Python**: 3.10+ (Conda recommended for NVIDIA CUDA GPUs; `venv` also supported)
* **Node.js & npm**: Node.js 18+ (Node 22 used in CI)
* **Git** (plus **Git LFS** if pushing binary model files >10 MB via Git to Hugging Face Spaces)
* **Optional Deployment CLIs**:
  * Firebase CLI (`firebase-tools` or `npx firebase-tools`) for Firebase Hosting
  * Hugging Face CLI (`huggingface_hub`) for Hugging Face Spaces uploads

## Installation & Setup

> **Important Note for New Repo Installers:**
> 1. **Firebase Project Required:** `frontend/src/services/firebaseAuth.js` initializes Firebase Auth from `frontend/.env`, and `frontend/src/App.js` gates the workspace behind Firebase login. Because `.env` files are gitignored for security, fresh clones only include `frontend/.env.example` and `backend/.env.example` with placeholder values (`your-firebase-project-id`, `your-firebase-web-api-key`). You **must** create your own Firebase project, enable **Email/Password** authentication in Firebase Console, and populate `backend/.env` and `frontend/.env` before you can sign in and use the app.
> 2. **Local Entrypoint (`backend/app.py` vs `app.py`):** Always run `python backend/app.py` for local development. The root `app.py` is the Hugging Face Spaces entrypoint and imports `gradio` and `spaces` (provided automatically by Hugging Face Spaces' Gradio SDK, not listed in `requirements.txt` or `backend/requirements.txt`), so running `python app.py` locally will fail unless you separately run `pip install gradio spaces`.

### 1. Clone the Repository
```powershell
git clone https://github.com/AlysterBenedict/NSE_Stock_Trend_AI.git
cd NSE_Stock_Trend_AI
```

### 2. Backend Environment

**Option A — Conda + NVIDIA CUDA GPU (Recommended):**
```powershell
conda create -n stock_predictor_env python=3.10 -y
conda activate stock_predictor_env

# Install PyTorch with CUDA first on NVIDIA GPU machines:
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r backend/requirements.txt
```

**Option B — Standard Python `venv` (CPU / Cross-Platform):**
```powershell
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r backend/requirements.txt
```
```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r backend/requirements.txt
```

**Optional — Install Test & Security Lint Tools:**
```powershell
pip install pytest pytest-asyncio ruff bandit
```

**Configure Backend Environment Variables:**
```powershell
# Windows (PowerShell / CMD)
copy backend\.env.example backend\.env
```
```bash
# macOS / Linux
cp backend/.env.example backend/.env
```

Edit `backend/.env` as needed:
* `APP_ENV=development` (`production` enforces fail-closed Firebase auth and strict CORS)
* `HOST=127.0.0.1` and `PORT=5000`
* `FIREBASE_PROJECT_ID=<your-firebase-project-id>` (or set `AUTH_REQUIRED=false` for local offline dev only; never disable auth in production)
* `ALLOWED_ORIGINS=http://localhost:3000,http://127.0.0.1:3000`
* `LM_STUDIO_API_URL=http://127.0.0.1:1234/v1/chat/completions` (optional local LLM)
* `NVIDIA_API_KEY=<your-nvidia-nim-key>` (optional server-level fallback; users can also bind encrypted per-account keys in the UI)
* `LLM_KEY_ENCRYPTION_SECRET=<32+ char random secret>` (recommended for persistent AES-256-GCM vault encryption)

### 3. Frontend Setup
```powershell
cd frontend
copy .env.example .env      # macOS/Linux: cp .env.example .env
npm install
cd ..
```

Edit `frontend/.env`:
* Populate `REACT_APP_FIREBASE_API_KEY`, `REACT_APP_FIREBASE_AUTH_DOMAIN`, `REACT_APP_FIREBASE_PROJECT_ID`, `REACT_APP_FIREBASE_STORAGE_BUCKET`, `REACT_APP_FIREBASE_MESSAGING_SENDER_ID`, and `REACT_APP_FIREBASE_APP_ID` with your Firebase Web App credentials.
* **Important for local development:** Leave `REACT_APP_API_BASE_URL` commented out or set to `""` in `frontend/.env` so Create React App's `"proxy": "http://127.0.0.1:5000"` in `frontend/package.json` forwards API requests to your local backend.

### 4. Verify Model Integrity & Retrain Models (Optional)
Pre-trained model artifacts and `models/manifest.sha256.json` are already included in `models/`. To verify their SHA-256 checksums:
```powershell
python -c "from backend.security.integrity import verify_all_manifest_entries; verify_all_manifest_entries()"
```

To retrain the global cross-sectional panel models from scratch (automatically refreshes `models/manifest.sha256.json` upon completion):
```powershell
python ml/train_one_day_models.py
python ml/train_long_range_models.py --resume
```

## Running the Application

### Local Full-Stack Development (Backend + Frontend)

**Terminal 1 — Async Backend (FastAPI / Uvicorn on `http://127.0.0.1:5000`):**
```powershell
conda activate stock_predictor_env   # or activate .venv
python backend/app.py
```
*(Alternative Uvicorn reload mode for development: `uvicorn backend.app:app --host 127.0.0.1 --port 5000 --reload`)*

**Terminal 2 — React Frontend (on `http://localhost:3000`):**
```powershell
cd frontend
npm start
```
> Ensure `REACT_APP_API_BASE_URL` in `frontend/.env` is `""` or commented out when running locally so API calls route through the local proxy (`http://127.0.0.1:5000`) instead of a remote deployment.

### Local Hugging Face Space Wrapper Test (Optional)
To test the root Gradio + FastAPI ASGI entrypoint (`app.py`, serving on port `7860` with `/gradio_api/*` routing) locally, install the Space SDK packages first:
```powershell
conda activate stock_predictor_env
pip install gradio spaces
python app.py
```

### Hybrid LLM Setup (Optional)
1. **Primary (Local LLM)**: Start **LM Studio**'s local OpenAI-compatible server on `http://127.0.0.1:1234/v1/chat/completions` with a loaded model (default `LOCAL_MODEL_NAME=google/gemma-4-26b-a4b`).
2. **Automatic Fallback (NVIDIA NIM)**: If local LM Studio is offline, requests automatically fail over to `nvidia/nemotron-3-ultra-550b-a55b` using either `NVIDIA_API_KEY` in `backend/.env` or a user's encrypted key saved via the UI.

## Verification & Testing

```powershell
# Backend lint, static security scan, model manifest check & test suite
ruff check backend ml tests
bandit -r backend -x backend/__pycache__ -ll -ii
python -c "from backend.security.integrity import verify_all_manifest_entries; verify_all_manifest_entries()"
pytest -v tests/

# Frontend unit/security tests & production build
cd frontend
npm test -- --watchAll=false
npm run build
cd ..
```

## Deployment & Pushing Commands

### 1. Push to GitHub (General Git Workflow)

Check your repository status and configured remotes:
```powershell
git status
git remote -v
```

Stage, commit, and push changes to GitHub (`origin`):
```powershell
git add .
git commit -m "Describe your changes"
git push origin master
```

Common general GitHub commands:
```powershell
# Add a new GitHub remote (if setting up a fresh repo or secondary remote)
git remote add origin https://github.com/<USERNAME>/<REPO>.git

# Push and set upstream tracking for a new branch
git checkout -b <branch-name>
git push -u origin <branch-name>

# Pull latest changes before pushing
git pull --rebase origin master

# Push to a secondary remote (e.g., team/mirror repo)
git remote add <remote-name> https://github.com/<ORG>/<REPO>.git
git push <remote-name> master
```

---

### 2. Push & Deploy Backend to Hugging Face Spaces

The root `app.py`, `requirements.txt`, and `packages.txt` configure NeuroStock for **Hugging Face Spaces (Gradio SDK, Blank template, ZeroGPU or CPU hardware)**.

#### A. Configure Space Secrets on Hugging Face
In your Space's **Settings -> Variables and secrets**, set:
* `APP_ENV` = `production`
* `AUTH_REQUIRED` = `true`
* `FIREBASE_PROJECT_ID` = `<your-firebase-project-id>` (e.g., `stockpredictor-48837`)
* `ALLOWED_ORIGINS` = `https://<your-site>.web.app,https://<your-site>.firebaseapp.com` (e.g., `https://neurostock.web.app,https://stockpredictor-48837.web.app,https://stockpredictor-48837.firebaseapp.com`)
* `TRUST_PROXY_HEADERS` = `true`
* `ENABLE_HSTS` = `true`
* `LLM_KEY_ENCRYPTION_SECRET` = `<32+ character random secret>`
* `NVIDIA_API_KEY` = `<optional-server-fallback-key>`

#### B. Option 1 — Push via Hugging Face CLI (`hf` / `huggingface-cli`, Recommended)
`hf upload` handles large model binaries (>10 MB `.pt` files in `models/`) automatically without manual Git LFS setup:
```powershell
pip install -U huggingface_hub
hf auth login

# Upload the repository to your Space (replace <HF_USERNAME>/<SPACE_NAME>)
# Note: hf upload does not read .gitignore automatically, so exclude local secrets/build/docs explicitly
hf upload <HF_USERNAME>/<SPACE_NAME> . --repo-type=space --exclude "frontend/node_modules/*" "frontend/build/*" "frontend/.env*" "backend/.env*" ".env*" ".git/*" ".firebase/*" ".firebaserc" "API keys.txt" "*.key" "*vault*.json" "docs/*" "*.log" "__pycache__/*"
```

#### C. Option 2 — Push via Git + Git LFS
Hugging Face rejects raw Git blobs >=10 MB (such as `models/one_day/global_AttentionNet.pt`), so enable Git LFS and migrate existing model commits before pushing via Git:
```powershell
# One-time Git LFS setup for model artifacts
git lfs install
git lfs track "*.pt" "*.joblib" "*.cbm"
git add .gitattributes

# If model binaries are already in your local commit history, rewrite them into LFS pointers:
git lfs migrate import --include="*.pt,*.joblib,*.cbm" --include-ref=refs/heads/master

# Add your Hugging Face Space as a Git remote
git remote add hf https://huggingface.co/spaces/<HF_USERNAME>/<SPACE_NAME>

# Push local master to the Space's main branch
git push hf master:main
```
*(Note: `git lfs migrate import` rewrites local commit SHAs—use a separate branch or `hf upload` above if you want to keep your GitHub history untouched. If your Space has an initial README commit on `main` that diverges from your local history, pull/merge it first or run `git push --force hf master:main`.)*

Once deployed, your FastAPI routes are exposed under `/gradio_api`:
`https://<hf-username>-<space-name>.hf.space/gradio_api` (health check: `/gradio_api/health`).

---

### 3. Build & Deploy Frontend to Firebase Hosting

`firebase.json` serves `frontend/build` as a single-page app (`"site": "neurostock"`) and `.firebaserc` targets the default Firebase project (`stockpredictor-48837`).

1. **Point `frontend/.env` to your deployed Hugging Face Space backend** (include `/gradio_api`):
   ```env
   REACT_APP_API_BASE_URL="https://<hf-username>-<space-name>.hf.space/gradio_api"
   ```
   *(Ensure your Firebase Web App `REACT_APP_FIREBASE_*` keys are also set in `frontend/.env`, and add your Firebase Hosting domain(s) to **Firebase Console -> Authentication -> Settings -> Authorized domains** as well as `ALLOWED_ORIGINS` on Hugging Face.)*

2. **Build the production React bundle:**
   ```powershell
   cd frontend
   npm run build
   cd ..
   ```

3. **Authenticate & deploy to Firebase Hosting** (from the repository root):
   ```powershell
   # Log in to Firebase (one-time)
   npx firebase-tools login

   # Select the target Firebase project (matches .firebaserc)
   npx firebase-tools use stockpredictor-48837

   # Deploy only Hosting
   npx firebase-tools deploy --only hosting
   ```
   *(If `firebase-tools` is installed globally via `npm install -g firebase-tools`, you can run `firebase login` and `firebase deploy --only hosting` directly.)*

4. **Switch back to local dev (optional):** Comment out or clear `REACT_APP_API_BASE_URL` in `frontend/.env` if you return to running `npm start` against your local `127.0.0.1:5000` backend.

## Disclaimer
This tool is for **educational and research purposes only**. Financial markets are inherently stochastic and forecasts do not guarantee future returns. Do not use this application for real financial trading.
