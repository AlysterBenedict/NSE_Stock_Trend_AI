# NSE Stock Trend Predictor AI

## 📌 Project Overview
The **NSE Stock Trend Predictor AI** is a sophisticated full-stack application designed to forecast stock prices for major National Stock Exchange (NSE) listed companies. It leverages a hybrid machine learning approach, combining high-accuracy short-term models with speculative long-term forecasters, and integrates a **Local LLM (Large Language Model)** via LM Studio to provide qualitative "AI Insights" based on technical indicators.

## 🚀 Key Features

*   **1-Day High-Accuracy Forecast**: Uses feature-rich models trained on OHLCV data + Technical Indicators (RSI, SMA) to predict the next day's closing price.
*   **Long-Range Speculative Forecast**: Uses a walk-forward approach to predict prices for future dates (highly speculative).
*   **Hybrid Model Architecture**:
    *   **LSTM (Long Short-Term Memory)**: For capturing temporal dependencies in time-series data.
    *   **Ensemble Regressors**: XGBoost, RandomForest, SVR, LinearRegression, DecisionTree.
*   **AI Insights**: Integrates with a local LLM (e.g., Gemma, Llama via LM Studio) to act as a "Financial Analyst," explaining market trends in plain English.
*   **User Authentication**: Integrated Firebase Authentication for secure user sign-up and login.
*   **Interactive UI**: A modern React-based frontend using `lightweight-charts` for professional-grade financial plotting.
*   **Comprehensive Data**: Fetches real-time and historical data using `yfinance`.

## 🛠️ Tech Stack

### Backend
*   **Language**: Python 3.x
*   **Framework**: Flask
*   **ML Libraries**: TensorFlow/Keras, Scikit-learn, XGBoost
*   **Data Processing**: Pandas, NumPy, yfinance
*   **Serialization**: Joblib

### Frontend
*   **Core**: React 18, React DOM
*   **State Management/Data**: Axios
*   **Visualization**: Lightweight Charts (TradingView)
*   **Utilities**: date-fns, web-vitals
*   **Authentication**: Firebase SDK
*   **Testing**: Jest, React Testing Library
*   **Styling**: CSS Modules / Standard CSS

### AI / LLM
*   **Tool**: LM Studio (Local Inference Server)
*   **Model**: Compatible with GGUF models (e.g., `google/gemma-3-4b`, Llama 3)

## 📂 Project Structure

```
Stock_Predictor/
├── app.py                     # Main Flask application entry point
├── train_one_day_models.py    # Script to train accurate 1-day feature-rich models
├── train_long_range_models.py # Script to train long-range speculative models
├── stock_models/              # Directory stores trained models (.h5, .joblib)
│   └── one_day/               # Subdirectory for 1-day specific models
├── website/                   # React Frontend project
│   ├── src/                   # Source code for React app
│   ├── public/                # Static assets
│   └── package.json           # Node dependencies
└── README.md                  # Project documentation
```

## ⚙️ Installation & Setup

### 1. Prerequisites
*   **Python 3.8+** installed.
*   **Node.js 16+** and **npm** installed.
*   (Optional) **LM Studio** installed for AI Insights features.

### 2. Backend Setup
1.  Navigate to the project root:
    ```bash
    cd "C:\Users\bened\Documents\Alyster Coding\Stock_Predictor"
    ```
2.  Install Python dependencies:
    ```bash
    pip install flask flask-cors yfinance numpy pandas joblib scikit-learn tensorflow xgboost
    ```
3.  **Train the Models** (Required before first run):
    *   Train the accurate 1-day models:
        ```bash
        python train_one_day_models.py
        ```
    *   Train the long-range models:
        ```bash
        python train_long_range_models.py
        ```

### 3. Frontend Setup
1.  Navigate to the `website` directory:
    ```bash
    cd website
    ```
2.  Install dependencies:
    npm install
    ```

### 4. Firebase Setup
1.  Go to the [Firebase Console](https://console.firebase.google.com/) and create a new project.
2.  Navigate to **Authentication** -> **Sign-in method** and enable **Email/Password** (or your preferred providers).
3.  Go to **Project Settings** -> **General** and register a web app.
4.  Copy the `firebaseConfig` object.
5.  Create a file named `firebase.js` in `website/src/` (if it doesn't exist) and add:
    ```javascript
    // website/src/firebase.js
    import { initializeApp } from "firebase/app";
    import { getAuth } from "firebase/auth";

    const firebaseConfig = {
      apiKey: "YOUR_API_KEY",
      authDomain: "YOUR_PROJECT_ID.firebaseapp.com",
      projectId: "YOUR_PROJECT_ID",
      storageBucket: "YOUR_PROJECT_ID.appspot.com",
      messagingSenderId: "YOUR_SENDER_ID",
      appId: "YOUR_APP_ID"
    };

    const app = initializeApp(firebaseConfig);
    export const auth = getAuth(app);
    export default app;
    ```

### 4. LM Studio Setup (For AI Insights)
1.  Download and install **LM Studio**.
2.  Load a model (e.g., `google/gemma-3-4b-it`).
3.  Start the **Local Inference Server**.
4.  Ensure the server is running at `http://127.0.0.1:1234`.
5.  *Note: If your port is different, update `LM_STUDIO_API_URL` in `app.py`.*

## 🏃‍♂️ Running the Application

### Step 1: Start the Backend
Open a terminal in the project root and run:
```bash
python app.py
```
*The server will start at `http://0.0.0.0:5000`.*

### Step 2: Start the Frontend
Open a new terminal in the `website` folder and run:
```bash
npm start
```
*The application will open in your browser at `http://localhost:3000`.*

## 📡 API Endpoints

### `POST /predict`
Generates a price prediction.
*   **Body**: `{ "stock_name": "Infosys", "future_date": "2024-12-10", "algorithm_name": "LSTM" }`
*   **Returns**: Predicted price, trend data for charting.

### `POST /get-ai-insights`
Generates text analysis using the local LLM.
*   **Body**: `{ "stock_name": "Infosys", "trend_data": {...} }`
*   **Returns**: `{ "insights": "Based on the RSI..." }`

### `GET /historical-data`
Fetches historical OHLCV, RSI, and SMA data for charting.
*   **Query**: `?stock_name=Infosys`

## ⚠️ Disclaimer
This tool is for **educational and research purposes only**. The "Long-Range" forecasts are highly speculative. Do not use this application for real financial trading.
