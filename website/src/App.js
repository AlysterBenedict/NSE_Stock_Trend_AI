import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { createChart, ColorType } from 'lightweight-charts';
import { format, parseISO, addBusinessDays } from 'date-fns';
import { auth, onAuthStateChanged, signOut } from './firebase'; // Import Firebase auth state listener
import Login from './Login';   // Import Login component
import Signup from './Signup'; // Import Signup component
import './App.css'; // Make sure this line exists
import AdvancedChart from './AdvancedChart'; // Import the new chart component
import FiftyTwoWeekCharts from './FiftyTwoWeekCharts'; // Import the new 52-week chart component
import AiChatBot from './AiChatBot'; // Import the new ChatBot component
import FinancialTeacher from './FinancialTeacher'; // Import the new Financial Teacher component
import SearchableSelect from './SearchableSelect'; // Import the new searchable select component
import PortfolioCalculator from './PortfolioCalculator';
import MarketDashboard from './MarketDashboard';
import PositionsDashboard from './PositionsDashboard'; // Import the new Portfolio Calculator
import SentimentDashboard from './SentimentDashboard';
import InvestmentEngine from './InvestmentEngine'; // Import Investment Engine



function getNextBusinessDay() {
  let tomorrow = addBusinessDays(new Date(), 1);
  return format(tomorrow, 'yyyy-MM-dd');
}
function isValidLineData(d) {
  // Added check for object type
  return typeof d === 'object' && d !== null && d.time && d.value !== null && typeof d.value === 'number' && isFinite(d.value);
}

// ----- React Components -----

/**
 * Header Component (Updated with Logout)
 */
const Header = ({ user, onLogout }) => (
  <header className="app-header">
    <div className="header-content"> {/* Added wrapper for layout */}
      <div className="logo-section" style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
        <h1>NeuroStock</h1>
      </div>
      {user && (
        <div className="user-info">
          {/* Display email if available */}
          <span>Welcome, {user.email || 'User'}</span>
          <button onClick={onLogout} className="logout-button">Logout</button>
        </div>
      )}
    </div>
    {/* Show different subtitle based on login state */}
    {!user && <p>Please login or sign up to use the predictor.</p>}
    {user && <p>Advanced AI-Driven Market Predictions & Analysis</p>}
  </header>
);

/**
 * Control Panel Component (Copied from your stable version)
 */
const ControlPanel = ({ onPredict, isLoading, stock, setStock, algorithm, setAlgorithm, date, setDate }) => {
  const handleSubmit = () => { onPredict(stock, algorithm, date); };
  return (
    <div className="card control-panel">
      <h2>Prediction Parameters</h2>
      {/* Stock Select */}
      <div className="form-group stock-select-group">
        <label htmlFor="stock-select">Select Stock:</label>
        <SearchableSelect
          options={[
            { value: "Infosys", label: "Infosys (INFY.NS)" },
            { value: "Yes Bank", label: "Yes Bank (YESBANK.NS)" },
            { value: "TCS", label: "TCS (TCS.NS)" },
            { value: "HDFC Bank", label: "HDFC Bank (HDFCBANK.NS)" },
            { value: "ITC", label: "ITC (ITC.NS)" },
            { value: "Power Grid Corp", label: "Power Grid Corp (POWERGRID.NS)" },
            { value: "Bajaj Finserv", label: "Bajaj Finserv (BAJAJFINSV.NS)" },
            { value: "Adani Ports", label: "Adani Ports (ADANIPORTS.NS)" },
            { value: "Tata Steel", label: "Tata Steel (TATASTEEL.NS)" },
            { value: "Asian Paints", label: "Asian Paints (ASIANPAINT.NS)" },
            { value: "JSW Steel", label: "JSW Steel (JSWSTEEL.NS)" },
            { value: "Bajaj Auto", label: "Bajaj Auto (BAJAJ-AUTO.NS)" },
            { value: "Lupin", label: "Lupin (LUPIN.NS)" },
            { value: "Hindalco", label: "Hindalco (HINDALCO.NS)" },
            { value: "LTIMindtree", label: "LTIMindtree (LTIM.NS)" },
            { value: "Grasim", label: "Grasim (GRASIM.NS)" },
            { value: "Cipla", label: "Cipla (CIPLA.NS)" },
            { value: "Tech Mahindra", label: "Tech Mahindra (TECHM.NS)" },
            { value: "Wipro", label: "Wipro (WIPRO.NS)" },
            { value: "Nestle India", label: "Nestle India (NESTLEIND.NS)" },
            { value: "Adani Green", label: "Adani Green (ADANIGREEN.NS)" },
            { value: "BEL", label: "BEL (BEL.NS)" },
            { value: "Varun Beverages", label: "Varun Beverages (VBL.NS)" },
            { value: "IndusInd Bank", label: "IndusInd Bank (INDUSINDBK.NS)" },
            { value: "Tata Consumer", label: "Tata Consumer (TATACONSUM.NS)" },
            { value: "Zomato", label: "Zomato (ZOMATO.NS)" },
            { value: "Britannia", label: "Britannia (BRITANNIA.NS)" },
            { value: "SBI Life", label: "SBI Life (SBILIFE.NS)" },
            { value: "HAL", label: "HAL (HAL.NS)" },
            { value: "Trent", label: "Trent (TRENT.NS)" }
          ]}
          value={stock}
          onChange={setStock}
          placeholder="Search or select a stock..."
        />
      </div>
      {/* Algorithm Select */}
      <div className="form-group algo-select-group">
        <label htmlFor="algo-select">Select Algorithm:</label>
        <SearchableSelect
          options={[
            { value: "LSTM", label: "LSTM" },
            { value: "XGBoost", label: "XGBoost" },
            { value: "RandomForest", label: "Random Forest" },
            { value: "DecisionTree", label: "Decision Tree" },
            { value: "SVR", label: "SVR" },
            { value: "LinearRegression", label: "Linear Regression" }
          ]}
          value={algorithm}
          onChange={setAlgorithm}
          placeholder="Search or select an algorithm..."
        />
      </div>
      {/* Date Select */}
      <div className="form-group">
        <label htmlFor="future-date">Select Future Date:</label>
        <input type="date" id="future-date" value={date} onChange={(e) => setDate(e.target.value)} min={format(new Date(), 'yyyy-MM-dd')} />
      </div>
      {/* Predict Button */}
      <button id="predict-btn" className="cta-button" onClick={handleSubmit} disabled={isLoading}>
        {isLoading && <svg className="button-spinner mini-spinner" viewBox="0 0 50 50"><circle className="path" cx="25" cy="25" r="20" fill="none" strokeWidth="5"></circle></svg>}
        <span>{isLoading ? 'Predicting...' : 'Predict Trend'}</span>
      </button>
    </div>
  );
};

/**
 * PredictionChart Component (Copied from your stable version)
 */
const PredictionChart = ({ trendData, predictedPrice }) => {
  const chartRef = useRef(); const chartInstance = useRef(); // Renamed to avoid conflict
  useEffect(() => {
    if (!trendData?.dates || !trendData.prices || !chartRef.current) return;
    const historyCutoff = trendData.history_cutoff;
    if (historyCutoff <= 0 || historyCutoff > trendData.prices.length) return;
    const lastActualPrice = parseFloat(trendData.prices[historyCutoff - 1]);
    const predictedPriceNum = parseFloat(predictedPrice);
    let priceClass = predictedPriceNum > lastActualPrice ? 'price-increase' : 'price-decrease';

    chartInstance.current = createChart(chartRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: '#FFFFFF' }, // Pure White
        textColor: '#2D3748', // Dark text
      },
      grid: {
        vertLines: { color: 'rgba(0, 0, 0, 0.05)' },
        horzLines: { color: 'rgba(0, 0, 0, 0.05)' },
      },
      width: chartRef.current.clientWidth,
      height: 300,
      timeScale: {
        borderColor: 'rgba(0, 0, 0, 0.1)',
        timeVisible: true,
        secondsVisible: false,
      },
      rightPriceScale: {
        borderColor: 'rgba(0, 0, 0, 0.1)',
      },
    });
    try {
      const historyDataForChart = trendData.prices.slice(0, historyCutoff).map((p, i) => ({ time: format(parseISO(trendData.dates[i]), 'yyyy-MM-dd'), value: p })).filter(isValidLineData);
      if (historyDataForChart.length > 0) { const lineSeries = chartInstance.current.addLineSeries({ color: '#2962FF', lineWidth: 2, title: 'Historical' }); lineSeries.setData(historyDataForChart); }
      const forecastDataForChart = []; const lastHistTimeStr = trendData.dates[historyCutoff - 1]; const lastHistValue = trendData.prices[historyCutoff - 1];
      if (lastHistTimeStr && lastHistValue !== null && isFinite(lastHistValue)) { forecastDataForChart.push({ time: format(parseISO(lastHistTimeStr), 'yyyy-MM-dd'), value: lastHistValue }); }
      for (let i = historyCutoff; i < trendData.dates.length; i++) { const futureTimeStr = trendData.dates[i]; const futureValue = trendData.prices[i]; if (futureTimeStr && futureValue !== null && isFinite(futureValue)) { forecastDataForChart.push({ time: format(parseISO(futureTimeStr), 'yyyy-MM-dd'), value: futureValue }); } }
      if (forecastDataForChart.length > 0) { const forecastSeries = chartInstance.current.addLineSeries({ color: priceClass === 'price-increase' ? '#00C853' : '#D50000', lineWidth: 2, lineStyle: 2, title: 'Forecast' }); forecastSeries.setData(forecastDataForChart); }
      chartInstance.current.timeScale().fitContent();
    } catch (chartError) { console.error("Chart error:", chartError); }
    const handleResize = () => { if (chartInstance.current) chartInstance.current.applyOptions({ width: chartRef.current.clientWidth }); }; window.addEventListener('resize', handleResize);
    return () => {
      window.removeEventListener('resize', handleResize);
      if (chartInstance.current) {
        try {
          chartInstance.current.remove();
        } catch (e) {
          console.warn("Chart disposal error:", e);
        }
        chartInstance.current = null;
      }
    };
  }, [trendData, predictedPrice]);
  return (<div className="modal-chart-container"> <h4>Prediction Trend (Last 100 Days + Forecast)</h4> <div className="chart-wrapper" ref={chartRef}></div> </div>);
};




/**
 * ResultDashboard Component (UPDATED with AI Insights)
 */
const ResultDashboard = ({ result, error }) => {
  if (error) return (<div className="card error-message result-dashboard"><h4>Request Error</h4><p>{error}</p></div>);

  // Show prompt to run prediction if no data
  if (!result.trend_data || !result.trend_data.prices) {
    return (<div className="card loading-indicator result-dashboard"><p>Run prediction to see results.</p></div>);
  }

  const { predicted_price, stock_name, future_date, algorithm_name, prediction_type, warning, trend_data } = result;
  const historyCutoff = trend_data.history_cutoff; let priceClass = '';
  if (historyCutoff > 0 && historyCutoff <= trend_data.prices.length) { const lastActualPrice = parseFloat(trend_data.prices[historyCutoff - 1]); const predictedPriceNum = parseFloat(predicted_price); if (predictedPriceNum > lastActualPrice) priceClass = 'price-increase'; if (predictedPriceNum < lastActualPrice) priceClass = 'price-decrease'; }

  return (
    <div className="card result-dashboard">
      <div className="algorithm-header">{algorithm_name}</div>
      <h3 className="modal-title">{prediction_type}</h3>
      {warning && (<div className="warning-box"> <strong>Warning:</strong> {warning} </div>)}

      <div className="modal-body">
        <div className="modal-summary">
          <p>Predicts <strong>{stock_name}</strong> will close at:</p>
          <div className="price-display"><span className="currency">₹</span><span className={`price-value ${priceClass}`}>{predicted_price}</span></div>
          <div className="result-details"><p><strong>On Date:</strong> <span>{future_date}</span></p></div>
        </div>
        <div className="chart-display-area">
          <PredictionChart key={stock_name + future_date} trendData={trend_data} predictedPrice={predicted_price} />
        </div>
      </div>
    </div>
  );
}


/**
 * ComparisonControlPanel Component
 */
const ComparisonControlPanel = ({ onPredict, isLoading, stock, setStock, algo1, setAlgo1, algo2, setAlgo2, date, setDate }) => {
  const handleSubmit = () => { onPredict(stock, algo1, algo2, date); };
  return (
    <div className="card control-panel">
      <h2>Comparison Parameters</h2>
      <div className="form-group stock-select-group">
        <label>Select Stock:</label>
        <SearchableSelect
          options={[
            { value: "Infosys", label: "Infosys (INFY.NS)" },
            { value: "Yes Bank", label: "Yes Bank (YESBANK.NS)" },
            { value: "TCS", label: "TCS (TCS.NS)" },
            { value: "HDFC Bank", label: "HDFC Bank (HDFCBANK.NS)" },
            { value: "ITC", label: "ITC (ITC.NS)" },
            { value: "Power Grid Corp", label: "Power Grid Corp (POWERGRID.NS)" },
            { value: "Bajaj Finserv", label: "Bajaj Finserv (BAJAJFINSV.NS)" },
            { value: "Adani Ports", label: "Adani Ports (ADANIPORTS.NS)" },
            { value: "Tata Steel", label: "Tata Steel (TATASTEEL.NS)" },
            { value: "Asian Paints", label: "Asian Paints (ASIANPAINT.NS)" },
            { value: "JSW Steel", label: "JSW Steel (JSWSTEEL.NS)" },
            { value: "Bajaj Auto", label: "Bajaj Auto (BAJAJ-AUTO.NS)" },
            { value: "Lupin", label: "Lupin (LUPIN.NS)" },
            { value: "Hindalco", label: "Hindalco (HINDALCO.NS)" },
            { value: "LTIMindtree", label: "LTIMindtree (LTIM.NS)" },
            { value: "Grasim", label: "Grasim (GRASIM.NS)" },
            { value: "Cipla", label: "Cipla (CIPLA.NS)" },
            { value: "Tech Mahindra", label: "Tech Mahindra (TECHM.NS)" },
            { value: "Wipro", label: "Wipro (WIPRO.NS)" },
            { value: "Nestle India", label: "Nestle India (NESTLEIND.NS)" },
            { value: "Adani Green", label: "Adani Green (ADANIGREEN.NS)" },
            { value: "BEL", label: "BEL (BEL.NS)" },
            { value: "Varun Beverages", label: "Varun Beverages (VBL.NS)" },
            { value: "IndusInd Bank", label: "IndusInd Bank (INDUSINDBK.NS)" },
            { value: "Tata Consumer", label: "Tata Consumer (TATACONSUM.NS)" },
            { value: "Zomato", label: "Zomato (ZOMATO.NS)" },
            { value: "Britannia", label: "Britannia (BRITANNIA.NS)" },
            { value: "SBI Life", label: "SBI Life (SBILIFE.NS)" },
            { value: "HAL", label: "HAL (HAL.NS)" },
            { value: "Trent", label: "Trent (TRENT.NS)" }
          ]}
          value={stock} onChange={setStock} placeholder="Select stock..."
        />
      </div>
      <div className="dashboard-row" style={{ gap: '20px' }}>
        <div className="form-group" style={{ flex: 1 }}>
          <label>Algorithm 1:</label>
          <SearchableSelect
            options={[
              { value: "LSTM", label: "LSTM" },
              { value: "XGBoost", label: "XGBoost" },
              { value: "RandomForest", label: "Random Forest" },
              { value: "DecisionTree", label: "Decision Tree" },
              { value: "SVR", label: "SVR" },
              { value: "LinearRegression", label: "Linear Regression" }
            ]}
            value={algo1} onChange={setAlgo1} placeholder="Select Algo 1..."
          />
        </div>
        <div className="form-group" style={{ flex: 1 }}>
          <label>Algorithm 2:</label>
          <SearchableSelect
            options={[
              { value: "LSTM", label: "LSTM" },
              { value: "XGBoost", label: "XGBoost" },
              { value: "RandomForest", label: "Random Forest" },
              { value: "DecisionTree", label: "Decision Tree" },
              { value: "SVR", label: "SVR" },
              { value: "LinearRegression", label: "Linear Regression" }
            ]}
            value={algo2} onChange={setAlgo2} placeholder="Select Algo 2..."
          />
        </div>
      </div>
      <div className="form-group">
        <label>Select Future Date:</label>
        <input type="date" value={date} onChange={(e) => setDate(e.target.value)} min={format(new Date(), 'yyyy-MM-dd')} />
      </div>
      <button className="cta-button" onClick={handleSubmit} disabled={isLoading}>
        {isLoading && <svg className="button-spinner mini-spinner" viewBox="0 0 50 50"><circle className="path" cx="25" cy="25" r="20" fill="none" strokeWidth="5"></circle></svg>}
        <span>{isLoading ? 'Compare Models' : 'Run Comparison'}</span>
      </button>
    </div>
  );
};

/**
 * Main App Component (Updated for Authentication & AI Insights)
 */
function App() {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [predictionResult, setPredictionResult] = useState({});
  const [user, setUser] = useState(null); // State to hold the logged-in user
  const [authForm, setAuthForm] = useState('login'); // 'login' or 'signup'
  const [authLoading, setAuthLoading] = useState(true); // Loading state for initial auth check

  // --- Tabs State ---
  const [activeTab, setActiveTab] = useState('market');

  // --- Lifted State for ControlPanel (Home) ---
  const [stock, setStock] = useState('Infosys');
  const [algorithm, setAlgorithm] = useState('LSTM');
  const [date, setDate] = useState(getNextBusinessDay());

  // --- Comparison State ---
  const [compStock, setCompStock] = useState('Infosys');
  const [compAlgo1, setCompAlgo1] = useState('LSTM');
  const [compAlgo2, setCompAlgo2] = useState('XGBoost');
  const [compDate, setCompDate] = useState(getNextBusinessDay());
  const [compResult1, setCompResult1] = useState({});
  const [compResult2, setCompResult2] = useState({});
  const [compLoading, setCompLoading] = useState(false);
  const [compError, setCompError] = useState(null);

  // --- 52 Week State ---
  const [fiftyTwoWeekStock, setFiftyTwoWeekStock] = useState('Infosys');

  // --- NEW AI State ---
  // Removed simple aiInsights state, now handled by ChatBot component





  // Effect to listen for authentication state changes
  useEffect(() => {
    setAuthLoading(true); // Start loading when checking auth state
    const unsubscribe = onAuthStateChanged(auth, (currentUser) => {
      console.log("Auth State Changed:", currentUser); // Debugging log
      setUser(currentUser);
      setAuthLoading(false); // Auth check complete
      if (currentUser) {
        // Reset prediction state only if logging IN
        setPredictionResult({});
        setError(null);
        setAuthForm('login'); // Default back to login view if logged out
      }
    });
    // Cleanup subscription on unmount
    return () => unsubscribe();
  }, []); // Run only once on mount

  // Logout handler
  const handleLogout = async () => {
    try {
      await signOut(auth);
      // User state will be updated by onAuthStateChanged listener
      console.log("User logged out");
      // Clear all data on logout
      // Clear all data on logout
      setPredictionResult({});
      setError(null);
    } catch (err) {
      console.error("Logout failed:", err);
      setError("Logout failed. Please try again."); // Show user feedback
    }
  };

  // Prediction handler (UPDATED to clear AI state)
  const handlePrediction = async (stock, algorithm, date) => {
    setIsLoading(true);
    setError(null);
    setPredictionResult({});

    try {
      const response = await axios.post('/predict', { stock_name: stock, future_date: date, algorithm_name: algorithm });
      setPredictionResult(response.data);
    } catch (err) {
      console.error("Prediction failed:", err); setError(err.response?.data?.error || "Prediction request failed."); setPredictionResult({});
    } finally { setIsLoading(false); }
  };

  // Comparison Prediction Handler
  const handleComparisonPredict = async (stock, algo1, algo2, date) => {
    setCompLoading(true);
    setCompError(null);
    setCompResult1({});
    setCompResult2({});

    try {
      // Optimized: Send single request with both algorithms to avoid concurrency issues
      const response = await axios.post('/predict', {
        stock_name: stock,
        future_date: date,
        algorithm_name: [algo1, algo2]
      });

      const results = response.data;

      // Handle Result 1
      if (results[algo1].error) {
        // If specific algo failed, set empty result but maybe don't fail whole request?
        // For now, let's just set the result as is, the dashboard handles errors if passed?
        // Actually ResultDashboard expects { predicted_price, ... } or displays error if passed.
        // Let's pass the result object which might contain 'error' key if backend set it.
        // But ResultDashboard props are `result` and `error`.
        setCompResult1({});
        // If both fail, we set compError. If one fails, we might want to show partial.
        // But for simplicity, if one has error, we might want to show it in the dashboard card?
        // The current ResultDashboard component takes `error` prop.
        // Let's assume results[algo1] is the full object.
      }
      setCompResult1(results[algo1]);

      // Handle Result 2
      setCompResult2(results[algo2]);

      // Check for global errors or if both failed
      if (results[algo1].error && results[algo2].error) {
        setCompError(`Both algorithms failed: ${results[algo1].error} / ${results[algo2].error}`);
      }

    } catch (err) {
      console.error("Comparison failed:", err);
      setCompError(err.response?.data?.error || "Comparison request failed. Please check inputs or try again.");
    } finally {
      setCompLoading(false);
    }
  };

  // --- Handler for AI Insights (Chat) ---
  const handleGetAiInsights = async (userQuestion) => {
    // Use comparison data if on comparison tab? For now, stick to home data or active context
    // Let's default to the main prediction result for now, or the first comparison result if active
    let targetResult = predictionResult;
    if (activeTab === 'comparison' && compResult1.trend_data) {
      targetResult = compResult1; // Default to first algo for now
    }

    // Allow chat even without prediction result (feature request)
    const stockName = targetResult?.stock_name || null;
    const trendData = targetResult?.trend_data || null;

    try {
      const response = await axios.post('http://127.0.0.1:5000/get-ai-insights', {
        stock_name: stockName,
        trend_data: trendData,
        user_question: userQuestion // Pass the custom question
      });

      if (response.data.insights) {
        return response.data.insights;
      } else {
        throw new Error("No insights returned");
      }
    } catch (err) {
      console.error("AI Error:", err);
      throw err; // Re-throw for ChatBot to handle
    }
  };


  // Render loading state while checking auth
  if (authLoading) {
    return <div className="loading-indicator fullscreen-loader"><div className="spinner"></div><p>Checking authentication...</p></div>;
  }

  return (
    <div className="app-container">
      {/* Pass user and logout handler to Header */}
      <Header user={user} onLogout={handleLogout} />

      {/* Conditionally render Auth forms or Dashboard */}
      {!user ? (
        // Pass the function to switch forms
        authForm === 'login' ? (
          <Login toggleForm={() => setAuthForm('signup')} />
        ) : (
          <Signup toggleForm={() => setAuthForm('login')} />
        )
      ) : (
        // Render the main dashboard content only when logged in
        <>
          {/* Tab Navigation */}
          <div className="tab-menu">
            <div className={`tab-item ${activeTab === 'market' ? 'active' : ''}`} onClick={() => setActiveTab('market')}>
              Market
            </div>
            <div className={`tab-item ${activeTab === 'investment' ? 'active' : ''}`} onClick={() => setActiveTab('investment')}>
              Invest Engine  {/* New Tab */}
            </div>
            <div className={`tab-item ${activeTab === 'home' ? 'active' : ''}`} onClick={() => setActiveTab('home')}>
              Prediction
            </div>
            <div className={`tab-item ${activeTab === 'comparison' ? 'active' : ''}`} onClick={() => setActiveTab('comparison')}>
              Comparison
            </div>
            <div className={`tab-item ${activeTab === 'portfolio' ? 'active' : ''}`} onClick={() => setActiveTab('portfolio')}>
              Portfolio
            </div>
            <div className={`tab-item ${activeTab === 'positions' ? 'active' : ''}`} onClick={() => setActiveTab('positions')}>
              Positions
            </div>
            <div className={`tab-item ${activeTab === 'fiftyTwoWeek' ? 'active' : ''}`} onClick={() => setActiveTab('fiftyTwoWeek')}>
              52 Week
            </div>
            <div className={`tab-item ${activeTab === 'sentiment' ? 'active' : ''}`} onClick={() => setActiveTab('sentiment')}>
              Sentiment
            </div>
            <div className={`tab-item ${activeTab === 'education' ? 'active' : ''}`} onClick={() => setActiveTab('education')}>
              Education
            </div>
          </div>
          <main className="content-area">

            {/* Tab Navigation */}


            {/* HOME TAB CONTENT */}
            {activeTab === 'home' && (
              <>
                <div className="dashboard-row">
                  <ControlPanel
                    onPredict={handlePrediction}
                    isLoading={isLoading}
                    stock={stock} setStock={setStock}
                    algorithm={algorithm} setAlgorithm={setAlgorithm}
                    date={date} setDate={setDate}
                  />
                  <ResultDashboard
                    result={predictionResult}
                    error={error}
                  />
                </div>

                <div className="advanced-chart-section">
                  <AdvancedChart stockName={stock} />
                </div>
              </>
            )}

            {/* INVESTMENT ENGINE TAB CONTENT */}
            {activeTab === 'investment' && (
              <InvestmentEngine />
            )}

            {/* COMPARISON TAB CONTENT */}

            {activeTab === 'comparison' && (
              <div className="comparison-container">
                <ComparisonControlPanel
                  onPredict={handleComparisonPredict}
                  isLoading={compLoading}
                  stock={compStock} setStock={setCompStock}
                  algo1={compAlgo1} setAlgo1={setCompAlgo1}
                  algo2={compAlgo2} setAlgo2={setCompAlgo2}
                  date={compDate} setDate={setCompDate}
                />

                {compError && <div className="card error-message"><p>{compError}</p></div>}

                <div className="comparison-results-row">
                  <ResultDashboard result={compResult1} error={null} />
                  <ResultDashboard result={compResult2} error={null} />
                </div>
              </div>
            )}

            {/* 52 WEEK TAB CONTENT */}
            {activeTab === 'fiftyTwoWeek' && (
              <div className="fifty-two-week-container">
                <div className="card control-panel">
                  <h2>Select Stock for 52-Week Analysis</h2>
                  <div className="form-group">
                    <label>Select Stock:</label>
                    <SearchableSelect
                      options={[
                        { value: "Infosys", label: "Infosys (INFY.NS)" },
                        { value: "Yes Bank", label: "Yes Bank (YESBANK.NS)" },
                        { value: "TCS", label: "TCS (TCS.NS)" },
                        { value: "HDFC Bank", label: "HDFC Bank (HDFCBANK.NS)" },
                        { value: "ITC", label: "ITC (ITC.NS)" },
                        { value: "Power Grid Corp", label: "Power Grid Corp (POWERGRID.NS)" },
                        { value: "Bajaj Finserv", label: "Bajaj Finserv (BAJAJFINSV.NS)" },
                        { value: "Adani Ports", label: "Adani Ports (ADANIPORTS.NS)" },
                        { value: "Tata Steel", label: "Tata Steel (TATASTEEL.NS)" },
                        { value: "Asian Paints", label: "Asian Paints (ASIANPAINT.NS)" },
                        { value: "JSW Steel", label: "JSW Steel (JSWSTEEL.NS)" },
                        { value: "Bajaj Auto", label: "Bajaj Auto (BAJAJ-AUTO.NS)" },
                        { value: "Lupin", label: "Lupin (LUPIN.NS)" },
                        { value: "Hindalco", label: "Hindalco (HINDALCO.NS)" },
                        { value: "LTIMindtree", label: "LTIMindtree (LTIM.NS)" },
                        { value: "Grasim", label: "Grasim (GRASIM.NS)" },
                        { value: "Cipla", label: "Cipla (CIPLA.NS)" },
                        { value: "Tech Mahindra", label: "Tech Mahindra (TECHM.NS)" },
                        { value: "Wipro", label: "Wipro (WIPRO.NS)" },
                        { value: "Nestle India", label: "Nestle India (NESTLEIND.NS)" },
                        { value: "Adani Green", label: "Adani Green (ADANIGREEN.NS)" },
                        { value: "BEL", label: "BEL (BEL.NS)" },
                        { value: "Varun Beverages", label: "Varun Beverages (VBL.NS)" },
                        { value: "IndusInd Bank", label: "IndusInd Bank (INDUSINDBK.NS)" },
                        { value: "Tata Consumer", label: "Tata Consumer (TATACONSUM.NS)" },
                        { value: "Zomato", label: "Zomato (ZOMATO.NS)" },
                        { value: "Britannia", label: "Britannia (BRITANNIA.NS)" },
                        { value: "SBI Life", label: "SBI Life (SBILIFE.NS)" },
                        { value: "HAL", label: "HAL (HAL.NS)" },
                        { value: "Trent", label: "Trent (TRENT.NS)" }
                      ]}
                      value={fiftyTwoWeekStock}
                      onChange={setFiftyTwoWeekStock}
                      placeholder="Select stock..."
                    />
                  </div>
                </div>
                <FiftyTwoWeekCharts stockName={fiftyTwoWeekStock} />
              </div>
            )}

            {activeTab === 'sentiment' && (
              <SentimentDashboard />
            )}

            {/* EDUCATION TAB CONTENT - Persist state by hiding instead of unmounting */}
            <div style={{ display: activeTab === 'education' ? 'block' : 'none', width: '100%', height: '100%' }}>
              <FinancialTeacher />
            </div>

            {/* MARKET TAB CONTENT */}
            {activeTab === 'market' && (
              <MarketDashboard />
            )}

            {/* POSITIONS TAB CONTENT */}
            {activeTab === 'positions' && (
              <PositionsDashboard />
            )}

            {/* PORTFOLIO TAB CONTENT */}
            {activeTab === 'portfolio' && (
              <PortfolioCalculator />
            )}

            {/* Show standard ChatBot on all tabs EXCEPT Education */}
            {activeTab !== 'education' && (
              <AiChatBot
                onGetInsights={handleGetAiInsights}
                stockName={activeTab === 'home' ? stock : (activeTab === 'comparison' ? compStock : 'Market Assistant')}
              />
            )}

          </main>
        </>
      )}

      <footer className="app-footer">
        <p>© {new Date().getFullYear()} Stock Price Predictor. All rights reserved.</p>
        {!user && (
          <div className="footer-features">
            <div className="feature-item">
              <span className="feature-icon">📈</span>
              <span>Advanced ML Predictions</span>
            </div>
            <div className="feature-separator">•</div>
            <div className="feature-item">
              <span className="feature-icon">📊</span>
              <span>Interactive Technical Charts</span>
            </div>
            <div className="feature-separator">•</div>
            <div className="feature-item">
              <span className="feature-icon">🤖</span>
              <span>AI-Powered Insights</span>
            </div>
          </div>
        )}
      </footer>
    </div>
  );
}

export default App;
