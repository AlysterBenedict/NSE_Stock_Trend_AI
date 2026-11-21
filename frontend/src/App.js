import React, { useEffect, useRef, useState } from 'react';
import { Icon, Loader } from './components/common/ui';
import AppFooter from './components/layout/AppFooter';
import AppHeader from './components/layout/AppHeader';
import AiMarketChatBot from './features/assistant/AiMarketChatBot';
import FinancialEducationTutor from './features/assistant/FinancialEducationTutor';
import LoginPage from './features/auth/LoginPage';
import SignupPage from './features/auth/SignupPage';
import AiInvestmentEngine from './features/investment/AiInvestmentEngine';
import PortfolioReturnCalculator from './features/investment/PortfolioReturnCalculator';
import MarketOverviewDashboard from './features/market/MarketOverviewDashboard';
import OpenPositionsDashboard from './features/market/OpenPositionsDashboard';
import FiftyTwoWeekDashboard from './features/prediction/FiftyTwoWeekDashboard';
import ModelComparisonDashboard from './features/prediction/ModelComparisonDashboard';
import PredictionDashboard from './features/prediction/PredictionDashboard';
import NewsSentimentDashboard from './features/sentiment/NewsSentimentDashboard';
import { getAiInsights, predictStockPrice } from './services/apiClient';
import { auth, onAuthStateChanged, signOut } from './services/firebaseAuth';
import './styles/App.css';
import { getNextBusinessDay, isCancelledError } from './utils/formatters';

const TAB_GROUPS = [
  {
    label: 'Markets',
    tabs: [
      { id: 'market', label: 'Market', icon: 'storefront' },
      { id: 'positions', label: 'Positions', icon: 'leaderboard' },
      { id: 'sentiment', label: 'Sentiment', icon: 'newspaper' },
    ],
  },
  {
    label: 'Forecasting',
    tabs: [
      { id: 'home', label: 'Prediction', icon: 'query_stats' },
      { id: 'comparison', label: 'Comparison', icon: 'compare_arrows' },
      { id: 'fiftyTwoWeek', label: '52 Week', icon: 'candlestick_chart' },
    ],
  },
  {
    label: 'Portfolio',
    tabs: [
      { id: 'investment', label: 'Invest Engine', icon: 'auto_awesome' },
      { id: 'portfolio', label: 'Portfolio', icon: 'account_balance_wallet' },
    ],
  },
  {
    label: 'Learn',
    tabs: [{ id: 'education', label: 'Education', icon: 'school' }],
  },
];

function App() {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [predictionResult, setPredictionResult] = useState({});
  const [user, setUser] = useState(null);
  const [authForm, setAuthForm] = useState('login');
  const [authLoading, setAuthLoading] = useState(true);
  const [activeTab, setActiveTab] = useState('market');
  const [tabResetKeys, setTabResetKeys] = useState({
    market: 0,
    positions: 0,
    sentiment: 0,
    home: 0,
    comparison: 0,
    fiftyTwoWeek: 0,
    investment: 0,
    portfolio: 0,
    education: 0,
  });

  // Single prediction state
  const [stock, setStock] = useState('');
  const [algorithm, setAlgorithm] = useState('Ensemble');
  const [date, setDate] = useState(getNextBusinessDay());

  // Comparison state
  const [compStock, setCompStock] = useState('');
  const [compAlgo1, setCompAlgo1] = useState('Ensemble');
  const [compAlgo2, setCompAlgo2] = useState('AttentionNet');
  const [compDate, setCompDate] = useState(getNextBusinessDay());
  const [compResult1, setCompResult1] = useState({});
  const [compResult2, setCompResult2] = useState({});
  const [compLoading, setCompLoading] = useState(false);
  const [compError, setCompError] = useState(null);

  // 52-week analysis state
  const [fiftyTwoWeekStock, setFiftyTwoWeekStock] = useState('');
  const predAbortRef = useRef(null);
  const compAbortRef = useRef(null);

  useEffect(() => {
    setAuthLoading(true);
    const unsubscribe = onAuthStateChanged(auth, (currentUser) => {
      setUser(currentUser);
      setAuthLoading(false);
      if (currentUser) {
        setPredictionResult({});
        setError(null);
        setAuthForm('login');
      }
    });
    return () => unsubscribe();
  }, []);

  const handleLogout = async () => {
    try {
      await signOut(auth);
      setPredictionResult({});
      setError(null);
    } catch (err) {
      console.error('Logout failed:', err);
      setError('Logout failed. Please try again.');
    }
  };

  const handlePrediction = async (selectedStock, selectedAlgo, selectedDate) => {
    predAbortRef.current?.abort();
    const controller = new AbortController();
    predAbortRef.current = controller;

    setIsLoading(true);
    setError(null);
    setPredictionResult({});
    try {
      const data = await predictStockPrice(
        {
          stock_name: selectedStock,
          future_date: selectedDate,
          algorithm_name: selectedAlgo,
        },
        controller.signal
      );
      if (controller.signal.aborted) return;
      setPredictionResult(data);
    } catch (err) {
      if (controller.signal.aborted || isCancelledError(err)) return;
      console.error('Prediction failed:', err);
      setError(err.response?.data?.error || 'Prediction request failed.');
      setPredictionResult({});
    } finally {
      if (predAbortRef.current === controller) {
        setIsLoading(false);
      }
    }
  };

  const handleComparisonPredict = async (selectedStock, algo1, algo2, selectedDate) => {
    compAbortRef.current?.abort();
    const controller = new AbortController();
    compAbortRef.current = controller;

    setCompLoading(true);
    setCompError(null);
    setCompResult1({});
    setCompResult2({});
    try {
      const results = await predictStockPrice(
        {
          stock_name: selectedStock,
          future_date: selectedDate,
          algorithm_name: [algo1, algo2],
        },
        controller.signal
      );
      if (controller.signal.aborted) return;
      setCompResult1(results[algo1].error ? {} : results[algo1]);
      setCompResult2(results[algo2].error ? {} : results[algo2]);
      if (results[algo1].error && results[algo2].error) {
        setCompError(`Both algorithms failed: ${results[algo1].error} / ${results[algo2].error}`);
      }
    } catch (err) {
      if (controller.signal.aborted || isCancelledError(err)) return;
      console.error('Comparison failed:', err);
      setCompError(
        err.response?.data?.error || 'Comparison request failed. Please check inputs or try again.'
      );
    } finally {
      if (compAbortRef.current === controller) {
        setCompLoading(false);
      }
    }
  };

  const handleGetAiInsights = async (userQuestion, signal) => {
    const targetResult =
      activeTab === 'comparison' && compResult1.trend_data ? compResult1 : predictionResult;
    const data = await getAiInsights(
      {
        stock_name: targetResult?.stock_name || null,
        trend_data: targetResult?.trend_data || null,
        user_question: userQuestion,
      },
      signal
    );
    if (!data.insights) throw new Error('No insights returned');
    return data.insights;
  };

  const handleResetCurrentTab = () => {
    setTabResetKeys((prev) => ({
      ...prev,
      [activeTab]: (prev[activeTab] || 0) + 1,
    }));
    if (activeTab === 'home') {
      predAbortRef.current?.abort();
      predAbortRef.current = null;
      setStock('');
      setAlgorithm('Ensemble');
      setDate(getNextBusinessDay());
      setPredictionResult({});
      setError(null);
      setIsLoading(false);
    } else if (activeTab === 'comparison') {
      compAbortRef.current?.abort();
      compAbortRef.current = null;
      setCompStock('');
      setCompAlgo1('Ensemble');
      setCompAlgo2('AttentionNet');
      setCompDate(getNextBusinessDay());
      setCompResult1({});
      setCompResult2({});
      setCompError(null);
      setCompLoading(false);
    } else if (activeTab === 'fiftyTwoWeek') {
      setFiftyTwoWeekStock('');
    }
  };

  if (authLoading) {
    return (
      <div className="app-shell">
        <div className="ambient-canvas" aria-hidden="true">
          <span className="ambient-blob ambient-blob--sage" />
          <span className="ambient-blob ambient-blob--rose" />
          <span className="ambient-blob ambient-blob--mist" />
        </div>
        <div className="fullscreen-loader">
          <Loader label="Checking session..." />
        </div>
      </div>
    );
  }

  return (
    <div className="app-shell">
      <div className="ambient-canvas" aria-hidden="true">
        <span className="ambient-blob ambient-blob--sage" />
        <span className="ambient-blob ambient-blob--rose" />
        <span className="ambient-blob ambient-blob--mist" />
      </div>

      <AppHeader
        user={user}
        onLogout={handleLogout}
        tabGroups={TAB_GROUPS}
        activeTab={activeTab}
        onTabChange={setActiveTab}
      />

      {!user ? (
        <main className="content-area content-area--auth">
          {authForm === 'login' ? (
            <LoginPage toggleForm={() => setAuthForm('signup')} />
          ) : (
            <SignupPage toggleForm={() => setAuthForm('login')} />
          )}
        </main>
      ) : (
        <main className="content-area">
          <div className="page-reset-bar">
            <button
              type="button"
              className="btn btn--secondary btn--sm"
              onClick={handleResetCurrentTab}
              title="Reset current page"
              aria-label="Reset current page"
            >
              <Icon name="restart_alt" size={15} />
              <span>Reset</span>
            </button>
          </div>

          <div key={`market-${tabResetKeys.market}`} style={{ display: activeTab === 'market' ? 'block' : 'none', width: '100%' }}>
            <MarketOverviewDashboard />
          </div>

          <div key={`positions-${tabResetKeys.positions}`} style={{ display: activeTab === 'positions' ? 'block' : 'none', width: '100%' }}>
            <OpenPositionsDashboard />
          </div>

          <div key={`home-${tabResetKeys.home}`} style={{ display: activeTab === 'home' ? 'block' : 'none', width: '100%' }}>
            <PredictionDashboard
              onPredict={handlePrediction}
              isLoading={isLoading}
              stock={stock}
              setStock={setStock}
              algorithm={algorithm}
              setAlgorithm={setAlgorithm}
              date={date}
              setDate={setDate}
              predictionResult={predictionResult}
              error={error}
            />
          </div>

          <div key={`comparison-${tabResetKeys.comparison}`} style={{ display: activeTab === 'comparison' ? 'block' : 'none', width: '100%' }}>
            <ModelComparisonDashboard
              onPredict={handleComparisonPredict}
              isLoading={compLoading}
              stock={compStock}
              setStock={setCompStock}
              algo1={compAlgo1}
              setAlgo1={setCompAlgo1}
              algo2={compAlgo2}
              setAlgo2={setCompAlgo2}
              date={compDate}
              setDate={setCompDate}
              result1={compResult1}
              result2={compResult2}
              error={compError}
            />
          </div>

          <div key={`fiftyTwoWeek-${tabResetKeys.fiftyTwoWeek}`} style={{ display: activeTab === 'fiftyTwoWeek' ? 'block' : 'none', width: '100%' }}>
            <FiftyTwoWeekDashboard stock={fiftyTwoWeekStock} setStock={setFiftyTwoWeekStock} />
          </div>

          <div key={`investment-${tabResetKeys.investment}`} style={{ display: activeTab === 'investment' ? 'block' : 'none', width: '100%' }}>
            <AiInvestmentEngine />
          </div>

          <div key={`portfolio-${tabResetKeys.portfolio}`} style={{ display: activeTab === 'portfolio' ? 'block' : 'none', width: '100%' }}>
            <PortfolioReturnCalculator />
          </div>

          <div key={`sentiment-${tabResetKeys.sentiment}`} style={{ display: activeTab === 'sentiment' ? 'block' : 'none', width: '100%' }}>
            <NewsSentimentDashboard />
          </div>

          <div key={`education-${tabResetKeys.education}`} style={{ display: activeTab === 'education' ? 'block' : 'none', width: '100%' }}>
            <FinancialEducationTutor />
          </div>
        </main>
      )}

      <AppFooter showFeatures={!user} />

      {user && activeTab !== 'education' && (
        <AiMarketChatBot
          onGetInsights={handleGetAiInsights}
          stockName={
            (activeTab === 'home' ? stock : activeTab === 'comparison' ? compStock : '') || 'Market Assistant'
          }
        />
      )}
    </div>
  );
}

export default App;
