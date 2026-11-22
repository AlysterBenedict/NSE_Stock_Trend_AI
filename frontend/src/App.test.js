import { fireEvent, render, screen } from '@testing-library/react';
import App from './App';
import { ThemeProvider } from './context/ThemeContext';
import {
  getAlphaRadar,
  getHistoricalData,
  getMarketOverview,
  optimizePortfolio,
  predictStockPrice,
  runBacktest,
} from './services/apiClient';
import { onAuthStateChanged } from './services/firebaseAuth';

jest.mock('react-markdown', () => ({ children }) => <div>{children}</div>);
jest.mock('remark-gfm', () => () => {});

jest.mock('lightweight-charts', () => ({
  ColorType: { Solid: 'solid' },
  CrosshairMode: { Normal: 0 },
  LineStyle: { Solid: 0, Dotted: 1, Dashed: 2, LargeDashed: 3 },
  createChart: () => ({
    addLineSeries: () => ({ setData: jest.fn(), applyOptions: jest.fn() }),
    addAreaSeries: () => ({ setData: jest.fn(), applyOptions: jest.fn() }),
    addCandlestickSeries: () => ({ setData: jest.fn(), applyOptions: jest.fn() }),
    addHistogramSeries: () => ({ setData: jest.fn(), applyOptions: jest.fn() }),
    timeScale: () => ({
      fitContent: jest.fn(),
      subscribeVisibleLogicalRangeChange: jest.fn(),
      setVisibleLogicalRange: jest.fn(),
    }),
    applyOptions: jest.fn(),
    remove: jest.fn(),
  }),
}));

jest.mock('./services/apiClient', () => ({
  predictStockPrice: jest.fn(),
  getHistoricalData: jest.fn(),
  getMarketOverview: jest.fn(),
  getStockMarketDetail: jest.fn(),
  getAlphaRadar: jest.fn(),
  getAiInsights: jest.fn(),
  getGeneralFinancialKnowledge: jest.fn(),
  getStockSentiment: jest.fn(),
  calculatePortfolioProjection: jest.fn(),
  optimizePortfolio: jest.fn(),
  runInvestmentEngine: jest.fn(),
  runBacktest: jest.fn(),
}));

jest.mock('./services/firebaseAuth', () => ({
  auth: {},
  onAuthStateChanged: jest.fn(),
  signOut: jest.fn(),
  signInWithEmailAndPassword: jest.fn(),
  createUserWithEmailAndPassword: jest.fn(),
}));

beforeAll(() => {
  global.ResizeObserver = class {
    observe() {}
    unobserve() {}
    disconnect() {}
  };
  Element.prototype.scrollIntoView = jest.fn();
});

beforeEach(() => {
  jest.clearAllMocks();
  window.localStorage.clear();
  document.documentElement.removeAttribute('data-theme');
  getMarketOverview.mockResolvedValue([
    { name: 'Infosys', symbol: 'INFY.NS', price: 1626.8, open: 1605, close: 1602, change: 24.8, pChange: 1.55 },
    { name: 'HDFC Bank', symbol: 'HDFCBANK.NS', price: 979.7, open: 984, close: 984, change: -4.3, pChange: -0.44 },
  ]);
  getHistoricalData.mockResolvedValue({
    ohlc: [{ time: '2025-10-01', open: 1600, high: 1630, low: 1595, close: 1625 }],
    sma: [{ time: '2025-10-01', value: 1610 }],
    volume: [{ time: '2025-10-01', value: 1000000 }],
  });
  getAlphaRadar.mockResolvedValue({
    requested_horizon: 21,
    active_horizon: 21,
    available_horizons: [1, 21],
    models_ready: true,
    as_of: '2026-11-20',
    macro_regime: {
      regime_label: 'Low-Vol Risk-On',
      tone: 'up',
      india_vix: 13.4,
      nifty_21d_pct: 2.1,
      bank_nifty_rel_5d_pct: 0.4,
      it_nifty_rel_5d_pct: 0.7,
      usdinr_5d_pct: -0.1,
      brent_5d_pct: -0.5,
    },
    sectors: [
      {
        sector: 'IT',
        stock_count: 8,
        avg_pred_ret_pct: 2.4,
        mom_21d_pct: 1.8,
        mom_63d_pct: 3.1,
        quadrant: 'Leading',
        top_pick: { name: 'Infosys', ticker: 'INFY.NS', predicted_return_pct: 3.2 },
      },
    ],
    stocks: [
      {
        alpha_rank: 1,
        name: 'Infosys',
        ticker: 'INFY.NS',
        sector: 'IT',
        price: 1626.8,
        day_change_pct: 1.55,
        predicted_return_pct: 3.2,
        predicted_price: 1678.8,
        lower_price: 1580.0,
        upper_price: 1790.0,
        rr_ratio: 2.4,
        bullish_models: 6,
        total_models: 7,
        consensus_pct: 86,
        consensus_direction: 'bullish',
        mom_rank: 88,
        vol_ann_pct: 21.4,
      },
    ],
  });
  predictStockPrice.mockResolvedValue({
    stock_name: 'Infosys',
    ticker: 'INFY.NS',
    future_date: '2026-11-23',
    algorithm_name: 'Ensemble',
    predicted_price: '1645.20',
    prediction_type: '1-Day Forecast (Global Cross-Sectional Panel)',
    confidence_interval: { lower: 1605.0, upper: 1685.0 },
    trend_data: {
      dates: ['2026-11-20', '2026-11-23'],
      prices: [1626.8, 1645.2],
      history_cutoff: 1,
    },
    explainability: {
      anchor_horizon: 1,
      consensus: {
        bullish_count: 6,
        bearish_count: 1,
        total_models: 7,
        agreement_pct: 85.7,
        direction: 'bullish',
        label: 'High Conviction',
      },
      model_votes: [
        { algorithm: 'LightGBM', predicted_return_pct: 1.12, predicted_price: 1645.0, weight: 0.2 },
      ],
      top_drivers: [
        {
          feature: 'sec_rank_ret_5d',
          label: '5D Within-Sector Rank',
          category: 'Sector Rotation',
          impact_pct: 0.42,
          direction: 'bullish',
        },
      ],
      term_structure: [
        { horizon: 1, label: '1D', predicted_return_pct: 1.13, predicted_price: 1645.2, lower_price: 1605, upper_price: 1685 },
      ],
    },
  });
  optimizePortfolio.mockResolvedValue({
    mode: 'ai_bl',
    final_value: 124500,
    roi: 24.5,
    cagr: 24.5,
    sharpe_ratio: 1.85,
    portfolio_beta: 0.92,
    portfolio_vol_pct: 16.4,
    diversification_benefit_pct: 4.2,
    probability_of_profit: 76.4,
    var_95: 94000,
    cvar_95: 89000,
    simulation_paths: 10000,
    allocations: [
      { name: 'Infosys', ticker: 'INFY.NS', sector: 'IT', weight_pct: 55, allocated_inr: 55000, expected_return_pct: 18.2, ann_vol_pct: 20.1, risk_contribution_pct: 51 },
      { name: 'HDFC Bank', ticker: 'HDFCBANK.NS', sector: 'Banking & Financials', weight_pct: 45, allocated_inr: 45000, expected_return_pct: 14.5, ann_vol_pct: 17.8, risk_contribution_pct: 49 },
    ],
    chart_data: [{ time: '2026-11-23', value: 100000 }],
    p10_chart_data: [{ time: '2026-11-23', value: 100000 }],
    p90_chart_data: [{ time: '2026-11-23', value: 100000 }],
    efficient_frontier: { curve: [], cloud: [], markers: {} },
    correlation_matrix: { tickers: ['INFY', 'HDFCBANK'], values: [[1, 0.42], [0.42, 1]] },
    stress_tests: [],
  });
  runBacktest.mockResolvedValue({
    strategy: 'top_n_long',
    horizon: 21,
    algorithm: 'Ensemble',
    replay_mode: 'strict_oos_holdout',
    metrics: {
      final_capital: 142000,
      total_return_pct: 42.0,
      cagr_pct: 19.2,
      benchmark_return_pct: 21.5,
      alpha_pct: 20.5,
      sharpe_ratio: 1.48,
      sortino_ratio: 2.04,
      max_drawdown_pct: 9.4,
      benchmark_max_drawdown_pct: 13.8,
      win_rate_pct: 66.7,
      num_rebalances: 18,
      avg_turnover_pct: 45.0,
    },
    equity_curve: [{ time: '2025-01-01', strategy: 100000, benchmark: 100000 }],
    drawdown_curve: [{ time: '2025-01-01', strategy_dd: 0, benchmark_dd: 0 }],
    recent_rebalances: [],
  });
});

function renderApp() {
  return render(
    <ThemeProvider>
      <App />
    </ThemeProvider>
  );
}

test('renders NeuroStock brand header and login view when unauthenticated', () => {
  onAuthStateChanged.mockImplementation((_auth, callback) => {
    callback(null);
    return jest.fn();
  });

  renderApp();
  expect(screen.getAllByText(/NeuroStock/i).length).toBeGreaterThan(0);
  expect(screen.getByRole('heading', { name: /Login/i })).toBeInTheDocument();

  fireEvent.click(screen.getByRole('button', { name: /Sign Up/i }));
  expect(screen.getByRole('heading', { name: /Sign Up/i })).toBeInTheDocument();
});

test('toggles light/dark mode from the header ribbon and persists selection', () => {
  onAuthStateChanged.mockImplementation((_auth, callback) => {
    callback({ email: 'analyst@neurostock.ai' });
    return jest.fn();
  });

  renderApp();

  const toggleBtn = screen.getByRole('button', { name: /Switch to dark mode/i });
  expect(document.documentElement.getAttribute('data-theme')).toBe('light');

  fireEvent.click(toggleBtn);
  expect(document.documentElement.getAttribute('data-theme')).toBe('dark');
  expect(window.localStorage.getItem('neurostock-theme')).toBe('dark');
  expect(screen.getByRole('button', { name: /Switch to light mode/i })).toBeInTheDocument();

  fireEvent.click(screen.getByRole('button', { name: /Switch to light mode/i }));
  expect(document.documentElement.getAttribute('data-theme')).toBe('light');
  expect(window.localStorage.getItem('neurostock-theme')).toBe('light');
});

test('navigates cleanly across all authenticated dashboards and TrendAI drawer', async () => {
  onAuthStateChanged.mockImplementation((_auth, callback) => {
    callback({ email: 'analyst@neurostock.ai' });
    return jest.fn();
  });

  renderApp();

  expect(await screen.findByRole('heading', { name: /Market Overview/i })).toBeInTheDocument();
  expect((await screen.findAllByText(/Infosys/i)).length).toBeGreaterThan(0);

  fireEvent.click(screen.getByRole('tab', { name: /Alpha Radar/i }));
  expect(await screen.findByRole('heading', { name: /NSE 100 Alpha Radar/i })).toBeInTheDocument();
  expect(await screen.findByText(/NSE Sector Rotation Matrix/i)).toBeInTheDocument();

  fireEvent.click(screen.getByRole('tab', { name: /Positions/i }));
  expect(await screen.findByRole('heading', { name: /Market Positions/i })).toBeInTheDocument();
  expect((await screen.findAllByText(/HDFC Bank/i)).length).toBeGreaterThan(0);

  fireEvent.click(screen.getByRole('tab', { name: /Prediction/i }));
  expect(await screen.findByRole('heading', { name: /Price Prediction/i })).toBeInTheDocument();

  fireEvent.click(screen.getByRole('tab', { name: /Comparison/i }));
  expect(screen.getByRole('heading', { name: /Model Comparison/i })).toBeInTheDocument();

  fireEvent.click(screen.getByRole('tab', { name: /52 Week/i }));
  expect(await screen.findByRole('heading', { name: /52-Week Analysis/i })).toBeInTheDocument();

  fireEvent.click(screen.getByRole('tab', { name: /Backtest Lab/i }));
  expect(screen.getByRole('button', { name: /Run Backtest/i })).toBeInTheDocument();

  fireEvent.click(screen.getByRole('tab', { name: /Invest Engine/i }));
  expect(screen.getByRole('heading', { name: /AI Investment Engine/i })).toBeInTheDocument();

  fireEvent.click(screen.getByRole('tab', { name: /Portfolio/i }));
  expect(screen.getByRole('heading', { name: /AI Portfolio Projector/i })).toBeInTheDocument();
  fireEvent.click(screen.getByRole('button', { name: /Multi-Stock AI Optimizer/i }));
  expect(screen.getByRole('button', { name: /Optimize & Simulate/i })).toBeInTheDocument();

  fireEvent.click(screen.getByRole('tab', { name: /Sentiment/i }));
  expect(screen.getByRole('heading', { level: 1, name: /Market Sentiment/i })).toBeInTheDocument();

  fireEvent.click(screen.getByRole('tab', { name: /Education/i }));
  expect(screen.getByRole('heading', { name: /FinTeach Financial Tutor/i })).toBeInTheDocument();

  fireEvent.click(screen.getByRole('tab', { name: /Market/i }));
  expect((await screen.findAllByText(/Infosys/i)).length).toBeGreaterThan(0);

  // Watchlist star toggle & drawer
  fireEvent.click(screen.getByRole('button', { name: /Add Infosys to watchlist/i }));
  fireEvent.click(screen.getByRole('button', { name: /Open watchlist/i }));
  expect(await screen.findByRole('dialog', { name: /Watchlist & Price Alerts/i })).toBeInTheDocument();
  fireEvent.click(screen.getByRole('button', { name: /Close watchlist/i }));

  // Ctrl+K Global Command Palette
  fireEvent.keyDown(window, { key: 'k', ctrlKey: true });
  expect(screen.getByRole('dialog', { name: /Global Command Palette/i })).toBeInTheDocument();
  fireEvent.change(screen.getByLabelText(/Command palette search/i), { target: { value: 'TCS' } });
  fireEvent.click(screen.getByRole('option', { name: /TCS/i }));
  expect(await screen.findByRole('heading', { name: /Price Prediction/i })).toBeInTheDocument();

  fireEvent.click(screen.getByRole('button', { name: /Open TrendAI assistant/i }));
  expect(screen.getByRole('heading', { name: /TrendAI/i })).toBeInTheDocument();
});
