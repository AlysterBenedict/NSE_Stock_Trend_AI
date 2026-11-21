import { fireEvent, render, screen } from '@testing-library/react';
import App from './App';
import { ThemeProvider } from './context/ThemeContext';
import { getHistoricalData, getMarketOverview } from './services/apiClient';
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
  getAiInsights: jest.fn(),
  getGeneralFinancialKnowledge: jest.fn(),
  getStockSentiment: jest.fn(),
  calculatePortfolioProjection: jest.fn(),
  runInvestmentEngine: jest.fn(),
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

  fireEvent.click(screen.getByRole('tab', { name: /Positions/i }));
  expect(await screen.findByRole('heading', { name: /Market Positions/i })).toBeInTheDocument();
  expect((await screen.findAllByText(/HDFC Bank/i)).length).toBeGreaterThan(0);

  fireEvent.click(screen.getByRole('tab', { name: /Prediction/i }));
  expect(await screen.findByRole('heading', { name: /Price Prediction/i })).toBeInTheDocument();

  fireEvent.click(screen.getByRole('tab', { name: /Comparison/i }));
  expect(screen.getByRole('heading', { name: /Model Comparison/i })).toBeInTheDocument();

  fireEvent.click(screen.getByRole('tab', { name: /52 Week/i }));
  expect(await screen.findByRole('heading', { name: /52-Week Analysis/i })).toBeInTheDocument();

  fireEvent.click(screen.getByRole('tab', { name: /Invest Engine/i }));
  expect(screen.getByRole('heading', { name: /AI Investment Engine/i })).toBeInTheDocument();

  fireEvent.click(screen.getByRole('tab', { name: /Portfolio/i }));
  expect(screen.getByRole('heading', { name: /AI Portfolio Projector/i })).toBeInTheDocument();

  fireEvent.click(screen.getByRole('tab', { name: /Sentiment/i }));
  expect(screen.getByRole('heading', { level: 1, name: /Market Sentiment/i })).toBeInTheDocument();

  fireEvent.click(screen.getByRole('tab', { name: /Education/i }));
  expect(screen.getByRole('heading', { name: /FinTeach Financial Tutor/i })).toBeInTheDocument();

  fireEvent.click(screen.getByRole('tab', { name: /Market/i }));
  expect((await screen.findAllByText(/Infosys/i)).length).toBeGreaterThan(0);
  fireEvent.click(screen.getByRole('button', { name: /Open TrendAI assistant/i }));
  expect(screen.getByRole('heading', { name: /TrendAI/i })).toBeInTheDocument();
});
