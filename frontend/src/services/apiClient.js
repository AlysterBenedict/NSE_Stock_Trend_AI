import axios from 'axios';
import { auth } from './firebaseAuth';

const api = axios.create({
  baseURL: process.env.REACT_APP_API_BASE_URL || '',
  timeout: 60000,
});

// Inject fresh Firebase ID tokens on every outbound API request
api.interceptors.request.use(async (config) => {
  try {
    const user = auth?.currentUser;
    if (user && typeof user.getIdToken === 'function') {
      const token = await user.getIdToken();
      if (token) {
        config.headers = config.headers || {};
        config.headers.Authorization = `Bearer ${token}`;
      }
    }
  } catch {
    // Let backend handle missing/expired auth cleanly
  }
  return config;
});

// Automatically refresh Firebase ID token once on 401
api.interceptors.response.use(
  (response) => response,
  async (error) => {
    const originalRequest = error?.config;
    if (
      error?.response?.status === 401 &&
      originalRequest &&
      !originalRequest._retried &&
      auth?.currentUser &&
      typeof auth.currentUser.getIdToken === 'function'
    ) {
      originalRequest._retried = true;
      try {
        const refreshed = await auth.currentUser.getIdToken(true);
        if (refreshed) {
          originalRequest.headers = originalRequest.headers || {};
          originalRequest.headers.Authorization = `Bearer ${refreshed}`;
          return api(originalRequest);
        }
      } catch {
        // Fall through to caller error handler
      }
    }
    return Promise.reject(error);
  }
);

let inflightOverviewPromise = null;

export const predictStockPrice = (payload, signal) =>
  api.post('/predict', payload, { signal, timeout: 90000 }).then((res) => res.data);

export const getHistoricalData = (stockName, signal) =>
  api.get('/historical-data', { params: { stock_name: stockName }, signal }).then((res) => res.data);

export const getMarketOverview = (signal) => {
  if (signal) {
    return api.get('/get-market-data', { signal }).then((res) => res.data);
  }
  if (!inflightOverviewPromise) {
    inflightOverviewPromise = api
      .get('/get-market-data')
      .then((res) => res.data)
      .finally(() => {
        inflightOverviewPromise = null;
      });
  }
  return inflightOverviewPromise;
};

export const getStockMarketDetail = (ticker, signal) =>
  api.get('/get-market-data', { params: { ticker }, signal }).then((res) => res.data);

export const getAlphaRadar = (horizon = 21, refresh = false, signal) =>
  api.get('/alpha-radar', { params: { horizon, refresh }, signal, timeout: 90000 }).then((res) => res.data);

export const getAiInsights = (payload, signal) =>
  api.post('/get-ai-insights', payload, { signal, timeout: 90000 }).then((res) => res.data);

export const getGeneralFinancialKnowledge = (userQuestion, signal) =>
  api
    .post('/get-general-knowledge', { user_question: userQuestion }, { signal, timeout: 90000 })
    .then((res) => res.data);

export const getStockSentiment = (stockName, signal) =>
  api.post('/get-sentiment', { stock_name: stockName }, { signal, timeout: 90000 }).then((res) => res.data);

export const calculatePortfolioProjection = (payload, signal) =>
  api.post('/calculate-portfolio', payload, { signal, timeout: 90000 }).then((res) => res.data);

export const optimizePortfolio = (payload, signal) =>
  api.post('/optimize-portfolio', payload, { signal, timeout: 120000 }).then((res) => res.data);

export const runBacktest = (payload, signal) =>
  api.post('/run-backtest', payload, { signal, timeout: 120000 }).then((res) => res.data);

export const runInvestmentEngine = (payload, signal) =>
  api.post('/investment-engine', payload, { signal, timeout: 120000 }).then((res) => res.data);

export const getLlmSettings = (signal) =>
  api.get('/llm-settings', { signal, timeout: 15000 }).then((res) => res.data);

export const saveLlmApiKey = (apiKey, signal) =>
  api.post('/llm-settings', { api_key: apiKey }, { signal, timeout: 15000 }).then((res) => res.data);

export const deleteLlmApiKey = (signal) =>
  api.delete('/llm-settings', { signal, timeout: 15000 }).then((res) => res.data);

export default api;
