import axios from 'axios';

const api = axios.create({
  baseURL: process.env.REACT_APP_API_BASE_URL || '',
});

export const predictStockPrice = (payload, signal) =>
  api.post('/predict', payload, { signal }).then((res) => res.data);

export const getHistoricalData = (stockName) =>
  api.get('/historical-data', { params: { stock_name: stockName } }).then((res) => res.data);

export const getMarketOverview = () =>
  api.get('/get-market-data').then((res) => res.data);

export const getStockMarketDetail = (ticker) =>
  api.get('/get-market-data', { params: { ticker } }).then((res) => res.data);

export const getAlphaRadar = (horizon = 21, refresh = false, signal) =>
  api.get('/alpha-radar', { params: { horizon, refresh }, signal }).then((res) => res.data);

export const getAiInsights = (payload, signal) =>
  api.post('/get-ai-insights', payload, { signal }).then((res) => res.data);

export const getGeneralFinancialKnowledge = (userQuestion, signal) =>
  api.post('/get-general-knowledge', { user_question: userQuestion }, { signal }).then((res) => res.data);

export const getStockSentiment = (stockName, signal) =>
  api.post('/get-sentiment', { stock_name: stockName }, { signal }).then((res) => res.data);

export const calculatePortfolioProjection = (payload, signal) =>
  api.post('/calculate-portfolio', payload, { signal }).then((res) => res.data);

export const optimizePortfolio = (payload, signal) =>
  api.post('/optimize-portfolio', payload, { signal }).then((res) => res.data);

export const runBacktest = (payload, signal) =>
  api.post('/run-backtest', payload, { signal }).then((res) => res.data);

export const runInvestmentEngine = (payload, signal) =>
  api.post('/investment-engine', payload, { signal }).then((res) => res.data);

export default api;
