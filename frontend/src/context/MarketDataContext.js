import React, { createContext, useCallback, useContext, useEffect, useRef, useState } from 'react';
import { getMarketOverview } from '../services/apiClient';
import { isCancelledError } from '../utils/formatters';

const MarketDataContext = createContext({
  marketData: [],
  loading: false,
  error: null,
  refreshMarketData: () => {},
});

const POLL_INTERVAL_MS = 60000;

export const MarketDataProvider = ({ enabled = false, children }) => {
  const [marketData, setMarketData] = useState([]);
  const [loading, setLoading] = useState(Boolean(enabled));
  const [error, setError] = useState(null);
  const abortRef = useRef(null);
  const lastFetchedRef = useRef(0);
  const mountedRef = useRef(true);

  const fetchMarket = useCallback(async () => {
    if (!enabled) return;
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    try {
      const data = await getMarketOverview(controller.signal);
      if (!mountedRef.current || controller.signal.aborted) return;
      if (Array.isArray(data)) {
        setMarketData(data);
        lastFetchedRef.current = Date.now();
      }
      setError(null);
    } catch (err) {
      if (!mountedRef.current || controller.signal.aborted || isCancelledError(err)) return;
      setError(err.response?.data?.error || 'Could not refresh live NSE quotes.');
    } finally {
      if (mountedRef.current && abortRef.current === controller) {
        setLoading(false);
      }
    }
  }, [enabled]);

  useEffect(() => {
    mountedRef.current = true;
    if (!enabled) {
      abortRef.current?.abort();
      setMarketData([]);
      setLoading(false);
      setError(null);
      return () => {
        mountedRef.current = false;
      };
    }

    setLoading(true);
    fetchMarket();

    const intervalId = setInterval(() => {
      if (document.visibilityState === 'visible') {
        fetchMarket();
      }
    }, POLL_INTERVAL_MS);

    const handleVisibility = () => {
      if (document.visibilityState === 'visible' && Date.now() - lastFetchedRef.current >= 45000) {
        fetchMarket();
      }
    };

    document.addEventListener('visibilitychange', handleVisibility);
    return () => {
      mountedRef.current = false;
      clearInterval(intervalId);
      document.removeEventListener('visibilitychange', handleVisibility);
      abortRef.current?.abort();
    };
  }, [enabled, fetchMarket]);

  return (
    <MarketDataContext.Provider
      value={{ marketData, loading, error, refreshMarketData: fetchMarket }}
    >
      {children}
    </MarketDataContext.Provider>
  );
};

export const useMarketData = () => useContext(MarketDataContext);
