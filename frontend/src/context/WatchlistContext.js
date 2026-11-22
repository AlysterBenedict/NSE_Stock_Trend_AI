import React, { createContext, useCallback, useContext, useEffect, useState } from 'react';

const WatchlistContext = createContext(null);

const storageKeyFor = (userKey) => `neurostock-watchlist-${userKey || 'guest'}`;

const loadStoredWatchlist = (userKey) => {
  try {
    const raw = window.localStorage.getItem(storageKeyFor(userKey));
    const parsed = raw ? JSON.parse(raw) : [];
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
};

export const WatchlistProvider = ({ userKey = 'guest', children }) => {
  const [items, setItems] = useState(() => loadStoredWatchlist(userKey));

  useEffect(() => {
    setItems(loadStoredWatchlist(userKey));
  }, [userKey]);

  const persist = useCallback(
    (next) => {
      setItems(next);
      try {
        window.localStorage.setItem(storageKeyFor(userKey), JSON.stringify(next));
      } catch {
        // Ignore storage quota errors
      }
    },
    [userKey]
  );

  const isWatched = useCallback(
    (symbolOrName) => {
      if (!symbolOrName) return false;
      const needle = String(symbolOrName).toLowerCase();
      return items.some(
        (item) =>
          item.symbol?.toLowerCase() === needle || item.name?.toLowerCase() === needle
      );
    },
    [items]
  );

  const toggleWatch = useCallback(
    (stock) => {
      if (!stock || (!stock.symbol && !stock.name)) return;
      const sym = stock.symbol || stock.name;
      const exists = items.some((it) => it.symbol === sym || it.name === stock.name);
      if (exists) {
        persist(items.filter((it) => it.symbol !== sym && it.name !== stock.name));
      } else {
        persist([
          {
            name: stock.name || sym.split('.')[0],
            symbol: sym,
            addedAt: new Date().toISOString(),
            alertAbove: null,
            alertBelow: null,
          },
          ...items,
        ]);
      }
    },
    [items, persist]
  );

  const removeWatch = useCallback(
    (symbol) => {
      persist(items.filter((it) => it.symbol !== symbol && it.name !== symbol));
    },
    [items, persist]
  );

  const setPriceAlert = useCallback(
    (symbol, { alertAbove, alertBelow }) => {
      const parsedAbove =
        alertAbove === '' || alertAbove === null || alertAbove === undefined
          ? null
          : Number(alertAbove);
      const parsedBelow =
        alertBelow === '' || alertBelow === null || alertBelow === undefined
          ? null
          : Number(alertBelow);

      persist(
        items.map((it) =>
          it.symbol === symbol
            ? {
                ...it,
                alertAbove: Number.isFinite(parsedAbove) && parsedAbove > 0 ? parsedAbove : null,
                alertBelow: Number.isFinite(parsedBelow) && parsedBelow > 0 ? parsedBelow : null,
              }
            : it
        )
      );
    },
    [items, persist]
  );

  const clearTriggered = useCallback(
    (symbol) => {
      persist(
        items.map((it) =>
          it.symbol === symbol ? { ...it, alertAbove: null, alertBelow: null } : it
        )
      );
    },
    [items, persist]
  );

  return (
    <WatchlistContext.Provider
      value={{
        items,
        isWatched,
        toggleWatch,
        removeWatch,
        setPriceAlert,
        clearTriggered,
      }}
    >
      {children}
    </WatchlistContext.Provider>
  );
};

export const useWatchlist = () => {
  const ctx = useContext(WatchlistContext);
  if (!ctx) {
    return {
      items: [],
      isWatched: () => false,
      toggleWatch: () => {},
      removeWatch: () => {},
      setPriceAlert: () => {},
      clearTriggered: () => {},
    };
  }
  return ctx;
};
