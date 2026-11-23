import React, { createContext, useCallback, useContext, useEffect, useState } from 'react';

const WatchlistContext = createContext(null);
const MAX_WATCHLIST_ITEMS = 100;

const sanitizeKeyPart = (raw) => String(raw || 'guest').replace(/[^a-zA-Z0-9_.-]/g, '_').slice(0, 80);
const storageKeyFor = (userKey) => `neurostock-watchlist-${sanitizeKeyPart(userKey)}`;

const sanitizePositiveNumber = (val) => {
  if (val === null || val === undefined || val === '') return null;
  const num = Number(val);
  return Number.isFinite(num) && num > 0 && num < 1e9 ? num : null;
};

export const sanitizeWatchlistItems = (rawParsed) => {
  if (!Array.isArray(rawParsed)) return [];
  const seen = new Set();
  const clean = [];
  for (const item of rawParsed) {
    if (!item || typeof item !== 'object' || Array.isArray(item)) continue;
    const symbol = typeof item.symbol === 'string' ? item.symbol.trim().slice(0, 32) : '';
    const name = typeof item.name === 'string' ? item.name.trim().slice(0, 64) : '';
    if (!symbol && !name) continue;
    const normSym = symbol || name;
    if (seen.has(normSym)) continue;
    seen.add(normSym);
    clean.push({
      name: name || normSym.split('.')[0],
      symbol: normSym,
      addedAt: typeof item.addedAt === 'string' ? item.addedAt.slice(0, 40) : new Date().toISOString(),
      alertAbove: sanitizePositiveNumber(item.alertAbove),
      alertBelow: sanitizePositiveNumber(item.alertBelow),
    });
    if (clean.length >= MAX_WATCHLIST_ITEMS) break;
  }
  return clean;
};

const loadStoredWatchlist = (userKey) => {
  try {
    const raw = window.localStorage.getItem(storageKeyFor(userKey));
    const parsed = raw ? JSON.parse(raw) : [];
    return sanitizeWatchlistItems(parsed);
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
      const sanitized = sanitizeWatchlistItems(next);
      setItems(sanitized);
      try {
        window.localStorage.setItem(storageKeyFor(userKey), JSON.stringify(sanitized));
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
      const parsedAbove = sanitizePositiveNumber(alertAbove);
      const parsedBelow = sanitizePositiveNumber(alertBelow);

      persist(
        items.map((it) =>
          it.symbol === symbol
            ? {
                ...it,
                alertAbove: parsedAbove,
                alertBelow: parsedBelow,
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
