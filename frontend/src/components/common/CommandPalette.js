import React, { useEffect, useMemo, useRef, useState } from 'react';
import {
  PREDICTION_STOCK_OPTIONS,
  SECTOR_MAP,
  STOCK_SYMBOL_MAP,
} from '../../constants/stocks';
import { Icon } from './ui';

const CommandPalette = ({
  isOpen,
  onClose,
  tabGroups = [],
  onSelectTab,
  onSelectStockForecast,
  onToggleTheme,
  onOpenWatchlist,
  isDark = false,
}) => {
  const [query, setQuery] = useState('');
  const [activeIdx, setActiveIdx] = useState(0);
  const inputRef = useRef(null);

  useEffect(() => {
    if (isOpen) {
      setQuery('');
      setActiveIdx(0);
      setTimeout(() => inputRef.current?.focus(), 10);
    }
  }, [isOpen]);

  const entries = useMemo(() => {
    const q = query.trim().toLowerCase();

    const quickActions = [
      {
        id: 'act-theme',
        type: 'Action',
        label: isDark ? 'Switch to Light Mode' : 'Switch to Dark Mode',
        meta: 'Appearance',
        icon: isDark ? 'light_mode' : 'dark_mode',
        run: () => onToggleTheme?.(),
      },
      {
        id: 'act-watchlist',
        type: 'Action',
        label: 'Open Watchlist & Price Alerts',
        meta: 'Tracked NSE equities',
        icon: 'star',
        run: () => onOpenWatchlist?.(),
      },
    ];

    const tabItems = tabGroups.flatMap((group) =>
      group.tabs.map((t) => ({
        id: `tab-${t.id}`,
        type: 'View',
        label: `Go to ${t.label}`,
        meta: group.label,
        icon: t.icon,
        run: () => onSelectTab?.(t.id),
      }))
    );

    const stockItems = PREDICTION_STOCK_OPTIONS.map((opt) => {
      const symbol = STOCK_SYMBOL_MAP[opt.value] || '';
      const sector = SECTOR_MAP[symbol] || 'NSE';
      return {
        id: `stock-${opt.value}`,
        type: 'Stock',
        label: opt.value,
        meta: `${symbol.split('.')[0]} · ${sector}`,
        icon: 'query_stats',
        run: () => onSelectStockForecast?.(opt.value),
      };
    });

    const all = [...quickActions, ...tabItems, ...stockItems];
    if (!q) {
      return [...quickActions, ...tabItems.slice(0, 5), ...stockItems.slice(0, 6)];
    }
    return all
      .filter(
        (item) =>
          item.label.toLowerCase().includes(q) ||
          item.meta.toLowerCase().includes(q) ||
          item.type.toLowerCase().includes(q)
      )
      .slice(0, 14);
  }, [query, tabGroups, isDark, onToggleTheme, onOpenWatchlist, onSelectTab, onSelectStockForecast]);

  useEffect(() => {
    setActiveIdx(0);
  }, [query]);

  if (!isOpen) return null;

  const executeEntry = (entry) => {
    if (!entry) return;
    entry.run();
    onClose();
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Escape') {
      e.preventDefault();
      onClose();
    } else if (e.key === 'ArrowDown') {
      e.preventDefault();
      setActiveIdx((prev) => (entries.length ? (prev + 1) % entries.length : 0));
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setActiveIdx((prev) => (entries.length ? (prev - 1 + entries.length) % entries.length : 0));
    } else if (e.key === 'Enter' && entries[activeIdx]) {
      e.preventDefault();
      executeEntry(entries[activeIdx]);
    }
  };

  return (
    <div
      className="cmd-backdrop"
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div
        className="cmd-modal"
        role="dialog"
        aria-modal="true"
        aria-label="Global Command Palette"
        onKeyDown={handleKeyDown}
      >
        <div className="cmd-input-row">
          <Icon name="search" size={20} className="cmd-input-row__icon" />
          <input
            ref={inputRef}
            type="text"
            className="cmd-input"
            placeholder="Search any NSE 100 stock, dashboard, or action..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            aria-label="Command palette search"
          />
          <kbd className="cmd-kbd">ESC</kbd>
        </div>

        <div className="cmd-results" role="listbox" aria-label="Search results">
          {entries.length === 0 ? (
            <div className="cmd-empty">No matching NSE stocks, views, or actions.</div>
          ) : (
            entries.map((item, idx) => (
              <button
                key={item.id}
                type="button"
                role="option"
                aria-selected={idx === activeIdx}
                className={`cmd-item${idx === activeIdx ? ' is-active' : ''}`}
                onMouseEnter={() => setActiveIdx(idx)}
                onClick={() => executeEntry(item)}
              >
                <span className="cmd-item__left">
                  <Icon name={item.icon} size={18} />
                  <span>
                    <strong>{item.label}</strong>
                    <small>{item.meta}</small>
                  </span>
                </span>
                <span className="cmd-item__type">{item.type}</span>
              </button>
            ))
          )}
        </div>
      </div>
    </div>
  );
};

export default CommandPalette;
