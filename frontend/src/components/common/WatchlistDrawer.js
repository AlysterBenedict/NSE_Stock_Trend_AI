import React, { useEffect, useMemo } from 'react';
import { useMarketData } from '../../context/MarketDataContext';
import { useWatchlist } from '../../context/WatchlistContext';
import { formatPrice, formatSigned } from '../../utils/formatters';
import { Badge, ChangeBadge, EmptyState, Icon } from './ui';

const WatchlistDrawer = ({ isOpen, onClose, onJumpToForecast }) => {
  const { items, removeWatch, setPriceAlert } = useWatchlist();
  const { marketData, loading: refreshing } = useMarketData();

  useEffect(() => {
    if (!isOpen) return undefined;
    const handleKeyDown = (e) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, onClose]);

  const quotesMap = useMemo(() => {
    const map = {};
    if (Array.isArray(marketData)) {
      marketData.forEach((row) => {
        map[row.symbol] = row;
        map[row.name] = row;
      });
    }
    return map;
  }, [marketData]);

  if (!isOpen) return null;

  return (
    <div
      className="drawer-backdrop"
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <aside
        className="watchlist-drawer"
        role="dialog"
        aria-modal="true"
        aria-label="Watchlist & Price Alerts"
      >
        <header className="watchlist-drawer__head">
          <div>
            <h3>Watchlist &amp; Price Alerts</h3>
            <span>
              {items.length} tracked {items.length === 1 ? 'equity' : 'equities'}
              {refreshing ? ' · syncing quotes...' : ''}
            </span>
          </div>
          <button
            type="button"
            className="icon-btn"
            onClick={onClose}
            aria-label="Close watchlist"
          >
            <Icon name="close" size={20} />
          </button>
        </header>

        <div className="watchlist-drawer__body">
          {items.length === 0 ? (
            <EmptyState
              icon="star"
              title="No Stocks Watched Yet"
              description="Click the star icon on any stock card in Market Overview or Alpha Radar to pin it here and set target price alerts."
            />
          ) : (
            <div className="watchlist-list">
              {items.map((item) => {
                const quote = quotesMap[item.symbol] || quotesMap[item.name];
                const spot = quote ? Number(quote.price) : null;
                const hitAbove =
                  Number.isFinite(spot) &&
                  item.alertAbove !== null &&
                  spot >= Number(item.alertAbove);
                const hitBelow =
                  Number.isFinite(spot) &&
                  item.alertBelow !== null &&
                  spot <= Number(item.alertBelow);

                return (
                  <div
                    key={item.symbol}
                    className={`watchlist-card${hitAbove || hitBelow ? ' is-triggered' : ''}`}
                  >
                    <div className="watchlist-card__top">
                      <div>
                        <strong>{item.name}</strong>
                        <span>{item.symbol}</span>
                      </div>
                      <div className="watchlist-card__quote">
                        {spot !== null ? (
                          <>
                            <strong>₹{formatPrice(spot)}</strong>
                            <ChangeBadge value={quote.pChange} />
                          </>
                        ) : (
                          <span className="muted-text">Quote unavailable</span>
                        )}
                      </div>
                    </div>

                    {(hitAbove || hitBelow) && (
                      <div className="watchlist-card__alerts">
                        {hitAbove && (
                          <Badge tone="up" icon="notifications_active">
                            Breakout Triggered &gt;= ₹{formatPrice(item.alertAbove)}
                          </Badge>
                        )}
                        {hitBelow && (
                          <Badge tone="down" icon="notifications_active">
                            Support Triggered &lt;= ₹{formatPrice(item.alertBelow)}
                          </Badge>
                        )}
                      </div>
                    )}

                    <div className="watchlist-alert-inputs">
                      <label>
                        <span>Alert Above (₹)</span>
                        <input
                          type="number"
                          className="input"
                          placeholder="Optional"
                          value={item.alertAbove ?? ''}
                          onChange={(e) =>
                            setPriceAlert(item.symbol, {
                              alertAbove: e.target.value,
                              alertBelow: item.alertBelow,
                            })
                          }
                        />
                      </label>
                      <label>
                        <span>Alert Below (₹)</span>
                        <input
                          type="number"
                          className="input"
                          placeholder="Optional"
                          value={item.alertBelow ?? ''}
                          onChange={(e) =>
                            setPriceAlert(item.symbol, {
                              alertAbove: item.alertAbove,
                              alertBelow: e.target.value,
                            })
                          }
                        />
                      </label>
                    </div>

                    <div className="watchlist-card__actions">
                      <button
                        type="button"
                        className="btn btn--primary btn--sm"
                        onClick={() => {
                          onJumpToForecast?.(item.name);
                          onClose();
                        }}
                      >
                        <Icon name="query_stats" size={15} />
                        <span>Forecast</span>
                      </button>
                      {quote && (
                        <span className="watchlist-card__delta">
                          Today: {formatSigned(quote.change)} ₹
                        </span>
                      )}
                      <button
                        type="button"
                        className="btn btn--secondary btn--sm"
                        onClick={() => removeWatch(item.symbol)}
                        aria-label={`Remove ${item.name} from watchlist`}
                      >
                        Remove
                      </button>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </aside>
    </div>
  );
};

export default WatchlistDrawer;
