import React from 'react';
import { ChangeBadge, Icon } from '../../components/common/ui';
import { SECTOR_MAP } from '../../constants/stocks';
import { useWatchlist } from '../../context/WatchlistContext';
import { formatPrice, formatSigned, toneOf } from '../../utils/formatters';

/**
 * Honest 3-point intraday path derived directly from real Prev Close -> Open -> Current Price.
 */
const IntradayTrace = ({ close, open, price, tone }) => {
  const c = Number(close);
  const o = Number(open) > 0 ? Number(open) : c;
  const p = Number(price);

  if (!Number.isFinite(c) || !Number.isFinite(p)) return null;

  const pts = [c, o, p];
  const min = Math.min(...pts);
  const max = Math.max(...pts);
  const span = max - min;

  const yOf = (val) => {
    if (span < 1e-6) return 16;
    return 25 - ((val - min) / span) * 18;
  };

  const y0 = yOf(c);
  const y1 = yOf(o);
  const y2 = yOf(p);

  return (
    <svg className={`stock-trace stock-trace--${tone}`} viewBox="0 0 64 32" aria-hidden="true">
      <line x1="2" y1={y0} x2="62" y2={y0} className="stock-trace__ref" />
      <path d={`M 4 ${y0} L 30 ${y1} L 60 ${y2}`} className="stock-trace__line" />
      <circle cx="60" cy={y2} r="2.75" className="stock-trace__dot" />
    </svg>
  );
};

const MarketStockGrid = ({ data, onSelectStock }) => {
  const { isWatched, toggleWatch } = useWatchlist();

  return (
    <div className="stock-grid">
      {data.map((stock) => {
        const tone = toneOf(stock.pChange);
        const ticker = stock.symbol.split('.')[0];
        const sector = SECTOR_MAP[stock.symbol];
        const watched = isWatched(stock.symbol) || isWatched(stock.name);

        return (
          <div
            key={stock.symbol}
            role="button"
            tabIndex={0}
            className="stock-card"
            onClick={() => onSelectStock(stock)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                onSelectStock(stock);
              }
            }}
          >
            <div className="stock-card__top">
              <div className="stock-card__identity">
                <h3 className="stock-card__name">{stock.name}</h3>
                <span className="stock-card__meta">
                  {ticker} · NSE{sector ? ` · ${sector}` : ''}
                </span>
              </div>
              <div className="stock-card__top-actions">
                <button
                  type="button"
                  className={`watch-star${watched ? ' is-watched' : ''}`}
                  onClick={(e) => {
                    e.stopPropagation();
                    toggleWatch(stock);
                  }}
                  aria-label={watched ? `Remove ${stock.name} from watchlist` : `Add ${stock.name} to watchlist`}
                  title={watched ? 'Remove from watchlist' : 'Add to watchlist'}
                >
                  <Icon name="star" size={16} fill={watched} />
                </button>
                <ChangeBadge value={stock.pChange} />
              </div>
            </div>

            <div className="stock-card__bottom">
              <div className="stock-card__pricing">
                <div className="stock-card__price">₹{formatPrice(stock.price)}</div>
                <div className={`stock-card__delta stock-card__delta--${tone}`}>
                  {formatSigned(stock.change)} today
                </div>
              </div>
              <IntradayTrace close={stock.close} open={stock.open} price={stock.price} tone={tone} />
            </div>
          </div>
        );
      })}
    </div>
  );
};

export default MarketStockGrid;
