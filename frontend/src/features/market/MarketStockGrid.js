import React from 'react';
import { ChangeBadge } from '../../components/common/ui';
import { SECTOR_MAP } from '../../constants/stocks';
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

const MarketStockGrid = ({ data, onSelectStock }) => (
  <div className="stock-grid">
    {data.map((stock) => {
      const tone = toneOf(stock.pChange);
      const ticker = stock.symbol.split('.')[0];
      const sector = SECTOR_MAP[stock.symbol];

      return (
        <button
          key={stock.symbol}
          type="button"
          className="stock-card"
          onClick={() => onSelectStock(stock)}
        >
          <div className="stock-card__top">
            <div className="stock-card__identity">
              <h3 className="stock-card__name">{stock.name}</h3>
              <span className="stock-card__meta">
                {ticker} · NSE{sector ? ` · ${sector}` : ''}
              </span>
            </div>
            <ChangeBadge value={stock.pChange} />
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
        </button>
      );
    })}
  </div>
);

export default MarketStockGrid;
