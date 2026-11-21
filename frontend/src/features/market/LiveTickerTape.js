import React from 'react';
import { formatPrice, toneOf } from '../../utils/formatters';

const LiveTickerTape = ({ data }) => {
  const loopData = [...data, ...data];

  return (
    <div className="ticker-tape" aria-label="Live NSE prices">
      <div className="ticker-track">
        {loopData.map((item, index) => {
          const tone = toneOf(item.pChange ?? item.change);
          const pct = Number(item.pChange);
          const formattedPct = Number.isFinite(pct)
            ? `${pct > 0 ? '+' : ''}${pct.toFixed(2)}%`
            : `${item.pChange}%`;

          return (
            <div key={`${item.symbol}-${index}`} className="ticker-item">
              <span className="ticker-symbol">{item.symbol.split('.')[0]}</span>
              <span className="ticker-price">₹{formatPrice(item.price)}</span>
              <span className={`ticker-delta ticker-delta--${tone}`}>{formattedPct}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default LiveTickerTape;
