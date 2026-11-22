import React from 'react';
import { Callout } from '../../components/common/ui';

const cellStyle = (val, isDiag) => {
  if (isDiag) {
    return {
      background: 'var(--primary-fixed)',
      color: 'var(--on-primary-fixed)',
      fontWeight: 700,
    };
  }
  const v = Number(val) || 0;
  if (v >= 0.7) {
    return { background: 'var(--tertiary-fixed)', color: 'var(--on-tertiary-fixed)', fontWeight: 700 };
  }
  if (v >= 0.45) {
    return { background: 'var(--secondary-fixed)', color: 'var(--on-secondary-fixed-variant)' };
  }
  return { background: 'var(--well-bg)', color: 'var(--on-surface)' };
};

const CorrelationHeatmap = ({ matrix }) => {
  if (!matrix || !matrix.tickers || matrix.tickers.length === 0) return null;

  const { tickers, values } = matrix;
  let highPairs = 0;
  for (let i = 0; i < values.length; i += 1) {
    for (let j = i + 1; j < values.length; j += 1) {
      if ((values[i]?.[j] || 0) >= 0.7) highPairs += 1;
    }
  }

  return (
    <div className="corr-wrap">
      <div className="table-scroll">
        <table className="corr-table">
          <thead>
            <tr>
              <th />
              {tickers.map((t) => (
                <th key={t}>{t}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {tickers.map((rowT, r) => (
              <tr key={rowT}>
                <th>{rowT}</th>
                {tickers.map((colT, c) => {
                  const val = values[r]?.[c] ?? 0;
                  return (
                    <td key={`${rowT}-${colT}`} style={cellStyle(val, r === c)}>
                      {Number(val).toFixed(2)}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {highPairs > 0 && (
        <Callout tone="secondary" icon="warning" title="Concentration Check">
          {highPairs} asset pair{highPairs > 1 ? 's have' : ' has'} correlation ≥ 0.70. Consider swapping one stock from a different NSE sector to improve diversification.
        </Callout>
      )}
    </div>
  );
};

export default CorrelationHeatmap;
