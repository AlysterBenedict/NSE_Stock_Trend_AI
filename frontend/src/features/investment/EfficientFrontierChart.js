import React, { useState } from 'react';
import { useTheme } from '../../context/ThemeContext';
import { getChartColors } from '../../utils/chartHelpers';

const MARKER_META = {
  equal_weight: { label: 'Equal Weight', key: 'equal_weight' },
  min_var: { label: 'Min Variance', key: 'min_var' },
  ai_bl: { label: 'AI Max Sharpe', key: 'ai_bl' },
  active: { label: 'Your Portfolio', key: 'active' },
};

const EfficientFrontierChart = ({ frontier }) => {
  const { theme } = useTheme();
  const colors = getChartColors(theme);
  const [hoverLabel, setHoverLabel] = useState(null);

  if (!frontier || !frontier.curve || frontier.curve.length === 0) return null;

  const { curve = [], cloud = [], markers = {} } = frontier;
  const allPts = [...curve, ...cloud, ...Object.values(markers).filter(Boolean)];
  const vols = allPts.map((p) => Number(p.vol_pct) || 0);
  const rets = allPts.map((p) => Number(p.ret_pct) || 0);

  const minV = Math.max(0, Math.min(...vols) - 1.5);
  const maxV = Math.max(minV + 5, Math.max(...vols) + 1.5);
  const minR = Math.min(...rets) - 2.0;
  const maxR = Math.max(minR + 4, Math.max(...rets) + 2.0);

  const width = 520;
  const height = 290;
  const pad = { top: 20, right: 20, bottom: 38, left: 48 };
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;

  const xOf = (v) => pad.left + ((v - minV) / (maxV - minV)) * plotW;
  const yOf = (r) => pad.top + ((maxR - r) / (maxR - minR)) * plotH;

  const sortedCurve = [...curve].sort((a, b) => a.vol_pct - b.vol_pct);
  const pathD = sortedCurve
    .map((p, i) => `${i === 0 ? 'M' : 'L'} ${xOf(p.vol_pct).toFixed(1)} ${yOf(p.ret_pct).toFixed(1)}`)
    .join(' ');

  const markerStyles = {
    equal_weight: { fill: colors.text, r: 5.5 },
    min_var: { fill: colors.secondary, r: 6 },
    ai_bl: { fill: colors.up, r: 6.5 },
    active: { fill: colors.primary, r: 7.5 },
  };

  return (
    <div className="frontier-wrap">
      <svg viewBox={`0 0 ${width} ${height}`} className="rotation-svg" role="img" aria-label="Markowitz Efficient Frontier">
        {/* Horizontal grid lines */}
        {[0.2, 0.5, 0.8].map((frac) => {
          const rVal = minR + (maxR - minR) * frac;
          const yy = yOf(rVal);
          return (
            <g key={frac}>
              <line x1={pad.left} y1={yy} x2={pad.left + plotW} y2={yy} stroke={colors.grid} strokeWidth="1" />
              <text x={pad.left - 6} y={yy + 3} textAnchor="end" fontSize="9.5" fill={colors.text}>
                {rVal.toFixed(0)}%
              </text>
            </g>
          );
        })}

        {/* Random portfolio cloud */}
        {cloud.map((pt, idx) => (
          <circle
            key={idx}
            cx={xOf(pt.vol_pct)}
            cy={yOf(pt.ret_pct)}
            r="2.4"
            fill={colors.primary}
            fillOpacity="0.22"
          />
        ))}

        {/* Efficient Frontier Curve */}
        {pathD && (
          <path
            d={pathD}
            fill="none"
            stroke={colors.primary}
            strokeWidth="2.4"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        )}

        {/* Strategy markers */}
        {Object.entries(MARKER_META).map(([key, meta]) => {
          const pt = markers[key];
          if (!pt) return null;
          const st = markerStyles[key] || { fill: colors.primary, r: 6 };
          const cx = xOf(pt.vol_pct);
          const cy = yOf(pt.ret_pct);
          return (
            <g
              key={key}
              onMouseEnter={() => setHoverLabel(`${meta.label}: Ret ${pt.ret_pct}% | Vol ${pt.vol_pct}% | Sharpe ${pt.sharpe}`)}
              onMouseLeave={() => setHoverLabel(null)}
              style={{ cursor: 'pointer' }}
            >
              <circle
                cx={cx}
                cy={cy}
                r={st.r}
                fill={st.fill}
                stroke="#ffffff"
                strokeWidth="2"
              />
            </g>
          );
        })}

        <text x={pad.left + plotW / 2} y={height - 8} textAnchor="middle" className="rotation-axis-title" fill={colors.text}>
          Annualized Portfolio Volatility (%)
        </text>
        <text
          x={14}
          y={pad.top + plotH / 2}
          textAnchor="middle"
          transform={`rotate(-90 14 ${pad.top + plotH / 2})`}
          className="rotation-axis-title"
          fill={colors.text}
        >
          Expected Return (%)
        </text>
      </svg>

      <div className="frontier-legend">
        {Object.entries(MARKER_META).map(([key, meta]) => {
          const pt = markers[key];
          if (!pt) return null;
          return (
            <span key={key} className="frontier-legend__item">
              <span className="frontier-dot" style={{ background: markerStyles[key].fill }} />
              {meta.label}: <strong>{pt.ret_pct}%</strong> / {pt.vol_pct}% vol
            </span>
          );
        })}
      </div>
      {hoverLabel && <div className="rotation-tooltip">{hoverLabel}</div>}
    </div>
  );
};

export default EfficientFrontierChart;
