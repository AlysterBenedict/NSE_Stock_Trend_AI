import React, { useState } from 'react';
import { useTheme } from '../../context/ThemeContext';
import { getChartColors } from '../../utils/chartHelpers';

const QUADRANT_BADGE_TONE = {
  Leading: 'up',
  Improving: 'secondary',
  Weakening: 'secondary',
  Lagging: 'down',
};

const SectorRotationChart = ({ sectors = [], activeSector = 'All', onSelectSector }) => {
  const { theme } = useTheme();
  const colors = getChartColors(theme);
  const [hovered, setHovered] = useState(null);

  if (!sectors.length) return null;

  const width = 540;
  const height = 310;
  const pad = { top: 26, right: 26, bottom: 38, left: 46 };
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;

  const maxAbsX = Math.max(2.5, ...sectors.map((s) => Math.abs(Number(s.mom_21d_pct) || 0) * 1.2));
  const maxAbsY = Math.max(3.5, ...sectors.map((s) => Math.abs(Number(s.mom_63d_pct) || 0) * 1.2));

  const xOf = (val) => pad.left + ((val + maxAbsX) / (2 * maxAbsX)) * plotW;
  const yOf = (val) => pad.top + ((maxAbsY - val) / (2 * maxAbsY)) * plotH;
  const zeroX = xOf(0);
  const zeroY = yOf(0);

  const colorForQuadrant = (quad) => {
    if (quad === 'Leading') return colors.up;
    if (quad === 'Lagging') return colors.down;
    return colors.secondary;
  };

  return (
    <div className="rotation-wrap">
      <svg
        viewBox={`0 0 ${width} ${height}`}
        className="rotation-svg"
        role="img"
        aria-label="NSE Sector Rotation Quadrant"
      >
        {/* Subtle quadrant tints */}
        <rect x={zeroX} y={pad.top} width={pad.left + plotW - zeroX} height={zeroY - pad.top} fill="rgba(93, 122, 104, 0.08)" />
        <rect x={pad.left} y={pad.top} width={zeroX - pad.left} height={zeroY - pad.top} fill="rgba(127, 85, 54, 0.05)" />
        <rect x={zeroX} y={zeroY} width={pad.left + plotW - zeroX} height={pad.top + plotH - zeroY} fill="rgba(127, 85, 54, 0.05)" />
        <rect x={pad.left} y={zeroY} width={zeroX - pad.left} height={pad.top + plotH - zeroY} fill="rgba(156, 100, 109, 0.08)" />

        {/* Crosshair axes */}
        <line x1={pad.left} y1={zeroY} x2={pad.left + plotW} y2={zeroY} stroke={colors.grid} strokeWidth="1.5" strokeDasharray="4 4" />
        <line x1={zeroX} y1={pad.top} x2={zeroX} y2={pad.top + plotH} stroke={colors.grid} strokeWidth="1.5" strokeDasharray="4 4" />

        {/* Quadrant corner labels */}
        <text x={pad.left + plotW - 6} y={pad.top + 14} textAnchor="end" className="rotation-quad-label" fill={colors.up}>
          LEADING
        </text>
        <text x={pad.left + 6} y={pad.top + 14} textAnchor="start" className="rotation-quad-label" fill={colors.secondary}>
          WEAKENING
        </text>
        <text x={pad.left + plotW - 6} y={pad.top + plotH - 6} textAnchor="end" className="rotation-quad-label" fill={colors.secondary}>
          IMPROVING
        </text>
        <text x={pad.left + 6} y={pad.top + plotH - 6} textAnchor="start" className="rotation-quad-label" fill={colors.down}>
          LAGGING
        </text>

        {/* Sector bubbles */}
        {sectors.map((sec) => {
          const cx = xOf(Number(sec.mom_21d_pct) || 0);
          const cy = yOf(Number(sec.mom_63d_pct) || 0);
          const r = Math.min(18, Math.max(7, 5 + Math.sqrt(sec.stock_count || 1) * 2.2));
          const isSelected = activeSector === sec.sector;
          const isHover = hovered?.sector === sec.sector;
          const fill = colorForQuadrant(sec.quadrant);

          return (
            <g
              key={sec.sector}
              className="rotation-node"
              onClick={() => onSelectSector?.(activeSector === sec.sector ? 'All' : sec.sector)}
              onMouseEnter={() => setHovered(sec)}
              onMouseLeave={() => setHovered(null)}
              style={{ cursor: 'pointer' }}
            >
              <circle
                cx={cx}
                cy={cy}
                r={isSelected || isHover ? r + 3 : r}
                fill={fill}
                fillOpacity={isSelected ? 0.92 : 0.72}
                stroke={isSelected ? colors.historical : '#ffffff'}
                strokeWidth={isSelected ? 2.5 : 1.5}
              />
              <text
                x={cx}
                y={cy - r - 5}
                textAnchor="middle"
                className="rotation-node__label"
                fill={colors.text}
              >
                {sec.sector}
              </text>
            </g>
          );
        })}

        {/* Axis captions */}
        <text x={pad.left + plotW / 2} y={height - 8} textAnchor="middle" className="rotation-axis-title" fill={colors.text}>
          21-Day Sector Momentum vs Nifty Median (%)
        </text>
        <text
          x={14}
          y={pad.top + plotH / 2}
          textAnchor="middle"
          transform={`rotate(-90 14 ${pad.top + plotH / 2})`}
          className="rotation-axis-title"
          fill={colors.text}
        >
          63-Day Trend (%)
        </text>
      </svg>

      {hovered ? (
        <div className="rotation-tooltip">
          <strong>{hovered.sector}</strong> ({hovered.stock_count} stocks) ·{' '}
          <span className={`badge badge--${QUADRANT_BADGE_TONE[hovered.quadrant] || 'neutral'}`}>
            {hovered.quadrant}
          </span>{' '}
          · AI Avg Forecast:{' '}
          <strong>
            {hovered.avg_pred_ret_pct >= 0 ? '+' : ''}
            {hovered.avg_pred_ret_pct}%
          </strong>{' '}
          · Top Pick: <strong>{hovered.top_pick?.name}</strong>
        </div>
      ) : (
        <div className="rotation-tooltip rotation-tooltip--muted">
          Click any sector bubble to filter the 100-stock Alpha Leaderboard below.
        </div>
      )}
    </div>
  );
};

export default SectorRotationChart;
