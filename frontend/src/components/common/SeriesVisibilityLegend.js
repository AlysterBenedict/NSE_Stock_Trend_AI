import React from 'react';
import { useTheme } from '../../context/ThemeContext';
import { getChartColors } from '../../utils/chartHelpers';

const SeriesVisibilityLegend = ({ visibility, onToggle }) => {
  const { theme } = useTheme();
  const colors = getChartColors(theme);

  return (
    <div className="legend" role="group" aria-label="Series visibility">
      <button
        type="button"
        className={`legend-item${!visibility.sma ? ' hidden' : ''}`}
        onClick={() => onToggle('sma')}
        aria-pressed={visibility.sma}
      >
        <span className="legend-swatch legend-swatch--line" style={{ background: colors.sma }} />
        SMA 50
      </button>

      <button
        type="button"
        className={`legend-item${!visibility.volume ? ' hidden' : ''}`}
        onClick={() => onToggle('volume')}
        aria-pressed={visibility.volume}
      >
        <span className="legend-swatch legend-swatch--bar" style={{ background: colors.volume }} />
        Volume
      </button>
    </div>
  );
};

export default SeriesVisibilityLegend;
