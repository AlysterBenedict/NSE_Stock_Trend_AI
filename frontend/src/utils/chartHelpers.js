import { ColorType, CrosshairMode, LineStyle } from 'lightweight-charts';

const LIGHT_CHART_COLORS = {
  primary: '#456150',
  primaryContainer: '#5d7a68',
  historical: '#273141',
  areaTop: 'rgba(93, 122, 104, 0.26)',
  areaBottom: 'rgba(93, 122, 104, 0.0)',
  up: '#5d7a68',
  down: '#9c646d',
  tertiary: '#804c55',
  secondary: '#7f5536',
  sma: '#7f5536',
  volume: 'rgba(175, 206, 185, 0.85)',
  bandUpper: 'rgba(93, 122, 104, 0.55)',
  bandLower: 'rgba(156, 100, 109, 0.55)',
  text: '#5f6660',
  grid: 'rgba(18, 28, 44, 0.045)',
  crosshair: 'rgba(69, 97, 80, 0.35)',
  crosshairLabelBg: '#456150',
};

const DARK_CHART_COLORS = {
  primary: '#86c8a0',
  primaryContainer: '#6ca884',
  historical: '#cbd6e6',
  areaTop: 'rgba(134, 200, 160, 0.22)',
  areaBottom: 'rgba(134, 200, 160, 0.0)',
  up: '#79be95',
  down: '#dc8594',
  tertiary: '#e58b99',
  secondary: '#e2b184',
  sma: '#e2b184',
  volume: 'rgba(134, 200, 160, 0.34)',
  bandUpper: 'rgba(134, 200, 160, 0.58)',
  bandLower: 'rgba(229, 139, 153, 0.58)',
  text: '#95a3b8',
  grid: 'rgba(235, 240, 247, 0.055)',
  crosshair: 'rgba(134, 200, 160, 0.42)',
  crosshairLabelBg: '#2b4a3a',
};

export function getChartColors(theme = 'light') {
  return theme === 'dark' ? DARK_CHART_COLORS : LIGHT_CHART_COLORS;
}

/** Re-apply active theme colors across Lightweight Charts instances and their series. */
export function applyChartTheme(charts, series = {}, theme = 'light') {
  const nextColors = getChartColors(theme);
  const themedOpts = getLightChartOptions({}, theme);
  const chartStyle = {
    layout: themedOpts.layout,
    grid: themedOpts.grid,
    crosshair: themedOpts.crosshair,
  };

  charts.forEach((chart) => {
    if (chart) chart.applyOptions(chartStyle);
  });

  if (series.line) {
    series.line.applyOptions({
      lineColor: nextColors.primary,
      topColor: nextColors.areaTop,
      bottomColor: nextColors.areaBottom,
    });
  }
  if (series.candle) {
    series.candle.applyOptions({
      upColor: nextColors.up,
      downColor: nextColors.down,
      wickUpColor: nextColors.up,
      wickDownColor: nextColors.down,
    });
  }
  if (series.sma) {
    series.sma.applyOptions({ color: nextColors.sma });
  }
  if (series.volume) {
    series.volume.applyOptions({ color: nextColors.volume });
  }
}

export function getLightChartOptions(overrides = {}, theme = 'light') {
  const colors = getChartColors(theme);
  return {
    layout: {
      background: { type: ColorType.Solid, color: 'transparent' },
      textColor: colors.text,
      fontFamily: "'Plus Jakarta Sans', 'Inter', system-ui, sans-serif",
      fontSize: 11,
      attributionLogo: false,
    },
    grid: {
      vertLines: { color: colors.grid },
      horzLines: { color: colors.grid },
    },
    crosshair: {
      mode: CrosshairMode.Normal,
      vertLine: {
        color: colors.crosshair,
        style: LineStyle.Dashed,
        labelBackgroundColor: colors.crosshairLabelBg,
      },
      horzLine: {
        color: colors.crosshair,
        style: LineStyle.Dashed,
        labelBackgroundColor: colors.crosshairLabelBg,
      },
    },
    rightPriceScale: { borderVisible: false },
    timeScale: {
      borderVisible: false,
      timeVisible: false,
      fixLeftEdge: true,
      fixRightEdge: true,
    },
    ...overrides,
  };
}

export function bindResponsiveCharts(chartRefs, containerRefs, extraSelector = null) {
  const handleResize = () => {
    chartRefs.forEach((chartRef, idx) => {
      const container = containerRefs[idx]?.current;
      if (chartRef.current && container && container.clientWidth > 0) {
        chartRef.current.applyOptions({ width: container.clientWidth });
      }
    });
  };

  const resizeObserver = new ResizeObserver(handleResize);
  containerRefs.forEach((ref) => {
    if (ref.current) resizeObserver.observe(ref.current);
  });

  if (extraSelector) {
    const extraEl = document.querySelector(extraSelector);
    if (extraEl) resizeObserver.observe(extraEl);
  }

  window.addEventListener('resize', handleResize);

  return () => {
    window.removeEventListener('resize', handleResize);
    resizeObserver.disconnect();
    chartRefs.forEach((ref) => {
      if (ref.current) {
        try {
          ref.current.remove();
        } catch (err) {
          console.warn('Chart disposal warning:', err);
        }
        ref.current = null;
      }
    });
  };
}
