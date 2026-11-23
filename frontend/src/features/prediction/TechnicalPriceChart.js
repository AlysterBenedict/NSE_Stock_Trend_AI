import React, { useEffect, useRef, useState } from 'react';
import { createChart } from 'lightweight-charts';
import ChartSyncToggle from '../../components/common/ChartSyncToggle';
import SeriesVisibilityLegend from '../../components/common/SeriesVisibilityLegend';
import { Callout, Card, CardHeader, EmptyState, Icon, Loader } from '../../components/common/ui';
import { useTheme } from '../../context/ThemeContext';
import { getHistoricalData } from '../../services/apiClient';
import { applyChartTheme, bindResponsiveCharts, getChartColors, getLightChartOptions } from '../../utils/chartHelpers';
import { isCancelledError } from '../../utils/formatters';

const TechnicalPriceChart = ({ stockName }) => {
  const chartContainerRef1 = useRef();
  const chartContainerRef2 = useRef();
  const chartInstance1 = useRef(null);
  const chartInstance2 = useRef(null);
  const seriesRefs = useRef({ candle: null, sma: null, volume: null });

  const { theme } = useTheme();
  const themeRef = useRef(theme);
  themeRef.current = theme;
  const colors = getChartColors(theme);

  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [visibility, setVisibility] = useState({ sma: true, volume: true });
  const visibilityRef = useRef(visibility);
  const [isSyncing, setIsSyncing] = useState(true);
  const isSyncingRef = useRef(true);
  const [layoutMode, setLayoutMode] = useState('stacked');

  const toggleSeries = (key) => {
    setVisibility((prev) => ({ ...prev, [key]: !prev[key] }));
  };

  const toggleSync = () => {
    setIsSyncing((prev) => {
      const next = !prev;
      isSyncingRef.current = next;
      return next;
    });
  };

  useEffect(() => {
    visibilityRef.current = visibility;
    if (seriesRefs.current.sma) {
      seriesRefs.current.sma.applyOptions({ visible: visibility.sma });
    }
    if (seriesRefs.current.volume) {
      seriesRefs.current.volume.applyOptions({ visible: visibility.volume });
    }
  }, [visibility]);

  useEffect(() => {
    const h1 = layoutMode === 'stacked' ? 380 : 420;
    const h2 = layoutMode === 'stacked' ? 250 : 420;
    const frameId = requestAnimationFrame(() => {
      if (chartInstance1.current && chartContainerRef1.current) {
        chartInstance1.current.applyOptions({
          width: chartContainerRef1.current.clientWidth,
          height: h1,
        });
      }
      if (chartInstance2.current && chartContainerRef2.current) {
        chartInstance2.current.applyOptions({
          width: chartContainerRef2.current.clientWidth,
          height: h2,
        });
      }
    });
    return () => cancelAnimationFrame(frameId);
  }, [layoutMode]);

  useEffect(() => {
    applyChartTheme([chartInstance1.current, chartInstance2.current], seriesRefs.current, theme);
  }, [theme]);

  useEffect(() => {
    const controller = new AbortController();

    const fetchDataAndRender = async () => {
      setIsLoading(true);
      setError(null);
      try {
        const data = await getHistoricalData(stockName, controller.signal);
        if (controller.signal.aborted) return;

        if (!data || !data.ohlc || data.ohlc.length === 0) {
          setError('No data available for this stock.');
          return;
        }

        if (chartContainerRef1.current && chartContainerRef2.current) {
          [chartInstance1, chartInstance2].forEach((ref) => {
            if (ref.current) {
              try {
                ref.current.remove();
              } catch (e) {
                console.warn(e);
              }
              ref.current = null;
            }
          });
          seriesRefs.current = { candle: null, sma: null, volume: null };

          const activeTheme = themeRef.current;
          const activeColors = getChartColors(activeTheme);

          const chart1 = createChart(
            chartContainerRef1.current,
            getLightChartOptions(
              {
                width: chartContainerRef1.current.clientWidth,
                height: chartContainerRef1.current.clientHeight || 380,
              },
              activeTheme
            )
          );
          chartInstance1.current = chart1;

          const candlestickSeries = chart1.addCandlestickSeries({
            upColor: activeColors.up,
            downColor: activeColors.down,
            borderVisible: false,
            wickUpColor: activeColors.up,
            wickDownColor: activeColors.down,
          });
          candlestickSeries.setData(data.ohlc);
          seriesRefs.current.candle = candlestickSeries;

          const chart2 = createChart(
            chartContainerRef2.current,
            getLightChartOptions(
              {
                width: chartContainerRef2.current.clientWidth,
                height: chartContainerRef2.current.clientHeight || 250,
              },
              activeTheme
            )
          );
          chartInstance2.current = chart2;

          const smaSeries = chart2.addLineSeries({
            color: activeColors.sma,
            lineWidth: 2,
            title: 'SMA 50',
          });
          smaSeries.setData(data.sma);
          seriesRefs.current.sma = smaSeries;
          smaSeries.applyOptions({ visible: visibilityRef.current.sma });

          const volumeSeries = chart2.addHistogramSeries({
            color: activeColors.volume,
            priceFormat: { type: 'volume' },
            priceScaleId: '',
            scaleMargins: { top: 0.65, bottom: 0 },
          });
          volumeSeries.setData(data.volume.map(({ color: _c, ...rest }) => rest));
          seriesRefs.current.volume = volumeSeries;
          volumeSeries.applyOptions({ visible: visibilityRef.current.volume });

          const timeScale1 = chart1.timeScale();
          const timeScale2 = chart2.timeScale();
          timeScale1.subscribeVisibleLogicalRangeChange((range) => {
            if (isSyncingRef.current && range) timeScale2.setVisibleLogicalRange(range);
          });
          timeScale2.subscribeVisibleLogicalRangeChange((range) => {
            if (isSyncingRef.current && range) timeScale1.setVisibleLogicalRange(range);
          });
        }
      } catch (err) {
        if (!controller.signal.aborted && !isCancelledError(err)) {
          setError('Failed to load chart data.');
        }
      } finally {
        if (!controller.signal.aborted) setIsLoading(false);
      }
    };

    if (stockName) {
      fetchDataAndRender();
    } else {
      setIsLoading(false);
      setError(null);
    }

    const cleanupResize = bindResponsiveCharts(
      [chartInstance1, chartInstance2],
      [chartContainerRef1, chartContainerRef2],
      '.tech-charts-card'
    );

    return () => {
      controller.abort();
      cleanupResize();
    };
  }, [stockName]);

  if (!stockName) {
    return (
      <Card className="result-card--state">
        <EmptyState
          icon="candlestick_chart"
          title="Technical Analysis"
          description="Select an equity in Forecast Parameters above to load synchronized OHLC candlesticks, 50-day SMA, and trading volume."
        />
      </Card>
    );
  }

  return (
    <Card className="tech-charts-card">
      <CardHeader
        icon="candlestick_chart"
        iconTone="primary"
        title={`Technical Analysis · ${stockName}`}
        subtitle="Synchronized OHLC price action, 50-day moving average, and trading volume"
        actions={
          <div className="toolbar-actions">
            <div className="segmented segmented--compact" role="group" aria-label="Chart layout">
              <button
                type="button"
                className={`segmented__btn${layoutMode === 'stacked' ? ' active' : ''}`}
                onClick={() => setLayoutMode('stacked')}
                title="Align time axes vertically"
              >
                <Icon name="view_agenda" size={15} />
                Stacked
              </button>
              <button
                type="button"
                className={`segmented__btn${layoutMode === 'split' ? ' active' : ''}`}
                onClick={() => setLayoutMode('split')}
                title="Show side by side"
              >
                <Icon name="view_column" size={15} />
                Split
              </button>
            </div>
            <ChartSyncToggle isSyncing={isSyncing} onToggle={toggleSync} />
          </div>
        }
      />

      {error && (
        <Callout tone="error" icon="error">
          {error}
        </Callout>
      )}

      <div className={`tech-panes tech-panes--${layoutMode}`}>
        <div className="chart-pane">
          <div className="chart-pane__head">
            <span className="chart-pane__title">OHLC Candlestick</span>
            <div className="chart-legend-inline">
              <span className="legend-dot-item">
                <span className="legend-swatch legend-swatch--bar" style={{ background: colors.up }} />
                Up
              </span>
              <span className="legend-dot-item">
                <span className="legend-swatch legend-swatch--bar" style={{ background: colors.down }} />
                Down
              </span>
            </div>
          </div>
          <div className="chart-well chart-well--relative">
            {isLoading && (
              <div className="chart-overlay">
                <Loader label="Loading historical OHLC..." />
              </div>
            )}
            <div
              ref={chartContainerRef1}
              className="chart-canvas"
              style={{ height: layoutMode === 'stacked' ? 380 : 420 }}
            />
          </div>
        </div>

        <div className="chart-pane">
          <div className="chart-pane__head">
            <span className="chart-pane__title">50-Day SMA &amp; Volume</span>
            <SeriesVisibilityLegend visibility={visibility} onToggle={toggleSeries} />
          </div>
          <div className="chart-well chart-well--relative">
            {isLoading && (
              <div className="chart-overlay">
                <Loader label="Loading indicators..." />
              </div>
            )}
            <div
              ref={chartContainerRef2}
              className="chart-canvas"
              style={{ height: layoutMode === 'stacked' ? 250 : 420 }}
            />
          </div>
        </div>
      </div>
    </Card>
  );
};

export default TechnicalPriceChart;
