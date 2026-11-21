import React, { useEffect, useRef, useState } from 'react';
import { createChart } from 'lightweight-charts';
import ChartSyncToggle from '../../components/common/ChartSyncToggle';
import SeriesVisibilityLegend from '../../components/common/SeriesVisibilityLegend';
import { Callout, Card, CardHeader, EmptyState, Loader, Stat } from '../../components/common/ui';
import { useTheme } from '../../context/ThemeContext';
import { getHistoricalData } from '../../services/apiClient';
import { applyChartTheme, bindResponsiveCharts, getChartColors, getLightChartOptions } from '../../utils/chartHelpers';
import { formatPrice, formatSigned } from '../../utils/formatters';

const FiftyTwoWeekAnalysisChart = ({ stockName }) => {
  const chartContainerRef1 = useRef();
  const chartContainerRef2 = useRef();
  const chartContainerRef3 = useRef();
  const chartInstance1 = useRef(null);
  const chartInstance2 = useRef(null);
  const chartInstance3 = useRef(null);
  const seriesRefs = useRef({ line: null, candle: null, sma: null, volume: null });

  const { theme } = useTheme();
  const themeRef = useRef(theme);
  themeRef.current = theme;

  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [metrics, setMetrics] = useState({ high: null, low: null, latest: null, periodReturn: null });
  const [visibility, setVisibility] = useState({ sma: true, volume: true });
  const visibilityRef = useRef(visibility);
  const [isSyncing, setIsSyncing] = useState(true);
  const isSyncingRef = useRef(true);

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
    applyChartTheme(
      [chartInstance1.current, chartInstance2.current, chartInstance3.current],
      seriesRefs.current,
      theme
    );
  }, [theme]);

  useEffect(() => {
    let cancelled = false;

    const fetchDataAndRender = async () => {
      setIsLoading(true);
      setError(null);
      try {
        const fullData = await getHistoricalData(stockName);
        if (cancelled) return;

        if (!fullData || !fullData.ohlc || fullData.ohlc.length === 0) {
          setError('No data available for this stock.');
          return;
        }

        const cutoffIndex = Math.max(0, fullData.ohlc.length - 252);
        const data = {
          ohlc: fullData.ohlc.slice(cutoffIndex),
          volume: fullData.volume.slice(cutoffIndex),
          sma: fullData.sma.slice(cutoffIndex),
        };

        let high = -Infinity;
        let low = Infinity;
        data.ohlc.forEach((d) => {
          if (d.high > high) high = d.high;
          if (d.low < low) low = d.low;
        });
        const firstClose = data.ohlc[0]?.close ?? null;
        const latestClose = data.ohlc[data.ohlc.length - 1]?.close ?? null;
        const periodReturn =
          firstClose && latestClose ? ((latestClose - firstClose) / firstClose) * 100 : null;
        setMetrics({ high, low, latest: latestClose, periodReturn });

        if (chartContainerRef1.current && chartContainerRef2.current && chartContainerRef3.current) {
          [chartInstance1, chartInstance2, chartInstance3].forEach((ref) => {
            if (ref.current) {
              try {
                ref.current.remove();
              } catch (e) {
                console.warn(e);
              }
              ref.current = null;
            }
          });
          seriesRefs.current = { line: null, candle: null, sma: null, volume: null };

          const activeTheme = themeRef.current;
          const activeColors = getChartColors(activeTheme);

          // 1. Primary Candlestick Chart (full width)
          const chart2 = createChart(
            chartContainerRef2.current,
            getLightChartOptions({ width: chartContainerRef2.current.clientWidth, height: 360 }, activeTheme)
          );
          chartInstance2.current = chart2;
          const candlestickSeries = chart2.addCandlestickSeries({
            upColor: activeColors.up,
            downColor: activeColors.down,
            borderVisible: false,
            wickUpColor: activeColors.up,
            wickDownColor: activeColors.down,
          });
          candlestickSeries.setData(data.ohlc);
          seriesRefs.current.candle = candlestickSeries;

          // 2. Close Price Area/Line Chart
          const chart1 = createChart(
            chartContainerRef1.current,
            getLightChartOptions({ width: chartContainerRef1.current.clientWidth, height: 300 }, activeTheme)
          );
          chartInstance1.current = chart1;
          const lineSeries = chart1.addAreaSeries({
            lineColor: activeColors.primary,
            topColor: activeColors.areaTop,
            bottomColor: activeColors.areaBottom,
            lineWidth: 2,
            title: 'Close Price',
          });
          lineSeries.setData(data.ohlc.map((d) => ({ time: d.time, value: d.close })));
          seriesRefs.current.line = lineSeries;

          // 3. Volume & SMA 50 Chart
          const chart3 = createChart(
            chartContainerRef3.current,
            getLightChartOptions({ width: chartContainerRef3.current.clientWidth, height: 300 }, activeTheme)
          );
          chartInstance3.current = chart3;

          const smaSeries = chart3.addLineSeries({
            color: activeColors.sma,
            lineWidth: 2,
            title: 'SMA 50',
          });
          smaSeries.setData(data.sma);
          seriesRefs.current.sma = smaSeries;
          smaSeries.applyOptions({ visible: visibilityRef.current.sma });

          const volumeSeries = chart3.addHistogramSeries({
            color: activeColors.volume,
            priceFormat: { type: 'volume' },
            priceScaleId: '',
            scaleMargins: { top: 0.65, bottom: 0 },
          });
          volumeSeries.setData(data.volume.map(({ color: _c, ...rest }) => rest));
          seriesRefs.current.volume = volumeSeries;
          volumeSeries.applyOptions({ visible: visibilityRef.current.volume });

          const charts = [chart1, chart2, chart3];
          charts.forEach((sourceChart, index) => {
            sourceChart.timeScale().subscribeVisibleLogicalRangeChange((range) => {
              if (isSyncingRef.current && range) {
                charts.forEach((targetChart, targetIndex) => {
                  if (index !== targetIndex) {
                    targetChart.timeScale().setVisibleLogicalRange(range);
                  }
                });
              }
            });
          });
        }
      } catch (err) {
        if (!cancelled) {
          console.error('Error fetching historical data:', err);
          setError('Failed to load chart data.');
        }
      } finally {
        if (!cancelled) setIsLoading(false);
      }
    };

    if (stockName) {
      fetchDataAndRender();
    } else {
      setIsLoading(false);
      setError(null);
      setMetrics({ high: null, low: null, latest: null, periodReturn: null });
    }

    const cleanupResize = bindResponsiveCharts(
      [chartInstance1, chartInstance2, chartInstance3],
      [chartContainerRef1, chartContainerRef2, chartContainerRef3]
    );

    return () => {
      cancelled = true;
      cleanupResize();
    };
  }, [stockName]);

  if (!stockName) {
    return (
      <Card className="result-card--state">
        <EmptyState
          icon="candlestick_chart"
          title="Ready to Inspect Trailing 252 Sessions"
          description="Choose an NSE stock from the selector above to inspect trailing 252-session price extremes, OHLC candlesticks, 50-day moving average, and volume."
        />
      </Card>
    );
  }

  const rangePos =
    metrics.high !== null &&
    metrics.low !== null &&
    metrics.latest !== null &&
    metrics.high > metrics.low
      ? Math.round(((metrics.latest - metrics.low) / (metrics.high - metrics.low)) * 100)
      : null;

  return (
    <div className="page-stack">
      {!isLoading && !error && metrics.high !== null && (
        <div className="kpi-grid kpi-grid--4">
          <Stat
            label="52-Week High"
            value={`₹${formatPrice(metrics.high)}`}
            hint="Trailing 252 sessions"
            tone="up"
            icon="north_east"
          />
          <Stat
            label="52-Week Low"
            value={`₹${formatPrice(metrics.low)}`}
            hint="Trailing 252 sessions"
            tone="down"
            icon="south_east"
          />
          <Stat
            label="Latest Close"
            value={`₹${formatPrice(metrics.latest)}`}
            hint={
              metrics.periodReturn !== null
                ? `${formatSigned(metrics.periodReturn)}% over 1Y`
                : 'Most recent session'
            }
            tone={metrics.periodReturn >= 0 ? 'up' : 'down'}
            icon="show_chart"
          />
          <Stat
            label="Range Position"
            value={rangePos !== null ? `${rangePos}%` : '—'}
            hint="Relative to 52W low–high"
            icon="linear_scale"
          />
        </div>
      )}

      {error && (
        <Callout tone="error" icon="error">
          {error}
        </Callout>
      )}

      <Card>
        <CardHeader
          icon="candlestick_chart"
          iconTone="primary"
          title={`Candlestick Price Action · ${stockName}`}
          subtitle="Daily open, high, low, and close over the last 52 weeks"
          actions={<ChartSyncToggle isSyncing={isSyncing} onToggle={toggleSync} />}
        />
        <div className="chart-well chart-well--relative">
          {isLoading && (
            <div className="chart-overlay">
              <Loader label="Loading 52-week candlestick data..." />
            </div>
          )}
          <div ref={chartContainerRef2} className="chart-canvas" style={{ height: 360 }} />
        </div>
      </Card>

      <div className="grid-2">
        <Card>
          <CardHeader
            icon="show_chart"
            iconTone="primary"
            title="Closing Price Curve"
            subtitle="Smoothed 1-year closing trajectory"
          />
          <div className="chart-well chart-well--relative">
            {isLoading && (
              <div className="chart-overlay">
                <Loader label="Loading close curve..." />
              </div>
            )}
            <div ref={chartContainerRef1} className="chart-canvas" style={{ height: 300 }} />
          </div>
        </Card>

        <Card>
          <CardHeader
            icon="bar_chart"
            iconTone="secondary"
            title="50-Day SMA & Volume"
            subtitle="Trend baseline and session turnover"
            actions={<SeriesVisibilityLegend visibility={visibility} onToggle={toggleSeries} />}
          />
          <div className="chart-well chart-well--relative">
            {isLoading && (
              <div className="chart-overlay">
                <Loader label="Loading indicators..." />
              </div>
            )}
            <div ref={chartContainerRef3} className="chart-canvas" style={{ height: 300 }} />
          </div>
        </Card>
      </div>
    </div>
  );
};

export default FiftyTwoWeekAnalysisChart;
