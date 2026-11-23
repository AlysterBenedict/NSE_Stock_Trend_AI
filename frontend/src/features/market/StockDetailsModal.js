import React, { useEffect, useRef, useState } from 'react';
import { createChart } from 'lightweight-charts';
import { Badge, Callout, Card, CardHeader, Icon, Loader } from '../../components/common/ui';
import { useTheme } from '../../context/ThemeContext';
import { useWatchlist } from '../../context/WatchlistContext';
import { getStockMarketDetail } from '../../services/apiClient';
import { getChartColors, getLightChartOptions } from '../../utils/chartHelpers';
import { formatPrice, isCancelledError } from '../../utils/formatters';

const MetricItem = ({ label, value, format = null }) => {
  if (value === undefined || value === null) return null;
  let displayVal = value;
  const num = Number(value);

  if (format === 'currency' && Number.isFinite(num)) {
    displayVal = `₹${formatPrice(num)}`;
  } else if (format === 'percent' && Number.isFinite(num)) {
    displayVal = `${(num * 100).toFixed(2)}%`;
  } else if (format === 'large' && Number.isFinite(num)) {
    displayVal = `₹${(num / 10000000).toLocaleString('en-IN', { maximumFractionDigits: 2 })} Cr`;
  } else if (format === 'integer' && Number.isFinite(num)) {
    displayVal = Math.round(num).toLocaleString('en-IN');
  } else if (format === 'decimal' && Number.isFinite(num)) {
    displayVal = num.toFixed(2);
  }

  return (
    <div className="metric-tile">
      <span className="metric-tile__label">{label}</span>
      <span className="metric-tile__value">{displayVal}</span>
    </div>
  );
};

const StockDetailsModal = ({ ticker, name, onBack }) => {
  const [details, setDetails] = useState(null);
  const [loading, setLoading] = useState(true);
  const chartContainerRef = useRef();
  const { theme } = useTheme();
  const { isWatched, toggleWatch } = useWatchlist();

  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);

    const fetchDetails = async () => {
      try {
        const data = await getStockMarketDetail(ticker, controller.signal);
        if (!controller.signal.aborted) setDetails(data);
      } catch (error) {
        if (controller.signal.aborted || isCancelledError(error)) return;
        if (process.env.NODE_ENV !== 'production') {
          console.error('Error fetching stock details:', error);
        }
      } finally {
        if (!controller.signal.aborted) setLoading(false);
      }
    };

    fetchDetails();
    return () => {
      controller.abort();
    };
  }, [ticker]);

  useEffect(() => {
    if (!details || !details.chart || !chartContainerRef.current) return undefined;

    const colors = getChartColors(theme);
    const chart = createChart(
      chartContainerRef.current,
      getLightChartOptions(
        {
          width: chartContainerRef.current.clientWidth,
          height: 340,
        },
        theme
      )
    );

    const areaSeries = chart.addAreaSeries({
      lineColor: colors.primary,
      topColor: colors.areaTop,
      bottomColor: colors.areaBottom,
      lineWidth: 2,
    });

    const byTime = new Map();
    details.chart.forEach((pt) => {
      if (pt && pt.time && Number.isFinite(Number(pt.value))) {
        byTime.set(pt.time, Number(pt.value));
      }
    });
    const sortedData = Array.from(byTime.entries())
      .sort(([a], [b]) => a.localeCompare(b))
      .map(([time, value]) => ({ time, value }));
    areaSeries.setData(sortedData);
    chart.timeScale().fitContent();

    const handleResize = () => {
      if (chartContainerRef.current) {
        chart.applyOptions({ width: chartContainerRef.current.clientWidth });
      }
    };

    const observer = new ResizeObserver(handleResize);
    observer.observe(chartContainerRef.current);
    window.addEventListener('resize', handleResize);

    return () => {
      observer.disconnect();
      window.removeEventListener('resize', handleResize);
      chart.remove();
    };
  }, [details, theme]);

  if (loading) {
    return (
      <div className="page-stack">
        <div>
          <button type="button" className="btn btn--secondary" onClick={onBack}>
            <Icon name="arrow_back" size={18} />
            Back to Market
          </button>
        </div>
        <Card>
          <Loader label={`Loading analysis for ${name}...`} />
        </Card>
      </div>
    );
  }

  if (!details) {
    return (
      <div className="page-stack">
        <div>
          <button type="button" className="btn btn--secondary" onClick={onBack}>
            <Icon name="arrow_back" size={18} />
            Back to Market
          </button>
        </div>
        <Callout tone="error" icon="error" title="Unable to load equity details">
          Could not retrieve market data for {name} ({ticker}).
        </Callout>
      </div>
    );
  }

  const { info } = details;
  const watched = isWatched(ticker) || isWatched(name);
  const spotPrice = info.currentPrice || info.regularMarketPrice;
  const low52 = Number(info.fiftyTwoWeekLow);
  const high52 = Number(info.fiftyTwoWeekHigh);
  const hasRange = Number.isFinite(low52) && Number.isFinite(high52) && high52 > low52 && Number.isFinite(Number(spotPrice));
  const rangePos = hasRange ? Math.min(100, Math.max(0, ((Number(spotPrice) - low52) / (high52 - low52)) * 100)) : null;

  return (
    <div className="page-stack">
      <div className="detail-toolbar">
        <button type="button" className="btn btn--secondary" onClick={onBack}>
          <Icon name="arrow_back" size={18} />
          Back to Market
        </button>
        <button
          type="button"
          className={`btn ${watched ? 'btn--primary' : 'btn--secondary'}`}
          onClick={() => toggleWatch({ name, symbol: ticker })}
        >
          <Icon name="star" size={17} fill={watched} />
          <span>{watched ? 'Watching' : 'Add to Watchlist'}</span>
        </button>
      </div>

      <Card className="detail-hero">
        <div className="detail-hero__main">
          <div className="detail-hero__identity">
            <div className="detail-hero__badges">
              <Badge tone="primary">{ticker}</Badge>
              {info.sector && <Badge tone="neutral">{info.sector}</Badge>}
              {info.industry && <Badge tone="neutral">{info.industry}</Badge>}
            </div>
            <h1 className="detail-hero__name">{name}</h1>
          </div>

          <div className="detail-hero__price-block">
            <span className="detail-hero__price-label">Current Market Price</span>
            <div className="detail-hero__price">₹{formatPrice(spotPrice)}</div>
          </div>
        </div>

        {hasRange && (
          <div className="range-bar-card">
            <div className="range-bar__labels">
              <span>52W Low: ₹{formatPrice(low52)}</span>
              <span className="range-bar__title">52-Week Range Position ({Math.round(rangePos)}%)</span>
              <span>52W High: ₹{formatPrice(high52)}</span>
            </div>
            <div className="range-track">
              <div className="range-track__fill" style={{ width: `${rangePos}%` }} />
              <span className="range-track__thumb" style={{ left: `${rangePos}%` }} />
            </div>
          </div>
        )}
      </Card>

      <Card>
        <CardHeader
          icon="show_chart"
          title="Price Trajectory"
          subtitle="Historical closing price series"
        />
        <div className="chart-well">
          <div className="chart-canvas" style={{ height: 340 }} ref={chartContainerRef} />
        </div>
      </Card>

      <div className="grid-2">
        <Card>
          <CardHeader icon="account_balance" iconTone="primary" title="Key Financials" subtitle="Valuation and profitability" />
          <div className="metric-grid">
            <MetricItem label="Market Cap" value={info.marketCap} format="large" />
            <MetricItem label="P/E Ratio" value={info.trailingPE} format="decimal" />
            <MetricItem label="Forward P/E" value={info.forwardPE} format="decimal" />
            <MetricItem label="EPS (TTM)" value={info.trailingEps} format="decimal" />
            <MetricItem label="Book Value" value={info.bookValue} format="currency" />
            <MetricItem label="Dividend Yield" value={info.dividendYield} format="percent" />
            <MetricItem label="Return on Equity" value={info.returnOnEquity} format="percent" />
            <MetricItem label="Total Revenue" value={info.totalRevenue} format="large" />
            <MetricItem label="Profit Margins" value={info.profitMargins} format="percent" />
          </div>
        </Card>

        <Card>
          <CardHeader icon="analytics" iconTone="secondary" title="Price Statistics" subtitle="Moving averages and liquidity" />
          <div className="metric-grid">
            <MetricItem label="52 Week High" value={info.fiftyTwoWeekHigh} format="currency" />
            <MetricItem label="52 Week Low" value={info.fiftyTwoWeekLow} format="currency" />
            <MetricItem label="50-Day SMA" value={info.fiftyDayAverage} format="currency" />
            <MetricItem label="200-Day SMA" value={info.twoHundredDayAverage} format="currency" />
            <MetricItem label="Beta (Volatility)" value={info.beta} format="decimal" />
            <MetricItem label="Volume" value={info.volume} format="integer" />
            <MetricItem label="Avg Volume" value={info.averageVolume} format="integer" />
          </div>
        </Card>
      </div>

      {info.longBusinessSummary && (
        <Card>
          <CardHeader icon="business" iconTone="neutral" title="Company Profile" />
          <p className="prose-block">{info.longBusinessSummary}</p>
        </Card>
      )}
    </div>
  );
};

export default StockDetailsModal;
