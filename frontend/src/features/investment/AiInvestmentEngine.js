import React, { useEffect, useRef, useState } from 'react';
import { createChart, ColorType } from 'lightweight-charts';
import ReactMarkdown from 'react-markdown';
import {
  Badge,
  Callout,
  Card,
  CardHeader,
  ChangeBadge,
  EmptyState,
  Field,
  Icon,
  PageHeader,
  Spinner,
} from '../../components/common/ui';
import { INVESTMENT_ENGINE_STOCKS, PRINCIPAL_PRESETS } from '../../constants/stocks';
import { useTheme } from '../../context/ThemeContext';
import { runInvestmentEngine } from '../../services/apiClient';
import { getChartColors } from '../../utils/chartHelpers';
import { formatINR, isCancelledError } from '../../utils/formatters';

const METHODOLOGY_STEPS = [
  {
    step: '01',
    icon: 'query_stats',
    title: 'Stacking Ensemble Return',
    detail: 'Forecasts target-date return across 100 liquid NSE bluechips.',
  },
  {
    step: '02',
    icon: 'shield',
    title: 'GARCH Volatility Calibration',
    detail: 'Penalizes unstable variance and measures model consensus.',
  },
  {
    step: '03',
    icon: 'newspaper',
    title: 'FinBERT News Sentiment',
    detail: 'Integrates headline tone into a composite alpha score.',
  },
];

const EngineChart = ({ trendData, height = 140 }) => {
  const chartRef = useRef();
  const chartInstance = useRef();
  const { theme } = useTheme();

  useEffect(() => {
    if (!trendData?.dates || !chartRef.current) return undefined;

    const colors = getChartColors(theme);
    chartInstance.current = createChart(chartRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: 'transparent' },
        textColor: colors.text,
        fontFamily: "'Plus Jakarta Sans', 'Inter', sans-serif",
        fontSize: 10,
        attributionLogo: false,
      },
      grid: {
        vertLines: { visible: false },
        horzLines: { color: colors.grid },
      },
      width: chartRef.current.clientWidth || 260,
      height,
      timeScale: { visible: false },
      rightPriceScale: { visible: true, borderVisible: false },
      handleScroll: false,
      handleScale: false,
    });

    const areaSeries = chartInstance.current.addAreaSeries({
      lineColor: colors.primary,
      topColor: colors.areaTop,
      bottomColor: colors.areaBottom,
      lineWidth: 2,
      crosshairMarkerVisible: false,
    });

    const data = [];
    trendData.dates.forEach((d, i) => {
      const val = trendData.prices[i];
      if (val) data.push({ time: d, value: val });
    });

    areaSeries.setData(data);
    chartInstance.current.timeScale().fitContent();

    const handleResize = () => {
      if (chartInstance.current && chartRef.current && chartRef.current.clientWidth > 0) {
        chartInstance.current.applyOptions({ width: chartRef.current.clientWidth });
        chartInstance.current.timeScale().fitContent();
      }
    };
    const observer = new ResizeObserver(handleResize);
    observer.observe(chartRef.current);

    return () => {
      observer.disconnect();
      if (chartInstance.current) {
        chartInstance.current.remove();
        chartInstance.current = null;
      }
    };
  }, [trendData, height, theme]);

  return <div ref={chartRef} className="engine-mini-chart" style={{ height }} />;
};

const sentimentTone = (score) => {
  if (score >= 65) return 'up';
  if (score >= 45) return 'secondary';
  return 'down';
};

const AiInvestmentEngine = () => {
  const [principal, setPrincipal] = useState(10000);
  const [withdrawalDate, setWithdrawalDate] = useState('');
  const [status, setStatus] = useState('IDLE');
  const [completedStocks, setCompletedStocks] = useState([]);
  const [currentStock, setCurrentStock] = useState(null);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const [formWarning, setFormWarning] = useState('');
  const intervalRef = useRef(null);
  const abortControllerRef = useRef(null);

  useEffect(
    () => () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
      abortControllerRef.current?.abort();
    },
    []
  );

  const startProcessing = async () => {
    if (!withdrawalDate) {
      setFormWarning('Please select a target withdrawal date to run screening.');
      return;
    }

    setFormWarning('');
    setStatus('PROCESSING');
    setCompletedStocks([]);
    setCurrentStock(INVESTMENT_ENGINE_STOCKS[0]);
    setError(null);

    if (intervalRef.current) clearInterval(intervalRef.current);
    let idx = 0;
    intervalRef.current = setInterval(() => {
      idx += 1;
      if (idx < INVESTMENT_ENGINE_STOCKS.length) {
        setCompletedStocks((prev) => [INVESTMENT_ENGINE_STOCKS[idx - 1], ...prev]);
        setCurrentStock(INVESTMENT_ENGINE_STOCKS[idx]);
      } else if (idx === INVESTMENT_ENGINE_STOCKS.length) {
        setCompletedStocks((prev) => [
          INVESTMENT_ENGINE_STOCKS[INVESTMENT_ENGINE_STOCKS.length - 1],
          ...prev,
        ]);
        setCurrentStock('Finalizing AI Verdict...');
      }
    }, 1200);

    abortControllerRef.current?.abort();
    const controller = new AbortController();
    abortControllerRef.current = controller;

    try {
      const data = await runInvestmentEngine(
        {
          principal,
          withdrawal_date: withdrawalDate,
        },
        controller.signal
      );
      if (controller.signal.aborted) return;
      clearInterval(intervalRef.current);
      intervalRef.current = null;
      setResults(data);
      setStatus('COMPLETE');
    } catch (err) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
      if (controller.signal.aborted || isCancelledError(err)) {
        return;
      }
      console.error(err);
      setError(`Analysis Failed: ${err.response?.data?.error || err.message}`);
      setStatus('ERROR');
    }
  };

  const calculateTotal = (profitPct) => {
    const p = parseFloat(principal) || 0;
    const profit = p * (parseFloat(profitPct) / 100);
    return formatINR(p + profit, 0);
  };

  const progressPct = Math.min(
    100,
    Math.round((completedStocks.length / INVESTMENT_ENGINE_STOCKS.length) * 100)
  );

  return (
    <div className="page-stack">
      <PageHeader
        eyebrow="Multi-Factor NSE 100 Screening"
        icon="auto_awesome"
        title="AI Investment Engine"
        description="Screens 100 liquid NSE constituents using Stacking Ensemble forecasts, GARCH volatility risk calibration, and FinBERT news sentiment."
        actions={
          status === 'COMPLETE' && (
            <button
              type="button"
              className="btn btn--secondary"
              onClick={() => {
                setStatus('IDLE');
                setResults(null);
              }}
            >
              <Icon name="refresh" size={18} />
              New Search
            </button>
          )
        }
      />

      <div className="bento-grid">
        <div className="bento-aside">
          <Card className="control-panel">
            <CardHeader
              icon="tune"
              iconTone="primary"
              title="Investment Parameters"
              subtitle="Capital and target horizon"
            />

            <div className="form-stack">
              <Field label="Principal Amount" hint="INR (₹)" htmlFor="engine-principal">
                <div className="input-shell">
                  <span className="input-shell__prefix">₹</span>
                  <input
                    id="engine-principal"
                    type="number"
                    min="1000"
                    className="input input--with-prefix input--display"
                    value={principal}
                    onChange={(e) => setPrincipal(e.target.value)}
                    disabled={status === 'PROCESSING'}
                  />
                </div>
                <div className="preset-row">
                  {PRINCIPAL_PRESETS.map((amt) => (
                    <button
                      key={amt}
                      type="button"
                      className={`preset-chip${Number(principal) === amt ? ' active' : ''}`}
                      onClick={() => setPrincipal(amt)}
                      disabled={status === 'PROCESSING'}
                    >
                      ₹{amt >= 100000 ? `${amt / 100000}L` : `${amt / 1000}K`}
                    </button>
                  ))}
                </div>
              </Field>

              <Field label="Withdrawal Date" htmlFor="engine-date">
                <input
                  id="engine-date"
                  type="date"
                  className="input"
                  value={withdrawalDate}
                  onChange={(e) => {
                    setWithdrawalDate(e.target.value);
                    if (formWarning) setFormWarning('');
                  }}
                  min={new Date().toISOString().split('T')[0]}
                  disabled={status === 'PROCESSING'}
                />
              </Field>

              {formWarning && (
                <Callout tone="secondary" icon="info">
                  {formWarning}
                </Callout>
              )}

              <button
                type="button"
                className="btn btn--primary btn--block"
                onClick={startProcessing}
                disabled={status === 'PROCESSING'}
              >
                {status === 'PROCESSING' ? <Spinner size="sm" /> : <Icon name="auto_awesome" size={18} />}
                <span>
                  {status === 'PROCESSING'
                    ? 'Screening Equities...'
                    : status === 'COMPLETE'
                    ? 'Re-run Analysis'
                    : 'Start Analysis'}
                </span>
              </button>
            </div>
          </Card>

          {(status === 'PROCESSING' || status === 'COMPLETE') && (
            <section className="inverse-card">
              <div className="inverse-card__head">
                <div className="inverse-card__lead">
                  <span className="inverse-card__pulse">
                    <Icon name={status === 'COMPLETE' ? 'check_circle' : 'grain'} size={20} />
                  </span>
                  <div>
                    <h3 className="inverse-card__title">
                      {status === 'COMPLETE' ? 'Screening Complete' : currentStock}
                    </h3>
                    <span className="inverse-card__sub">
                      {status === 'COMPLETE'
                        ? 'NSE 100 Multi-Factor Pipeline'
                        : 'Volatility · Forecast · Sentiment'}
                    </span>
                  </div>
                </div>
                <span className="inverse-card__badge">
                  {status === 'COMPLETE' ? '100%' : `${progressPct}%`}
                </span>
              </div>

              <div className="inverse-progress">
                <div
                  className="inverse-progress__fill"
                  style={{ width: `${status === 'COMPLETE' ? 100 : progressPct}%` }}
                />
              </div>

              <div className="inverse-progress__meta">
                <span>
                  {status === 'COMPLETE' ? INVESTMENT_ENGINE_STOCKS.length : completedStocks.length} of{' '}
                  {INVESTMENT_ENGINE_STOCKS.length} evaluated
                </span>
                <span>{status === 'COMPLETE' ? 'Top 3 Ranked' : 'Active'}</span>
              </div>

              {completedStocks.length > 0 && (
                <div className="inverse-checklist">
                  {completedStocks.map((stock, i) => (
                    <div key={`${stock}-${i}`} className="inverse-checklist__row">
                      <Icon name="check_circle" size={15} className="inverse-checklist__icon" />
                      <span>{stock}</span>
                    </div>
                  ))}
                </div>
              )}
            </section>
          )}
        </div>

        <div className="bento-main">
          {status === 'IDLE' && (
            <Card className="engine-idle-card">
              <EmptyState
                icon="auto_awesome"
                title="Quantitative Equity Screening"
                description="Specify your principal capital and withdrawal date on the left to evaluate 100 NSE bluechips and rank the top three risk-adjusted candidates."
              />

              <div className="methodology-grid">
                {METHODOLOGY_STEPS.map((m) => (
                  <div key={m.step} className="methodology-tile">
                    <div className="methodology-tile__top">
                      <span className="icon-tile icon-tile--primary">
                        <Icon name={m.icon} size={18} />
                      </span>
                      <span className="methodology-tile__step">{m.step}</span>
                    </div>
                    <h4>{m.title}</h4>
                    <p>{m.detail}</p>
                  </div>
                ))}
              </div>
            </Card>
          )}

          {status === 'PROCESSING' && (
            <Card className="engine-processing-panel">
              <div className="processing-hero">
                <span className="processing-hero__orb">
                  <Spinner size="lg" />
                </span>
                <span className="eyebrow-inline">Live Multi-Factor Scoring</span>
                <h2 className="processing-hero__stock">{currentStock}</h2>
                <p className="processing-hero__copy">
                  Evaluating Stacking Ensemble return projections, GARCH volatility risk, and FinBERT headline
                  sentiment across NSE 100 constituents.
                </p>
                <div className="processing-hero__bar">
                  <div className="processing-hero__fill" style={{ width: `${progressPct}%` }} />
                </div>
                <span className="processing-hero__pct">{progressPct}% Complete</span>
              </div>
            </Card>
          )}

          {status === 'ERROR' && (
            <Card>
              <Callout tone="error" icon="error" title="Analysis Error">
                {error}
              </Callout>
              <div style={{ marginTop: 16 }}>
                <button type="button" className="btn btn--secondary" onClick={() => setStatus('IDLE')}>
                  <Icon name="refresh" size={18} />
                  Try Again
                </button>
              </div>
            </Card>
          )}

          {status === 'COMPLETE' && results && (
            <div className="page-stack">
              <div className="section-bar">
                <div>
                  <span className="eyebrow-inline">Ensemble Selection</span>
                  <h2 className="section-bar__title">Top 3 Investment Picks</h2>
                </div>
                <Badge tone="primary" icon="verified">
                  Principal {formatINR(parseFloat(principal) || 0, 0)}
                </Badge>
              </div>

              <div className="picks-grid">
                {results.top_stocks.map((stock, index) => (
                  <article key={stock.ticker} className={`card pick-card pick-card--${index + 1}`}>
                    <div className="pick-card__head">
                      <div>
                        <h3 className="pick-card__name">{stock.name}</h3>
                        <span className="pick-card__ticker">{stock.ticker}</span>
                      </div>
                      <span className={`rank-badge rank-badge--${index + 1}`}>Rank #{index + 1}</span>
                    </div>

                    <div className="pick-card__total">
                      <span className="pick-card__total-label">Est. Total Value</span>
                      <span className="pick-card__total-val">{calculateTotal(stock.profit_pct)}</span>
                    </div>

                    <dl className="pick-metrics">
                      <div className="pick-metrics__row">
                        <dt>Profit Forecast</dt>
                        <dd>
                          <ChangeBadge value={stock.profit_pct} />
                        </dd>
                      </div>
                      {stock.alpha_score !== undefined && (
                        <div className="pick-metrics__row">
                          <dt>Alpha Score</dt>
                          <dd className="pick-metrics__strong">{stock.alpha_score} / 100</dd>
                        </div>
                      )}
                      <div className="pick-metrics__row">
                        <dt>Model Consensus</dt>
                        <dd>{stock.consensus_score ?? Math.round(100 - stock.volatility)}%</dd>
                      </div>
                      <div className="pick-metrics__row">
                        <dt>Ann. Volatility</dt>
                        <dd>{stock.volatility}%</dd>
                      </div>
                      <div className="pick-metrics__row">
                        <dt>FinBERT Sentiment</dt>
                        <dd>
                          <Badge tone={sentimentTone(stock.sentiment_score)}>
                            {stock.sentiment_score} / 100
                          </Badge>
                        </dd>
                      </div>
                      <div className="pick-metrics__row">
                        <dt>NSE Rank</dt>
                        <dd>#{stock.nse_rank}</dd>
                      </div>
                    </dl>

                    <div className="pick-card__chart">
                      <EngineChart trendData={stock.trend_data} />
                    </div>
                  </article>
                ))}
              </div>

              <Card className="verdict-card">
                <CardHeader
                  icon="psychology"
                  iconTone="primary"
                  eyebrow="Quantitative Synthesis"
                  title="AI Investment Verdict"
                  subtitle="Cross-sectional rationale based on return, volatility, and sentiment scores"
                />
                <div className="markdown-body">
                  <ReactMarkdown>{results.ai_advice}</ReactMarkdown>
                </div>
              </Card>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default AiInvestmentEngine;
