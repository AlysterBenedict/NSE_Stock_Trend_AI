import React, { useEffect, useMemo, useRef, useState } from 'react';
import { createChart, LineStyle } from 'lightweight-charts';
import SearchableSelect from '../../components/common/SearchableSelect';
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
import { PREDICTION_STOCK_OPTIONS, PRINCIPAL_PRESETS } from '../../constants/stocks';
import { useTheme } from '../../context/ThemeContext';
import {
  calculatePortfolioProjection,
  getAlphaRadar,
  optimizePortfolio,
} from '../../services/apiClient';
import { getChartColors, getLightChartOptions } from '../../utils/chartHelpers';
import { exportPortfolioBrief } from '../../utils/exportResearchBrief';
import { formatINR, formatSigned, isCancelledError } from '../../utils/formatters';
import CorrelationHeatmap from './CorrelationHeatmap';
import EfficientFrontierChart from './EfficientFrontierChart';

const riskTone = (score) => {
  const s = (score || '').toLowerCase();
  if (s.includes('low')) return 'up';
  if (s.includes('high')) return 'down';
  return 'secondary';
};

const OPT_MODES = [
  { value: 'ai_bl', label: 'AI Max Sharpe' },
  { value: 'min_var', label: 'Min Variance' },
  { value: 'risk_parity', label: 'Risk Parity' },
  { value: 'custom', label: 'Custom Sliders' },
];

const PortfolioReturnCalculator = ({ seedStocks = null }) => {
  const [labMode, setLabMode] = useState(seedStocks?.length >= 2 ? 'multi' : 'single');
  const [stock, setStock] = useState('');
  const [principal, setPrincipal] = useState(10000);
  const [targetDate, setTargetDate] = useState('');

  // Single-stock state
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');

  // Multi-stock Optimizer state
  const [basket, setBasket] = useState(
    seedStocks?.length >= 2 ? seedStocks.slice(0, 10) : ['Infosys', 'HDFC Bank', 'Reliance', 'Sun Pharma', 'L&T']
  );
  const [basketPicker, setBasketPicker] = useState('');
  const [optAlgoMode, setOptAlgoMode] = useState('ai_bl');
  const [customSliders, setCustomSliders] = useState({});
  const [optLoading, setOptLoading] = useState(false);
  const [optResult, setOptResult] = useState(null);
  const [optError, setOptError] = useState('');
  const [loadingTopPicks, setLoadingTopPicks] = useState(false);

  const chartContainerRef = useRef(null);
  const abortControllerRef = useRef(null);
  const { theme } = useTheme();
  const colors = getChartColors(theme);

  useEffect(() => {
    if (seedStocks && seedStocks.length >= 2) {
      setBasket(seedStocks.slice(0, 10));
      setLabMode('multi');
      setOptResult(null);
    }
  }, [seedStocks]);

  useEffect(() => () => abortControllerRef.current?.abort(), []);

  useEffect(() => {
    const d = new Date();
    d.setFullYear(d.getFullYear() + 1);
    setTargetDate(d.toISOString().split('T')[0]);
  }, []);

  const horizonHint = useMemo(() => {
    if (!targetDate) return '';
    const diffDays = Math.round((new Date(targetDate) - new Date()) / 86400000);
    if (!Number.isFinite(diffDays) || diffDays <= 0) return 'Select a future date';
    const months = Math.max(1, Math.round(diffDays / 30.4));
    return `${months} ${months === 1 ? 'month' : 'months'} forward horizon`;
  }, [targetDate]);

  const handleCalculate = async () => {
    if (!stock || principal <= 0 || !targetDate) {
      setError('Please fill all fields correctly.');
      return;
    }

    abortControllerRef.current?.abort();
    const controller = new AbortController();
    abortControllerRef.current = controller;

    setError('');
    setLoading(true);
    setResult(null);

    try {
      const data = await calculatePortfolioProjection(
        {
          stock_name: stock,
          principal: parseFloat(principal),
          future_date: targetDate,
        },
        controller.signal
      );
      if (controller.signal.aborted) return;
      setResult(data);
    } catch (err) {
      if (controller.signal.aborted || isCancelledError(err)) {
        return;
      }
      console.error(err);
      setError(err.response?.data?.error || 'Calculation failed. Try a different stock or date.');
    } finally {
      if (abortControllerRef.current === controller) {
        setLoading(false);
      }
    }
  };

  const handleAddBasketStock = (selected) => {
    setBasketPicker('');
    if (!selected || basket.includes(selected) || basket.length >= 10) return;
    setBasket((prev) => [...prev, selected]);
  };

  const handleRemoveBasketStock = (name) => {
    setBasket((prev) => prev.filter((s) => s !== name));
  };

  const handleLoadTopAiPicks = async () => {
    setLoadingTopPicks(true);
    setOptError('');
    try {
      const radar = await getAlphaRadar(21, false);
      const top5 = (radar?.stocks || []).slice(0, 5).map((s) => s.name);
      if (top5.length >= 2) {
        setBasket(top5);
        setOptResult(null);
      }
    } catch (err) {
      console.error('Could not load AI top picks:', err);
      setOptError('Failed to fetch top AI alpha picks.');
    } finally {
      setLoadingTopPicks(false);
    }
  };

  const handleOptimizePortfolio = async () => {
    if (basket.length < 2 || principal <= 0 || !targetDate) {
      setOptError('Select 2-10 stocks, positive principal, and a valid future date.');
      return;
    }

    abortControllerRef.current?.abort();
    const controller = new AbortController();
    abortControllerRef.current = controller;

    setOptError('');
    setOptLoading(true);
    setOptResult(null);

    try {
      const weightsArr =
        optAlgoMode === 'custom'
          ? basket.map((name) => Number(customSliders[name] ?? 20))
          : null;

      const data = await optimizePortfolio(
        {
          stocks: basket,
          principal: parseFloat(principal),
          future_date: targetDate,
          mode: optAlgoMode,
          custom_weights: weightsArr,
        },
        controller.signal
      );
      if (controller.signal.aborted) return;
      setOptResult(data);
    } catch (err) {
      if (controller.signal.aborted || isCancelledError(err)) return;
      console.error(err);
      setOptError(err.response?.data?.error || 'Portfolio optimization failed.');
    } finally {
      if (abortControllerRef.current === controller) {
        setOptLoading(false);
      }
    }
  };

  const activeChartResult = labMode === 'single' ? result : optResult;

  useEffect(() => {
    if (!activeChartResult || !chartContainerRef.current) return undefined;

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
      lineWidth: 2.5,
      title: 'Median (P50)',
    });

    const baselineSeries = chart.addLineSeries({
      color: colors.down,
      lineWidth: 1.5,
      lineStyle: LineStyle.Dotted,
      title: 'Principal',
    });

    if (activeChartResult.chart_data) {
      const areaData = activeChartResult.chart_data.map((d) => ({ time: d.time, value: d.value }));
      if (areaData.length > 0) {
        const numPrincipal = parseFloat(principal);
        const baselineData = areaData.map((d) => ({ time: d.time, value: numPrincipal }));
        areaSeries.setData(areaData);
        baselineSeries.setData(baselineData);
      }

      if (Array.isArray(activeChartResult.p90_chart_data) && activeChartResult.p90_chart_data.length > 0) {
        const p90Series = chart.addLineSeries({
          color: colors.up,
          lineWidth: 1.5,
          lineStyle: LineStyle.Dashed,
          title: 'Optimistic (P90)',
        });
        p90Series.setData(activeChartResult.p90_chart_data);
      }

      if (Array.isArray(activeChartResult.p10_chart_data) && activeChartResult.p10_chart_data.length > 0) {
        const p10Series = chart.addLineSeries({
          color: colors.secondary,
          lineWidth: 1.5,
          lineStyle: LineStyle.Dashed,
          title: 'Conservative (P10)',
        });
        p10Series.setData(activeChartResult.p10_chart_data);
      }

      chart.timeScale().fitContent();
    }

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
  }, [activeChartResult, principal, theme, colors]);

  const pctVsPrincipal = (val) => {
    const p = parseFloat(principal);
    if (!p || !Number.isFinite(Number(val))) return null;
    return ((Number(val) - p) / p) * 100;
  };

  return (
    <div className="page-stack">
      <PageHeader
        eyebrow="Stochastic Wealth & Portfolio Optimization"
        icon="account_balance_wallet"
        title="AI Portfolio Projector"
        description="Simulate 10,000 GARCH(1,1)-t paths on a single equity or optimize a correlated 2-10 NSE basket with Efficient Frontier & stress testing."
        actions={
          <div className="segmented" role="group" aria-label="Portfolio mode">
            <button
              type="button"
              className={`segmented__btn${labMode === 'single' ? ' active' : ''}`}
              onClick={() => setLabMode('single')}
            >
              Single-Stock Simulator
            </button>
            <button
              type="button"
              className={`segmented__btn${labMode === 'multi' ? ' active' : ''}`}
              onClick={() => setLabMode('multi')}
            >
              Multi-Stock AI Optimizer
            </button>
          </div>
        }
      />

      <div className="bento-grid">
        <div className="bento-aside">
          <Card className="control-panel">
            <CardHeader
              icon="tune"
              iconTone="primary"
              title={labMode === 'single' ? 'Simulation Parameters' : 'Portfolio Basket & Engine'}
              subtitle={labMode === 'single' ? 'Configure asset, principal, and exit date' : 'Optimize weights, frontier & correlated risk'}
            />

            <div className="form-stack">
              {labMode === 'single' ? (
                <Field label="Core Equity" hint="NSE">
                  <SearchableSelect
                    options={PREDICTION_STOCK_OPTIONS}
                    onChange={setStock}
                    placeholder="Select stock..."
                    value={stock}
                  />
                </Field>
              ) : (
                <>
                  <Field label="NSE Stock Basket (2–10)" hint={`${basket.length}/10 selected`}>
                    <SearchableSelect
                      options={PREDICTION_STOCK_OPTIONS.filter((o) => !basket.includes(o.value))}
                      onChange={handleAddBasketStock}
                      placeholder="Add equity to basket..."
                      value={basketPicker}
                    />
                    <div className="basket-chips">
                      {basket.map((nm) => (
                        <span key={nm} className="basket-chip">
                          <span>{nm}</span>
                          <button
                            type="button"
                            onClick={() => handleRemoveBasketStock(nm)}
                            aria-label={`Remove ${nm}`}
                          >
                            ×
                          </button>
                        </span>
                      ))}
                    </div>
                    <button
                      type="button"
                      className="btn btn--secondary btn--sm"
                      onClick={handleLoadTopAiPicks}
                      disabled={loadingTopPicks}
                    >
                      {loadingTopPicks ? <Spinner size="sm" /> : <Icon name="auto_awesome" size={15} />}
                      <span>Load Top 5 AI Alpha Picks</span>
                    </button>
                  </Field>

                  <Field label="Optimization Engine">
                    <div className="preset-row">
                      {OPT_MODES.map((mOpt) => (
                        <button
                          key={mOpt.value}
                          type="button"
                          className={`preset-chip${optAlgoMode === mOpt.value ? ' active' : ''}`}
                          onClick={() => setOptAlgoMode(mOpt.value)}
                        >
                          {mOpt.label}
                        </button>
                      ))}
                    </div>
                  </Field>

                  {optAlgoMode === 'custom' && basket.length > 0 && (
                    <Field label="Custom Allocation Sliders">
                      <div className="custom-sliders">
                        {basket.map((nm) => (
                          <div key={nm} className="slider-row">
                            <span>{nm}</span>
                            <input
                              type="range"
                              min="5"
                              max="100"
                              value={Number(customSliders[nm] ?? 20)}
                              onChange={(e) =>
                                setCustomSliders((prev) => ({ ...prev, [nm]: Number(e.target.value) }))
                              }
                            />
                            <strong>{Number(customSliders[nm] ?? 20)}</strong>
                          </div>
                        ))}
                      </div>
                    </Field>
                  )}
                </>
              )}

              <Field label="Principal Amount" hint="INR (₹)" htmlFor="portfolio-principal">
                <div className="input-shell">
                  <span className="input-shell__prefix">₹</span>
                  <input
                    id="portfolio-principal"
                    type="number"
                    min="100"
                    className="input input--with-prefix input--display"
                    value={principal}
                    onChange={(e) => setPrincipal(parseFloat(e.target.value) || 0)}
                  />
                </div>
                <div className="preset-row">
                  {PRINCIPAL_PRESETS.map((amt) => (
                    <button
                      key={amt}
                      type="button"
                      className={`preset-chip${Number(principal) === amt ? ' active' : ''}`}
                      onClick={() => setPrincipal(amt)}
                    >
                      ₹{amt >= 100000 ? `${amt / 100000}L` : `${amt / 1000}K`}
                    </button>
                  ))}
                </div>
              </Field>

              <Field label="Target Withdrawal Date" hint={horizonHint} htmlFor="portfolio-date">
                <input
                  id="portfolio-date"
                  type="date"
                  className="input"
                  value={targetDate}
                  onChange={(e) => setTargetDate(e.target.value)}
                />
              </Field>

              {labMode === 'single' ? (
                <button
                  type="button"
                  className="btn btn--primary btn--block"
                  onClick={handleCalculate}
                  disabled={loading || !stock}
                >
                  {loading ? <Spinner size="sm" /> : <Icon name="trending_up" size={18} />}
                  <span>{loading ? 'Simulating 10,000 Paths...' : 'Project Growth'}</span>
                </button>
              ) : (
                <button
                  type="button"
                  className="btn btn--primary btn--block"
                  onClick={handleOptimizePortfolio}
                  disabled={optLoading || basket.length < 2}
                >
                  {optLoading ? <Spinner size="sm" /> : <Icon name="auto_graph" size={18} />}
                  <span>{optLoading ? 'Optimizing & Simulating...' : 'Optimize & Simulate'}</span>
                </button>
              )}

              {(labMode === 'single' ? error : optError) && (
                <Callout tone="error" icon="error">
                  {labMode === 'single' ? error : optError}
                </Callout>
              )}
            </div>
          </Card>

          <Card className="note-card">
            <span className="icon-tile icon-tile--primary">
              <Icon name="calculate" size={18} />
            </span>
            <div>
              <h4 className="note-card__title">
                {labMode === 'single' ? 'Monte Carlo Calibration' : 'Cholesky-Correlated GARCH'}
              </h4>
              <p className="note-card__copy">
                {labMode === 'single'
                  ? 'Simulates 10,000 synthetic return trajectories using fat-tailed Student-t innovations to capture volatility clustering and downside tail risk.'
                  : 'Combines Global Panel expected returns with Ledoit-Wolf shrinkage covariance and correlated Student-t simulations across every equity in your basket.'}
              </p>
            </div>
          </Card>
        </div>

        <div className="bento-main">
          {labMode === 'single' && !result && (
            <Card className="result-card--state">
              <EmptyState
                icon="stacked_line_chart"
                title="Ready to Simulate Wealth Cone"
                description="Select an NSE stock, principal capital, and target date on the left to generate a 10,000-path probabilistic portfolio forecast."
              />
            </Card>
          )}

          {labMode === 'multi' && !optResult && (
            <Card className="result-card--state">
              <EmptyState
                icon="donut_large"
                title="Ready to Optimize Multi-Stock Basket"
                description="Choose 2–10 NSE equities (or auto-load Top 5 AI Alpha Picks) and an optimization engine to compute Efficient Frontier weights, correlation heatmap, and stress scenarios."
              />
            </Card>
          )}

          {labMode === 'single' && result && (
            <div className="page-stack">
              <div className="kpi-grid kpi-grid--3">
                <Card className="kpi-card">
                  <div className="kpi-card__head">
                    <span>Median Value (P50)</span>
                    <Icon name="account_balance" size={18} className="kpi-card__icon kpi-card__icon--primary" />
                  </div>
                  <div className="kpi-card__value">{formatINR(result.final_value, 0)}</div>
                  <div className="kpi-card__foot">
                    <ChangeBadge value={result.roi} />
                    <span>P(Profit): {result.probability_of_profit}%</span>
                  </div>
                </Card>

                <Card className="kpi-card">
                  <div className="kpi-card__head">
                    <span>Annualized Return (CAGR)</span>
                    <Icon name="auto_graph" size={18} className="kpi-card__icon kpi-card__icon--secondary" />
                  </div>
                  <div className="kpi-card__value">{result.cagr}%</div>
                  <div className="kpi-card__foot">
                    <span>
                      P10–P90: {formatINR(result.p10_value, 0)} – {formatINR(result.p90_value, 0)}
                    </span>
                  </div>
                </Card>

                <Card className="kpi-card">
                  <div className="kpi-card__head">
                    <span>Risk &amp; Ratios</span>
                    <Icon name="shield" size={18} className="kpi-card__icon kpi-card__icon--tertiary" />
                  </div>
                  <div className="kpi-card__tag-row">
                    <Badge tone={riskTone(result.risk_score)}>{result.risk_score}</Badge>
                    <span className="kpi-card__sharpe">Sharpe {result.sharpe_ratio}</span>
                  </div>
                  <div className="kpi-card__foot">
                    <span>Vol: {result.risk_value}%</span>
                    <span>95% VaR: {formatINR(result.var_95, 0)}</span>
                  </div>
                </Card>
              </div>

              <Card>
                <CardHeader
                  icon="show_chart"
                  iconTone="primary"
                  title="Monte Carlo Wealth Cone"
                  subtitle={`${stock} · ${result.volatility_model}`}
                  actions={
                    <div className="chart-legend-inline">
                      <span className="legend-dot-item">
                        <span className="legend-swatch legend-swatch--line" style={{ background: colors.primary }} />
                        P50 Median
                      </span>
                      <span className="legend-dot-item">
                        <span className="legend-swatch legend-swatch--dashed" style={{ borderColor: colors.up }} />
                        P90 Optimistic
                      </span>
                      <span className="legend-dot-item">
                        <span className="legend-swatch legend-swatch--dashed" style={{ borderColor: colors.secondary }} />
                        P10 Conservative
                      </span>
                      <span className="legend-dot-item">
                        <span className="legend-swatch legend-swatch--dotted" style={{ borderColor: colors.down }} />
                        Principal
                      </span>
                    </div>
                  }
                />
                <div className="chart-well">
                  <div ref={chartContainerRef} className="chart-canvas" style={{ height: 340 }} />
                </div>
              </Card>

              <Card>
                <CardHeader
                  icon="equalizer"
                  iconTone="secondary"
                  title="Outcome Distribution Spectrum"
                  subtitle={`Projected portfolio value at ${targetDate} across percentile bands`}
                />
                <div className="scenario-grid">
                  <div className="scenario-tile">
                    <div className="scenario-tile__head">
                      <span>Conservative (P10)</span>
                      <Badge tone="down">10th pct</Badge>
                    </div>
                    <div className="scenario-tile__value">{formatINR(result.p10_value, 0)}</div>
                    {pctVsPrincipal(result.p10_value) !== null && (
                      <ChangeBadge value={pctVsPrincipal(result.p10_value)} />
                    )}
                  </div>

                  <div className="scenario-tile scenario-tile--featured">
                    <div className="scenario-tile__head">
                      <span>Median Baseline (P50)</span>
                      <Badge tone="primary">50th pct</Badge>
                    </div>
                    <div className="scenario-tile__value">{formatINR(result.final_value, 0)}</div>
                    <ChangeBadge value={result.roi} />
                  </div>

                  <div className="scenario-tile">
                    <div className="scenario-tile__head">
                      <span>Optimistic (P90)</span>
                      <Badge tone="up">90th pct</Badge>
                    </div>
                    <div className="scenario-tile__value">{formatINR(result.p90_value, 0)}</div>
                    {pctVsPrincipal(result.p90_value) !== null && (
                      <ChangeBadge value={pctVsPrincipal(result.p90_value)} />
                    )}
                  </div>
                </div>
              </Card>
            </div>
          )}

          {labMode === 'multi' && optResult && (
            <div className="page-stack">
              <div className="kpi-grid kpi-grid--4">
                <Card className="kpi-card">
                  <div className="kpi-card__head">
                    <span>Median Value (P50)</span>
                    <Icon name="account_balance" size={18} className="kpi-card__icon kpi-card__icon--primary" />
                  </div>
                  <div className="kpi-card__value">{formatINR(optResult.final_value, 0)}</div>
                  <div className="kpi-card__foot">
                    <ChangeBadge value={optResult.roi} />
                    <span>P(Profit): {optResult.probability_of_profit}%</span>
                  </div>
                </Card>

                <Card className="kpi-card">
                  <div className="kpi-card__head">
                    <span>Expected CAGR</span>
                    <Icon name="auto_graph" size={18} className="kpi-card__icon kpi-card__icon--secondary" />
                  </div>
                  <div className="kpi-card__value">{optResult.cagr}%</div>
                  <div className="kpi-card__foot">
                    <span>Sharpe {optResult.sharpe_ratio}</span>
                    <span>Beta {optResult.portfolio_beta}</span>
                  </div>
                </Card>

                <Card className="kpi-card">
                  <div className="kpi-card__head">
                    <span>Portfolio Risk</span>
                    <Icon name="shield" size={18} className="kpi-card__icon kpi-card__icon--tertiary" />
                  </div>
                  <div className="kpi-card__value">{optResult.portfolio_vol_pct}%</div>
                  <div className="kpi-card__foot">
                    <span>Vol Saved: {optResult.diversification_benefit_pct}%</span>
                  </div>
                </Card>

                <Card className="kpi-card">
                  <div className="kpi-card__head">
                    <span>95% VaR / CVaR</span>
                    <Icon name="warning" size={18} className="kpi-card__icon kpi-card__icon--tertiary" />
                  </div>
                  <div className="kpi-card__value">{formatINR(optResult.var_95, 0)}</div>
                  <div className="kpi-card__foot">
                    <span>CVaR: {formatINR(optResult.cvar_95, 0)}</span>
                  </div>
                </Card>
              </div>

              <Card>
                <CardHeader
                  icon="pie_chart"
                  iconTone="primary"
                  title="Optimal Capital Allocation &amp; Risk Budget"
                  subtitle={`${optResult.allocations.length}-asset portfolio (${OPT_MODES.find((m) => m.value === optResult.mode)?.label || optResult.mode})`}
                  actions={
                    <div className="table-actions">
                      <button
                        type="button"
                        className="btn btn--secondary btn--sm"
                        onClick={() => exportPortfolioBrief(optResult, principal, targetDate, 'md')}
                        title="Download Markdown Portfolio Brief"
                      >
                        <Icon name="download" size={14} />
                        <span>.md Brief</span>
                      </button>
                      <button
                        type="button"
                        className="btn btn--secondary btn--sm"
                        onClick={() => exportPortfolioBrief(optResult, principal, targetDate, 'pdf')}
                        title="Print / Save PDF Portfolio Brief"
                      >
                        <Icon name="print" size={14} />
                        <span>Print PDF</span>
                      </button>
                    </div>
                  }
                />
                <div className="table-scroll">
                  <table className="data-table">
                    <thead>
                      <tr>
                        <th>Equity</th>
                        <th>Sector</th>
                        <th className="num">Weight</th>
                        <th className="num">Allocated (₹)</th>
                        <th className="num">Exp. Return</th>
                        <th className="num">Ann. Vol</th>
                        <th className="num">Risk Share</th>
                      </tr>
                    </thead>
                    <tbody>
                      {optResult.allocations.map((row) => (
                        <tr key={row.ticker}>
                          <td>
                            <div className="table-symbol">
                              <strong>{row.name}</strong>
                              <span>{row.ticker.split('.')[0]}</span>
                            </div>
                          </td>
                          <td>{row.sector}</td>
                          <td className="num"><strong>{row.weight_pct}%</strong></td>
                          <td className="num">{formatINR(row.allocated_inr, 0)}</td>
                          <td className="num">
                            <ChangeBadge value={row.expected_return_pct} />
                          </td>
                          <td className="num">{row.ann_vol_pct}%</td>
                          <td className="num">{row.risk_contribution_pct}%</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </Card>

              <Card>
                <CardHeader
                  icon="show_chart"
                  iconTone="primary"
                  title="Correlated Monte Carlo Wealth Cone"
                  subtitle={`${optResult.simulation_paths.toLocaleString('en-IN')} Cholesky GARCH(1,1)-t paths`}
                />
                <div className="chart-well">
                  <div ref={chartContainerRef} className="chart-canvas" style={{ height: 340 }} />
                </div>
              </Card>

              <div className="grid-2">
                <Card>
                  <CardHeader
                    icon="scatter_plot"
                    iconTone="primary"
                    title="Markowitz Efficient Frontier"
                    subtitle="AI Max Sharpe vs Min-Variance & Equal Weight"
                  />
                  <EfficientFrontierChart frontier={optResult.efficient_frontier} />
                </Card>

                <Card>
                  <CardHeader
                    icon="grid_on"
                    iconTone="secondary"
                    title="Pairwise Correlation Matrix"
                    subtitle="Flagging overlapping sector risk (>= 0.70)"
                  />
                  <CorrelationHeatmap matrix={optResult.correlation_matrix} />
                </Card>
              </div>

              {optResult.stress_tests?.length > 0 && (
                <Card>
                  <CardHeader
                    icon="bolt"
                    iconTone="tertiary"
                    title="Macro &amp; Historical Crisis Stress Replay"
                    subtitle="Estimated basket impact under historical drawdowns and market shocks"
                  />
                  <div className="stress-grid">
                    {optResult.stress_tests.map((st) => (
                      <div key={st.scenario} className="stress-tile">
                        <span className="stress-tile__label">{st.scenario}</span>
                        <div className="stress-tile__row">
                          <span className="stress-tile__impact is-down">{formatSigned(st.impact_pct)}%</span>
                          <span className="stress-tile__val">{formatINR(st.projected_value, 0)}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </Card>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default PortfolioReturnCalculator;
