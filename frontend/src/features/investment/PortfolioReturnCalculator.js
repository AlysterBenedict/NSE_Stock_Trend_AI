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
import { calculatePortfolioProjection } from '../../services/apiClient';
import { getChartColors, getLightChartOptions } from '../../utils/chartHelpers';
import { formatINR, isCancelledError } from '../../utils/formatters';

const riskTone = (score) => {
  const s = (score || '').toLowerCase();
  if (s.includes('low')) return 'up';
  if (s.includes('high')) return 'down';
  return 'secondary';
};

const PortfolioReturnCalculator = () => {
  const [stock, setStock] = useState('');
  const [principal, setPrincipal] = useState(10000);
  const [targetDate, setTargetDate] = useState('');

  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');

  const chartContainerRef = useRef(null);
  const abortControllerRef = useRef(null);
  const { theme } = useTheme();
  const colors = getChartColors(theme);

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

  useEffect(() => {
    if (!result || !chartContainerRef.current) return undefined;

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

    if (result.chart_data) {
      const areaData = result.chart_data.map((d) => ({ time: d.time, value: d.value }));
      if (areaData.length > 0) {
        const numPrincipal = parseFloat(principal);
        const baselineData = areaData.map((d) => ({ time: d.time, value: numPrincipal }));
        areaSeries.setData(areaData);
        baselineSeries.setData(baselineData);
      }

      if (Array.isArray(result.p90_chart_data) && result.p90_chart_data.length > 0) {
        const p90Series = chart.addLineSeries({
          color: colors.up,
          lineWidth: 1.5,
          lineStyle: LineStyle.Dashed,
          title: 'Optimistic (P90)',
        });
        p90Series.setData(result.p90_chart_data);
      }

      if (Array.isArray(result.p10_chart_data) && result.p10_chart_data.length > 0) {
        const p10Series = chart.addLineSeries({
          color: colors.secondary,
          lineWidth: 1.5,
          lineStyle: LineStyle.Dashed,
          title: 'Conservative (P10)',
        });
        p10Series.setData(result.p10_chart_data);
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
  }, [result, principal, theme, colors]);

  const pctVsPrincipal = (val) => {
    const p = parseFloat(principal);
    if (!p || !Number.isFinite(Number(val))) return null;
    return ((Number(val) - p) / p) * 100;
  };

  return (
    <div className="page-stack">
      <PageHeader
        eyebrow="Stochastic Wealth Simulation"
        icon="account_balance_wallet"
        title="AI Portfolio Projector"
        description="10,000-path GARCH(1,1) Student-t Monte Carlo wealth simulation with P10/P50/P90 outcome cones, Sharpe ratio, and 95% Value at Risk."
      />

      <div className="bento-grid">
        <div className="bento-aside">
          <Card className="control-panel">
            <CardHeader
              icon="tune"
              iconTone="primary"
              title="Simulation Parameters"
              subtitle="Configure asset, principal, and exit date"
            />

            <div className="form-stack">
              <Field label="Core Equity" hint="NSE">
                <SearchableSelect
                  options={PREDICTION_STOCK_OPTIONS}
                  onChange={setStock}
                  placeholder="Select stock..."
                  value={stock}
                />
              </Field>

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

              <button
                type="button"
                className="btn btn--primary btn--block"
                onClick={handleCalculate}
                disabled={loading || !stock}
              >
                {loading ? <Spinner size="sm" /> : <Icon name="trending_up" size={18} />}
                <span>{loading ? 'Simulating 10,000 Paths...' : 'Project Growth'}</span>
              </button>

              {error && (
                <Callout tone="error" icon="error">
                  {error}
                </Callout>
              )}
            </div>
          </Card>

          <Card className="note-card">
            <span className="icon-tile icon-tile--primary">
              <Icon name="calculate" size={18} />
            </span>
            <div>
              <h4 className="note-card__title">Monte Carlo Calibration</h4>
              <p className="note-card__copy">
                Simulates 10,000 synthetic return trajectories using fat-tailed Student-t innovations to capture
                volatility clustering and downside tail risk.
              </p>
            </div>
          </Card>
        </div>

        <div className="bento-main">
          {!result ? (
            <Card className="result-card--state">
              <EmptyState
                icon="stacked_line_chart"
                title="Ready to Simulate Wealth Cone"
                description="Select an NSE stock, principal capital, and target date on the left to generate a 10,000-path probabilistic portfolio forecast."
              />
            </Card>
          ) : (
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
        </div>
      </div>
    </div>
  );
};

export default PortfolioReturnCalculator;
