import React, { useEffect, useRef, useState } from 'react';
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
import { ALGORITHM_OPTIONS, PREDICTION_STOCK_OPTIONS, PRINCIPAL_PRESETS } from '../../constants/stocks';
import { useTheme } from '../../context/ThemeContext';
import { runBacktest } from '../../services/apiClient';
import { getChartColors, getLightChartOptions } from '../../utils/chartHelpers';
import { formatINR, formatSigned, isCancelledError } from '../../utils/formatters';

const STRATEGIES = [
  { value: 'top_n_long', label: 'Top-N AI Long Rotation (NSE 100)' },
  { value: 'long_short_quintile', label: 'Market-Neutral Long/Short Quintile' },
  { value: 'single_stock', label: 'Single-Stock AI Timing vs Buy & Hold' },
];

const HORIZONS = [
  { value: 1, label: '1D' },
  { value: 5, label: '5D' },
  { value: 10, label: '10D' },
  { value: 21, label: '21D' },
  { value: 63, label: '63D' },
];

const LOOKBACKS = [
  { value: 12, label: '12M' },
  { value: 24, label: '24M' },
  { value: 36, label: '36M' },
  { value: 0, label: 'Max' },
];

const BacktestLabDashboard = () => {
  const [strategy, setStrategy] = useState('top_n_long');
  const [horizon, setHorizon] = useState(21);
  const [algorithm, setAlgorithm] = useState('Ensemble');
  const [topN, setTopN] = useState(10);
  const [costBps, setCostBps] = useState(15);
  const [lookbackMonths, setLookbackMonths] = useState(24);
  const [initialCapital, setInitialCapital] = useState(100000);
  const [stockName, setStockName] = useState('Infosys');

  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');

  const equityChartRef = useRef(null);
  const ddChartRef = useRef(null);
  const abortRef = useRef(null);

  const { theme } = useTheme();
  const colors = getChartColors(theme);

  useEffect(() => () => abortRef.current?.abort(), []);

  const handleRunBacktest = async () => {
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    setLoading(true);
    setError('');
    try {
      const data = await runBacktest(
        {
          strategy,
          horizon,
          algorithm,
          top_n: topN,
          cost_bps: costBps,
          initial_capital: initialCapital,
          lookback_months: lookbackMonths,
          stock_name: strategy === 'single_stock' ? stockName : null,
        },
        controller.signal
      );
      if (controller.signal.aborted) return;
      setResult(data);
    } catch (err) {
      if (controller.signal.aborted || isCancelledError(err)) return;
      console.error('Backtest error:', err);
      setError(err.response?.data?.error || 'Could not run strategy backtest.');
    } finally {
      if (abortRef.current === controller) {
        setLoading(false);
      }
    }
  };

  useEffect(() => {
    if (!result || !equityChartRef.current || !ddChartRef.current) return undefined;

    const eqChart = createChart(
      equityChartRef.current,
      getLightChartOptions({ width: equityChartRef.current.clientWidth, height: 320 }, theme)
    );
    const stratSeries = eqChart.addLineSeries({
      color: colors.primary,
      lineWidth: 2.5,
      title: 'NeuroStock AI',
    });
    const benchSeries = eqChart.addLineSeries({
      color: colors.secondary,
      lineWidth: 2,
      lineStyle: LineStyle.Dashed,
      title: strategy === 'single_stock' ? 'Buy & Hold' : 'NSE Universe Avg',
    });

    stratSeries.setData(result.equity_curve.map((p) => ({ time: p.time, value: p.strategy })));
    benchSeries.setData(result.equity_curve.map((p) => ({ time: p.time, value: p.benchmark })));
    eqChart.timeScale().fitContent();

    const ddChart = createChart(
      ddChartRef.current,
      getLightChartOptions({ width: ddChartRef.current.clientWidth, height: 190 }, theme)
    );
    const ddStrat = ddChart.addAreaSeries({
      lineColor: colors.down,
      topColor: 'rgba(156, 100, 109, 0.25)',
      bottomColor: 'rgba(156, 100, 109, 0.02)',
      lineWidth: 2,
      title: 'Strategy DD%',
    });
    const ddBench = ddChart.addLineSeries({
      color: colors.text,
      lineWidth: 1.5,
      lineStyle: LineStyle.Dotted,
      title: 'Benchmark DD%',
    });
    ddStrat.setData(result.drawdown_curve.map((p) => ({ time: p.time, value: p.strategy_dd })));
    ddBench.setData(result.drawdown_curve.map((p) => ({ time: p.time, value: p.benchmark_dd })));
    ddChart.timeScale().fitContent();

    const handleResize = () => {
      if (equityChartRef.current) eqChart.applyOptions({ width: equityChartRef.current.clientWidth });
      if (ddChartRef.current) ddChart.applyOptions({ width: ddChartRef.current.clientWidth });
    };
    const obs = new ResizeObserver(handleResize);
    obs.observe(equityChartRef.current);
    obs.observe(ddChartRef.current);
    window.addEventListener('resize', handleResize);

    return () => {
      obs.disconnect();
      window.removeEventListener('resize', handleResize);
      eqChart.remove();
      ddChart.remove();
    };
  }, [result, strategy, theme, colors]);

  const m = result?.metrics;

  return (
    <div className="page-stack">
      <PageHeader
        eyebrow="Paper-Trade & Tear Sheet"
        icon="science"
        title="Walk-Forward Strategy Backtest Lab"
        description="Replay non-overlapping AI portfolio strategies with real NSE STT, brokerage, and slippage deductions against equal-weighted market and buy-and-hold benchmarks."
      />

      <div className="bento-grid">
        <div className="bento-aside">
          <Card className="control-panel">
            <CardHeader
              icon="tune"
              iconTone="primary"
              title="Strategy Parameters"
              subtitle="Non-overlapping rebalance rules"
            />

            <div className="form-stack">
              <Field label="Strategy Architecture">
                <select className="input" value={strategy} onChange={(e) => setStrategy(e.target.value)}>
                  {STRATEGIES.map((st) => (
                    <option key={st.value} value={st.value}>
                      {st.label}
                    </option>
                  ))}
                </select>
              </Field>

              {strategy === 'single_stock' && (
                <Field label="Target Equity" hint="NSE">
                  <SearchableSelect
                    options={PREDICTION_STOCK_OPTIONS}
                    value={stockName}
                    onChange={setStockName}
                    placeholder="Select stock..."
                  />
                </Field>
              )}

              {strategy === 'top_n_long' && (
                <Field label="Basket Size (Top N Picks)" hint={`${topN} stocks`}>
                  <div className="preset-row">
                    {[5, 10, 15].map((nVal) => (
                      <button
                        key={nVal}
                        type="button"
                        className={`preset-chip${topN === nVal ? ' active' : ''}`}
                        onClick={() => setTopN(nVal)}
                      >
                        Top {nVal}
                      </button>
                    ))}
                  </div>
                </Field>
              )}

              <Field label="Rebalance Horizon">
                <div className="preset-row">
                  {HORIZONS.map((hOpt) => (
                    <button
                      key={hOpt.value}
                      type="button"
                      className={`preset-chip${horizon === hOpt.value ? ' active' : ''}`}
                      onClick={() => setHorizon(hOpt.value)}
                    >
                      {hOpt.label}
                    </button>
                  ))}
                </div>
              </Field>

              <Field label="Forecasting Model">
                <select className="input" value={algorithm} onChange={(e) => setAlgorithm(e.target.value)}>
                  {ALGORITHM_OPTIONS.map((a) => (
                    <option key={a.value} value={a.value}>
                      {a.label}
                    </option>
                  ))}
                </select>
              </Field>

              <Field label="Evaluation Window">
                <div className="preset-row">
                  {LOOKBACKS.map((lb) => (
                    <button
                      key={lb.value}
                      type="button"
                      className={`preset-chip${lookbackMonths === lb.value ? ' active' : ''}`}
                      onClick={() => setLookbackMonths(lb.value)}
                    >
                      {lb.label}
                    </button>
                  ))}
                </div>
              </Field>

              <Field label="NSE STT + Brokerage + Slippage" hint={`${costBps} bps`}>
                <input
                  type="range"
                  min="0"
                  max="35"
                  step="5"
                  value={costBps}
                  onChange={(e) => setCostBps(Number(e.target.value))}
                />
              </Field>

              <Field label="Initial Capital" hint="INR (₹)">
                <div className="preset-row">
                  {PRINCIPAL_PRESETS.map((amt) => (
                    <button
                      key={amt}
                      type="button"
                      className={`preset-chip${Number(initialCapital) === amt ? ' active' : ''}`}
                      onClick={() => setInitialCapital(amt)}
                    >
                      ₹{amt >= 100000 ? `${amt / 100000}L` : `${amt / 1000}K`}
                    </button>
                  ))}
                </div>
              </Field>

              <button
                type="button"
                className="btn btn--primary btn--block"
                onClick={handleRunBacktest}
                disabled={loading}
              >
                {loading ? <Spinner size="sm" /> : <Icon name="play_arrow" size={18} />}
                <span>{loading ? 'Replaying Strategy...' : 'Run Backtest'}</span>
              </button>

              {error && (
                <Callout tone="error" icon="error">
                  {error}
                </Callout>
              )}
            </div>
          </Card>
        </div>

        <div className="bento-main">
          {!result ? (
            <Card className="result-card--state">
              <EmptyState
                icon="science"
                title="Ready to Paper-Trade AI Signals"
                description="Choose a rebalance horizon, strategy, and realistic NSE transaction cost on the left to inspect cumulative equity, drawdown, Sharpe, and historical picks."
              />
            </Card>
          ) : (
            <div className="page-stack">
              <div className="backtest-mode-bar">
                <Badge
                  tone={result.replay_mode === 'strict_oos_holdout' ? 'up' : 'secondary'}
                  icon="verified"
                >
                  {result.replay_mode === 'strict_oos_holdout'
                    ? 'Strict Unseen Holdout Replay (20% Test Split)'
                    : 'Live Panel Walk-Forward Replay'}
                </Badge>
                <span>
                  {m.num_rebalances} non-overlapping rebalances · Avg turnover {m.avg_turnover_pct}%
                </span>
              </div>

              <div className="kpi-grid kpi-grid--4">
                <Card className="kpi-card">
                  <div className="kpi-card__head">
                    <span>Net Portfolio Value</span>
                    <Icon name="account_balance" size={18} className="kpi-card__icon kpi-card__icon--primary" />
                  </div>
                  <div className="kpi-card__value">{formatINR(m.final_capital, 0)}</div>
                  <div className="kpi-card__foot">
                    <ChangeBadge value={m.total_return_pct} />
                    <span>CAGR {m.cagr_pct}%</span>
                  </div>
                </Card>

                <Card className="kpi-card">
                  <div className="kpi-card__head">
                    <span>Net Alpha vs Benchmark</span>
                    <Icon name="trending_up" size={18} className="kpi-card__icon kpi-card__icon--secondary" />
                  </div>
                  <div className={`kpi-card__value ${m.alpha_pct >= 0 ? 'is-up' : 'is-down'}`}>
                    {formatSigned(m.alpha_pct)}%
                  </div>
                  <div className="kpi-card__foot">
                    <span>Bench {formatSigned(m.benchmark_return_pct)}%</span>
                  </div>
                </Card>

                <Card className="kpi-card">
                  <div className="kpi-card__head">
                    <span>Sharpe &amp; Sortino</span>
                    <Icon name="equalizer" size={18} className="kpi-card__icon kpi-card__icon--primary" />
                  </div>
                  <div className="kpi-card__value">{m.sharpe_ratio}</div>
                  <div className="kpi-card__foot">
                    <span>Sortino {m.sortino_ratio}</span>
                    <span>Win {m.win_rate_pct}%</span>
                  </div>
                </Card>

                <Card className="kpi-card">
                  <div className="kpi-card__head">
                    <span>Max Drawdown</span>
                    <Icon name="shield" size={18} className="kpi-card__icon kpi-card__icon--tertiary" />
                  </div>
                  <div className="kpi-card__value is-down">-{m.max_drawdown_pct}%</div>
                  <div className="kpi-card__foot">
                    <span>Bench -{m.benchmark_max_drawdown_pct}%</span>
                  </div>
                </Card>
              </div>

              <Card>
                <CardHeader
                  icon="show_chart"
                  iconTone="primary"
                  title="Cumulative Equity Curve"
                  subtitle={`Net of ${costBps} bps round-trip NSE costs (${algorithm} · h=${result.horizon}D)`}
                />
                <div className="chart-well">
                  <div ref={equityChartRef} className="chart-canvas" style={{ height: 320 }} />
                </div>
              </Card>

              <Card>
                <CardHeader
                  icon="waterfall_chart"
                  iconTone="tertiary"
                  title="Underwater Drawdown (%)"
                  subtitle="Peak-to-trough capital drawdown comparison"
                />
                <div className="chart-well">
                  <div ref={ddChartRef} className="chart-canvas" style={{ height: 190 }} />
                </div>
              </Card>

              {result.recent_rebalances?.length > 0 && (
                <Card>
                  <CardHeader
                    icon="history"
                    iconTone="secondary"
                    title="Recent Rebalance Audit Log"
                    subtitle="Last 8 non-overlapping periods and top AI selections"
                  />
                  <div className="table-scroll">
                    <table className="data-table">
                      <thead>
                        <tr>
                          <th>Rebalance Date</th>
                          <th>Top Holdings</th>
                          <th className="num">Turnover</th>
                          <th className="num">Strategy Return</th>
                          <th className="num">Benchmark</th>
                        </tr>
                      </thead>
                      <tbody>
                        {result.recent_rebalances.map((rb) => (
                          <tr key={rb.date}>
                            <td>{rb.date}</td>
                            <td>{(rb.top_holdings || []).join(', ')}</td>
                            <td className="num">{rb.turnover_pct}%</td>
                            <td className="num">
                              <ChangeBadge value={rb.period_return_pct} />
                            </td>
                            <td className="num">{formatSigned(rb.benchmark_return_pct)}%</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
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

export default BacktestLabDashboard;
