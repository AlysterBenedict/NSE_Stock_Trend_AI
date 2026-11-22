import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  Badge,
  Callout,
  Card,
  CardHeader,
  ChangeBadge,
  EmptyState,
  Icon,
  Loader,
  PageHeader,
} from '../../components/common/ui';
import { MARKET_CATEGORIES } from '../../constants/stocks';
import { useWatchlist } from '../../context/WatchlistContext';
import { getAlphaRadar } from '../../services/apiClient';
import { formatPrice, formatSigned } from '../../utils/formatters';
import SectorRotationChart from './SectorRotationChart';

const HORIZON_OPTIONS = [
  { value: 1, label: '1D' },
  { value: 5, label: '5D' },
  { value: 10, label: '10D' },
  { value: 21, label: '21D' },
  { value: 63, label: '63D' },
];

const AlphaRadarDashboard = ({ onJumpToForecast, onAddToOptimizer }) => {
  const { isWatched, toggleWatch } = useWatchlist();
  const [horizon, setHorizon] = useState(21);
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [sectorFilter, setSectorFilter] = useState('All');
  const [sortKey, setSortKey] = useState('predicted_return_pct');
  const [sortDesc, setSortDesc] = useState(true);

  const abortRef = useRef(null);

  const fetchRadar = useCallback(async (targetHorizon, forceRefresh = false) => {
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    setLoading(true);
    setError(null);
    try {
      const res = await getAlphaRadar(targetHorizon, forceRefresh, controller.signal);
      if (controller.signal.aborted) return;
      setData(res);
    } catch (err) {
      if (controller.signal.aborted) return;
      console.error('Alpha Radar error:', err);
      setError(err.response?.data?.error || 'Could not compute live NSE 100 Alpha Radar.');
    } finally {
      if (abortRef.current === controller) {
        setLoading(false);
      }
    }
  }, []);

  useEffect(() => {
    fetchRadar(horizon, false);
    return () => abortRef.current?.abort();
  }, [horizon, fetchRadar]);

  const handleSort = (key) => {
    if (sortKey === key) {
      setSortDesc((d) => !d);
    } else {
      setSortKey(key);
      setSortDesc(true);
    }
  };

  const filteredStocks = useMemo(() => {
    const stocks = data?.stocks || [];
    const q = searchTerm.trim().toLowerCase();
    const list = stocks.filter((s) => {
      const matchSector = sectorFilter === 'All' || s.sector === sectorFilter;
      const matchQuery =
        !q || s.name.toLowerCase().includes(q) || s.ticker.toLowerCase().includes(q);
      return matchSector && matchQuery;
    });

    return [...list].sort((a, b) => {
      const av = Number(a[sortKey]) || 0;
      const bv = Number(b[sortKey]) || 0;
      return sortDesc ? bv - av : av - bv;
    });
  }, [data, searchTerm, sectorFilter, sortKey, sortDesc]);

  const topLeaders = useMemo(() => (data?.stocks || []).slice(0, 5), [data]);
  const laggards = useMemo(() => {
    const arr = data?.stocks || [];
    return arr.length > 5 ? [...arr].slice(-5).reverse() : [];
  }, [data]);

  const macro = data?.macro_regime;

  return (
    <div className="page-stack">
      <PageHeader
        eyebrow="Cross-Sectional AI Screening"
        icon="radar"
        title="NSE 100 Alpha Radar"
        description="Scores all 100 NSE bluechips in a single vectorized Global Panel pass with 7-model consensus, reward-to-risk quantile ratios, sector rotation quadrants, and live macro regime."
        actions={
          <div className="radar-controls-head">
            <div className="segmented segmented--compact" role="group" aria-label="Forecast horizon">
              {HORIZON_OPTIONS.map((opt) => (
                <button
                  key={opt.value}
                  type="button"
                  className={`segmented__btn${horizon === opt.value ? ' active' : ''}`}
                  onClick={() => setHorizon(opt.value)}
                >
                  {opt.label}
                </button>
              ))}
            </div>
            <button
              type="button"
              className="btn btn--secondary btn--sm"
              onClick={() => fetchRadar(horizon, true)}
              disabled={loading}
            >
              <Icon name="refresh" size={16} />
              <span>Refresh</span>
            </button>
          </div>
        }
      />

      {error && (
        <Callout tone="error" icon="error" title="Alpha Radar Unavailable">
          {error}
        </Callout>
      )}

      {!loading && data && !data.models_ready && (
        <Callout tone="secondary" icon="info" title="Factor-Only Fallback Mode">
          Global Panel trained models for h={horizon}D are not loaded on disk yet. Displaying live Momentum-30 &amp; cross-sectional factor rankings until training finishes.
        </Callout>
      )}

      {!loading && data && data.active_horizon !== data.requested_horizon && data.models_ready && (
        <Callout tone="secondary" icon="tune" title="Fallback Horizon Active">
          Showing h={data.active_horizon}D Global Panel scores (h={data.requested_horizon}D artifacts not trained yet).
        </Callout>
      )}

      {macro && (
        <Card className="macro-ribbon">
          <div className="macro-ribbon__lead">
            <Badge tone={macro.tone} icon="public">
              {macro.regime_label}
            </Badge>
            <span className="macro-ribbon__date">
              As of {data?.as_of || 'Latest Session'} · Horizon h={data?.active_horizon}D
            </span>
          </div>

          <div className="macro-ribbon__metrics">
            <div className="macro-chip">
              <span>India VIX</span>
              <strong>{macro.india_vix}</strong>
            </div>
            <div className="macro-chip">
              <span>Nifty 21D</span>
              <strong className={macro.nifty_21d_pct >= 0 ? 'is-up' : 'is-down'}>
                {formatSigned(macro.nifty_21d_pct)}%
              </strong>
            </div>
            <div className="macro-chip">
              <span>Bank / Nifty 5D</span>
              <strong className={macro.bank_nifty_rel_5d_pct >= 0 ? 'is-up' : 'is-down'}>
                {formatSigned(macro.bank_nifty_rel_5d_pct)}%
              </strong>
            </div>
            <div className="macro-chip">
              <span>IT / Nifty 5D</span>
              <strong className={macro.it_nifty_rel_5d_pct >= 0 ? 'is-up' : 'is-down'}>
                {formatSigned(macro.it_nifty_rel_5d_pct)}%
              </strong>
            </div>
            <div className="macro-chip">
              <span>USD/INR 5D</span>
              <strong>{formatSigned(macro.usdinr_5d_pct)}%</strong>
            </div>
            <div className="macro-chip">
              <span>Brent 5D</span>
              <strong>{formatSigned(macro.brent_5d_pct)}%</strong>
            </div>
          </div>
        </Card>
      )}

      {loading ? (
        <Card>
          <Loader label={`Scoring 100 NSE equities across h=${horizon}D Global Panel...`} />
        </Card>
      ) : (
        <>
          <div className="grid-2">
            <Card>
              <CardHeader
                icon="donut_large"
                iconTone="primary"
                title="NSE Sector Rotation Matrix"
                subtitle="21D relative momentum vs 63D trend leadership"
                actions={
                  sectorFilter !== 'All' && (
                    <button
                      type="button"
                      className="btn btn--secondary btn--sm"
                      onClick={() => setSectorFilter('All')}
                    >
                      Clear: {sectorFilter}
                    </button>
                  )
                }
              />
              <SectorRotationChart
                sectors={data?.sectors || []}
                activeSector={sectorFilter}
                onSelectSector={setSectorFilter}
              />
            </Card>

            <Card>
              <CardHeader
                icon="leaderboard"
                iconTone="secondary"
                title="Top Alpha Leaders vs Caution Laggards"
                subtitle={`Highest & lowest expected h=${data?.active_horizon || horizon}D cross-sectional alpha`}
                actions={
                  onAddToOptimizer &&
                  topLeaders.length >= 2 && (
                    <button
                      type="button"
                      className="btn btn--primary btn--sm"
                      onClick={() => onAddToOptimizer(topLeaders.map((s) => s.name))}
                    >
                      <Icon name="account_balance_wallet" size={15} />
                      <span>Optimize Top 5</span>
                    </button>
                  )
                }
              />

              <div className="leaders-split">
                <div>
                  <h5 className="leaders-split__heading is-up">Top 5 AI Long Leaders</h5>
                  <div className="leaders-list">
                    {topLeaders.map((s) => (
                      <button
                        key={s.ticker}
                        type="button"
                        className="leader-row"
                        onClick={() => onJumpToForecast?.(s.name)}
                      >
                        <div>
                          <strong>#{s.alpha_rank} {s.name}</strong>
                          <span>{s.sector} · R:R {s.rr_ratio}x</span>
                        </div>
                        <ChangeBadge value={s.predicted_return_pct} />
                      </button>
                    ))}
                  </div>
                </div>

                <div>
                  <h5 className="leaders-split__heading is-down">Bottom 5 Caution Watch</h5>
                  <div className="leaders-list">
                    {laggards.map((s) => (
                      <button
                        key={s.ticker}
                        type="button"
                        className="leader-row"
                        onClick={() => onJumpToForecast?.(s.name)}
                      >
                        <div>
                          <strong>#{s.alpha_rank} {s.name}</strong>
                          <span>{s.sector} · Vol {s.vol_ann_pct}%</span>
                        </div>
                        <ChangeBadge value={s.predicted_return_pct} />
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            </Card>
          </div>

          <Card>
            <CardHeader
              icon="table_chart"
              iconTone="primary"
              title="Full NSE 100 Alpha Leaderboard"
              subtitle={`Showing ${filteredStocks.length} of ${(data?.stocks || []).length} tracked equities`}
            />

            <div className="toolbar">
              <div className="search-field">
                <Icon name="search" size={18} className="search-field__icon" />
                <input
                  type="text"
                  className="input search-field__input"
                  placeholder="Filter by company or NSE ticker..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                />
                {searchTerm && (
                  <button
                    type="button"
                    className="icon-btn search-field__clear"
                    onClick={() => setSearchTerm('')}
                    aria-label="Clear search"
                  >
                    <Icon name="close" size={16} />
                  </button>
                )}
              </div>

              <div className="chip-bar">
                {MARKET_CATEGORIES.map((sec) => (
                  <button
                    key={sec}
                    type="button"
                    className={`filter-chip${sectorFilter === sec ? ' active' : ''}`}
                    onClick={() => setSectorFilter(sec)}
                  >
                    {sec}
                  </button>
                ))}
              </div>
            </div>

            {filteredStocks.length === 0 ? (
              <EmptyState
                icon="search_off"
                title="No matching equities"
                description="Try clearing the sector filter or searching for another NSE symbol."
              />
            ) : (
              <div className="table-scroll">
                <table className="data-table data-table--interactive">
                  <thead>
                    <tr>
                      <th onClick={() => handleSort('alpha_rank')}>Rank</th>
                      <th>Company</th>
                      <th className="num">Spot (₹)</th>
                      <th className="num" onClick={() => handleSort('predicted_return_pct')}>
                        AI Return ({data?.active_horizon}D)
                      </th>
                      <th className="num">90% Range</th>
                      <th className="num" onClick={() => handleSort('rr_ratio')}>R:R</th>
                      <th className="num" onClick={() => handleSort('consensus_pct')}>Consensus</th>
                      <th className="num" onClick={() => handleSort('mom_rank')}>Mom-30</th>
                      <th className="num" onClick={() => handleSort('vol_ann_pct')}>Ann. Vol</th>
                      <th className="num">Actions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {filteredStocks.map((s) => (
                      <tr key={s.ticker}>
                        <td>
                          <span className="rank-badge">{s.alpha_rank}</span>
                        </td>
                        <td>
                          <div className="table-symbol">
                            <strong>{s.name}</strong>
                            <span>{s.ticker.split('.')[0]} · {s.sector}</span>
                          </div>
                        </td>
                        <td className="num">
                          <div>₹{formatPrice(s.price)}</div>
                          <small className={s.day_change_pct >= 0 ? 'is-up' : 'is-down'}>
                            {formatSigned(s.day_change_pct)}%
                          </small>
                        </td>
                        <td className="num">
                          <ChangeBadge value={s.predicted_return_pct} />
                        </td>
                        <td className="num">
                          <small>
                            ₹{formatPrice(s.lower_price)} – ₹{formatPrice(s.upper_price)}
                          </small>
                        </td>
                        <td className="num">{s.rr_ratio}x</td>
                        <td className="num">
                          <Badge tone={s.consensus_direction === 'bullish' ? 'up' : 'down'}>
                            {s.bullish_models}/{s.total_models}
                          </Badge>
                        </td>
                        <td className="num">{s.mom_rank}</td>
                        <td className="num">{s.vol_ann_pct}%</td>
                        <td className="num">
                          <div className="table-actions">
                            <button
                              type="button"
                              className={`watch-star${isWatched(s.ticker) || isWatched(s.name) ? ' is-watched' : ''}`}
                              onClick={() => toggleWatch({ name: s.name, symbol: s.ticker })}
                              aria-label={isWatched(s.ticker) ? `Remove ${s.name} from watchlist` : `Add ${s.name} to watchlist`}
                              title={isWatched(s.ticker) ? 'Remove from watchlist' : 'Add to watchlist'}
                            >
                              <Icon name="star" size={16} fill={isWatched(s.ticker) || isWatched(s.name)} />
                            </button>
                            <button
                              type="button"
                              className="btn btn--secondary btn--sm"
                              onClick={() => onJumpToForecast?.(s.name)}
                              title={`Forecast ${s.name}`}
                            >
                              Forecast
                            </button>
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </Card>
        </>
      )}
    </div>
  );
};

export default AlphaRadarDashboard;
