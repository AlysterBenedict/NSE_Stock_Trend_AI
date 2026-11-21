import React, { useEffect, useMemo, useState } from 'react';
import { Card, ChangeBadge, EmptyState, Icon, Loader, PageHeader, Stat } from '../../components/common/ui';
import { getMarketOverview } from '../../services/apiClient';
import { formatPrice, formatSigned } from '../../utils/formatters';

const FILTER_TABS = [
  { id: 'all', label: 'All' },
  { id: 'gainers', label: 'Gainers' },
  { id: 'losers', label: 'Losers' },
];

const OpenPositionsDashboard = () => {
  const [positions, setPositions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [sortConfig, setSortConfig] = useState({ key: 'pChange', direction: 'desc' });
  const [searchTerm, setSearchTerm] = useState('');
  const [directionFilter, setDirectionFilter] = useState('all');

  useEffect(() => {
    let isMounted = true;

    const fetchPositions = async () => {
      try {
        const data = await getMarketOverview();
        if (isMounted) {
          setPositions([...data].sort((a, b) => b.pChange - a.pChange));
          setLoading(false);
        }
      } catch (error) {
        console.error('Error fetching positions:', error);
        if (isMounted) setLoading(false);
      }
    };

    fetchPositions();
    return () => {
      isMounted = false;
    };
  }, []);

  const handleSort = (key) => {
    const direction = sortConfig.key === key && sortConfig.direction === 'desc' ? 'asc' : 'desc';
    setSortConfig({ key, direction });
  };

  const stats = useMemo(() => {
    if (!positions.length) return null;
    const gainers = positions.filter((p) => p.pChange > 0);
    const losers = positions.filter((p) => p.pChange < 0);
    const sortedByChange = [...positions].sort((a, b) => b.pChange - a.pChange);
    const maxAbs = Math.max(0.5, ...positions.map((p) => Math.abs(Number(p.pChange) || 0)));
    return {
      advancing: gainers.length,
      declining: losers.length,
      leader: sortedByChange[0],
      laggard: sortedByChange[sortedByChange.length - 1],
      maxAbs,
    };
  }, [positions]);

  const visibleRows = useMemo(() => {
    const q = searchTerm.trim().toLowerCase();
    const filtered = positions.filter((item) => {
      const matchesQuery = !q || item.name.toLowerCase().includes(q) || item.symbol.toLowerCase().includes(q);
      if (!matchesQuery) return false;
      if (directionFilter === 'gainers') return item.pChange > 0;
      if (directionFilter === 'losers') return item.pChange < 0;
      return true;
    });

    return [...filtered].sort((a, b) => {
      const av = a[sortConfig.key];
      const bv = b[sortConfig.key];
      if (typeof av === 'string' && typeof bv === 'string') {
        return sortConfig.direction === 'asc' ? av.localeCompare(bv) : bv.localeCompare(av);
      }
      if (av < bv) return sortConfig.direction === 'asc' ? -1 : 1;
      if (av > bv) return sortConfig.direction === 'asc' ? 1 : -1;
      return 0;
    });
  }, [positions, searchTerm, directionFilter, sortConfig]);

  const sortIcon = (key) => {
    if (sortConfig.key !== key) return 'unfold_more';
    return sortConfig.direction === 'asc' ? 'arrow_upward' : 'arrow_downward';
  };

  const rankBadgeClass = (index) => {
    if (index === 0) return 'rank-pill rank-pill--1';
    if (index === 1) return 'rank-pill rank-pill--2';
    if (index === 2) return 'rank-pill rank-pill--3';
    return 'rank-pill';
  };

  return (
    <div className="page-stack">
      <PageHeader
        eyebrow="NSE Daily Movers"
        icon="leaderboard"
        title="Market Positions"
        description="Ranked daily gainers and decliners across tracked NSE constituents. Click any column header to sort."
      />

      {stats && (
        <div className="kpi-grid kpi-grid--4">
          <Stat
            label="Advancing"
            value={stats.advancing}
            hint={`of ${positions.length} equities`}
            tone="up"
            icon="north_east"
          />
          <Stat
            label="Declining"
            value={stats.declining}
            hint={`of ${positions.length} equities`}
            tone="down"
            icon="south_east"
          />
          <Stat
            label="Top Gainer"
            value={`${formatSigned(stats.leader.pChange)}%`}
            hint={`${stats.leader.name} · ₹${formatPrice(stats.leader.price)}`}
            tone="up"
            icon="trending_up"
          />
          <Stat
            label="Top Decliner"
            value={`${formatSigned(stats.laggard.pChange)}%`}
            hint={`${stats.laggard.name} · ₹${formatPrice(stats.laggard.price)}`}
            tone="down"
            icon="trending_down"
          />
        </div>
      )}

      <Card className="table-card">
        <div className="table-toolbar">
          <div className="search-pill search-pill--compact">
            <Icon name="search" size={18} className="search-pill__icon" />
            <input
              type="text"
              className="search-pill__input"
              placeholder="Filter table by stock or ticker..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
            />
            {searchTerm && (
              <button
                type="button"
                className="search-pill__clear"
                onClick={() => setSearchTerm('')}
                aria-label="Clear filter"
              >
                <Icon name="close" size={16} />
              </button>
            )}
          </div>

          <div className="segmented" role="group" aria-label="Filter direction">
            {FILTER_TABS.map((t) => (
              <button
                key={t.id}
                type="button"
                className={`segmented__btn${directionFilter === t.id ? ' active' : ''}`}
                onClick={() => setDirectionFilter(t.id)}
              >
                {t.label}
              </button>
            ))}
          </div>
        </div>

        {loading ? (
          <Loader label="Loading market positions..." />
        ) : visibleRows.length === 0 ? (
          <EmptyState
            icon="filter_alt_off"
            title="No positions match this filter"
            description="Adjust the search text or switch back to All."
          />
        ) : (
          <div className="table-scroll">
            <table className="data-table">
              <thead>
                <tr>
                  <th className="col-rank" onClick={() => handleSort('pChange')}>
                    <span className="th-btn">
                      Rank <Icon name={sortIcon('pChange')} size={15} />
                    </span>
                  </th>
                  <th onClick={() => handleSort('name')}>
                    <span className="th-btn">
                      Stock <Icon name={sortIcon('name')} size={15} />
                    </span>
                  </th>
                  <th className="col-num" onClick={() => handleSort('price')}>
                    <span className="th-btn th-btn--right">
                      Current Price <Icon name={sortIcon('price')} size={15} />
                    </span>
                  </th>
                  <th className="col-num" onClick={() => handleSort('open')}>
                    <span className="th-btn th-btn--right">
                      Open <Icon name={sortIcon('open')} size={15} />
                    </span>
                  </th>
                  <th className="col-num" onClick={() => handleSort('close')}>
                    <span className="th-btn th-btn--right">
                      Prev Close <Icon name={sortIcon('close')} size={15} />
                    </span>
                  </th>
                  <th className="col-num" onClick={() => handleSort('pChange')}>
                    <span className="th-btn th-btn--right">
                      Gain / Loss % <Icon name={sortIcon('pChange')} size={15} />
                    </span>
                  </th>
                </tr>
              </thead>
              <tbody>
                {visibleRows.map((stock, index) => {
                  const pctAbs = Math.abs(Number(stock.pChange) || 0);
                  const barWidth = stats ? Math.min(100, Math.round((pctAbs / stats.maxAbs) * 100)) : 0;
                  const isUp = stock.pChange >= 0;

                  return (
                    <tr key={stock.symbol}>
                      <td className="col-rank">
                        <span className={rankBadgeClass(index)}>{index + 1}</span>
                      </td>
                      <td>
                        <div className="table-stock">
                          <span className="table-stock__name">{stock.name}</span>
                          <span className="table-stock__ticker">{stock.symbol.split('.')[0]}</span>
                        </div>
                      </td>
                      <td className="col-num table-price">₹{formatPrice(stock.price)}</td>
                      <td className="col-num table-muted">₹{formatPrice(stock.open)}</td>
                      <td className="col-num table-muted">₹{formatPrice(stock.close)}</td>
                      <td className="col-num">
                        <div className="table-change-cell">
                          <div className="magnitude-bar" aria-hidden="true">
                            <span
                              className={`magnitude-bar__fill ${isUp ? 'is-up' : 'is-down'}`}
                              style={{ width: `${barWidth}%` }}
                            />
                          </div>
                          <ChangeBadge value={stock.pChange} />
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </Card>
    </div>
  );
};

export default OpenPositionsDashboard;
