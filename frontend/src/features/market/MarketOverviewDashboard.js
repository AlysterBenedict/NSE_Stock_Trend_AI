import React, { useEffect, useMemo, useState } from 'react';
import { Badge, ChangeBadge, EmptyState, Icon, Loader, PageHeader } from '../../components/common/ui';
import { MARKET_CATEGORIES, SECTOR_MAP } from '../../constants/stocks';
import { getMarketOverview } from '../../services/apiClient';
import { formatSigned, toneOf } from '../../utils/formatters';
import LiveTickerTape from './LiveTickerTape';
import MarketStockGrid from './MarketStockGrid';
import StockDetailsModal from './StockDetailsModal';

const SORT_OPTIONS = [
  { id: 'name', label: 'A–Z' },
  { id: 'gainers', label: 'Gainers' },
  { id: 'losers', label: 'Losers' },
  { id: 'price', label: 'Price' },
];

const MarketOverviewDashboard = () => {
  const [marketData, setMarketData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedStock, setSelectedStock] = useState(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [activeCategory, setActiveCategory] = useState('All');
  const [sortBy, setSortBy] = useState('name');

  useEffect(() => {
    let isMounted = true;

    const fetchMarketData = async () => {
      try {
        const data = await getMarketOverview();
        if (isMounted) {
          setMarketData(data);
          setLoading(false);
        }
      } catch (error) {
        console.error('Error fetching market data:', error);
        if (isMounted) setLoading(false);
      }
    };

    fetchMarketData();
    const interval = setInterval(fetchMarketData, 60000);
    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, []);

  const summary = useMemo(() => {
    if (!marketData.length) return null;
    let advancing = 0;
    let declining = 0;
    let unchanged = 0;
    let sumPct = 0;
    let topGainer = marketData[0];
    let topDecliner = marketData[0];

    marketData.forEach((s) => {
      const pct = Number(s.pChange) || 0;
      sumPct += pct;
      if (pct > 0) advancing += 1;
      else if (pct < 0) declining += 1;
      else unchanged += 1;
      if (pct > (Number(topGainer.pChange) || 0)) topGainer = s;
      if (pct < (Number(topDecliner.pChange) || 0)) topDecliner = s;
    });

    const total = marketData.length;
    const advPct = Math.round((advancing / total) * 100);
    const avgPct = sumPct / total;

    return { total, advancing, declining, unchanged, advPct, avgPct, topGainer, topDecliner };
  }, [marketData]);

  const categoryCounts = useMemo(() => {
    const counts = { All: marketData.length };
    marketData.forEach((stock) => {
      const sector = SECTOR_MAP[stock.symbol] || 'Others';
      counts[sector] = (counts[sector] || 0) + 1;
    });
    return counts;
  }, [marketData]);

  const filteredData = useMemo(() => {
    const q = searchTerm.trim().toLowerCase();
    const filtered = marketData.filter((stock) => {
      const matchesSearch =
        !q || stock.name.toLowerCase().includes(q) || stock.symbol.toLowerCase().includes(q);
      const stockSector = SECTOR_MAP[stock.symbol] || 'Others';
      const matchesCategory = activeCategory === 'All' || stockSector === activeCategory;
      return matchesSearch && matchesCategory;
    });

    return [...filtered].sort((a, b) => {
      if (sortBy === 'gainers') return (b.pChange || 0) - (a.pChange || 0);
      if (sortBy === 'losers') return (a.pChange || 0) - (b.pChange || 0);
      if (sortBy === 'price') return (b.price || 0) - (a.price || 0);
      return a.name.localeCompare(b.name);
    });
  }, [marketData, searchTerm, activeCategory, sortBy]);

  return (
    <div className="page-stack">
      {!loading && marketData.length > 0 && <LiveTickerTape data={marketData} />}

      {selectedStock ? (
        <StockDetailsModal
          ticker={selectedStock.symbol}
          name={selectedStock.name}
          onBack={() => setSelectedStock(null)}
        />
      ) : (
        <>
          <PageHeader
            eyebrow="NSE Cash Market"
            icon="storefront"
            title="Market Overview"
            description="Live prices, intraday breadth, and sector breakdown across tracked NSE equities. Select any equity to inspect charts and fundamentals."
            actions={
              summary && (
                <div className="header-pill-group">
                  <Badge tone="neutral" icon="monitoring">
                    {summary.total} Equities
                  </Badge>
                  <Badge tone={toneOf(summary.avgPct)}>
                    Avg {formatSigned(summary.avgPct)}%
                  </Badge>
                </div>
              )
            }
          />

          {summary && (
            <section className="card breadth-ribbon" aria-label="Session breadth summary">
              <div className="breadth-ribbon__col breadth-ribbon__col--bar">
                <div className="breadth-bar__head">
                  <span className="breadth-bar__title">
                    <span className="status-dot" />
                    Market Breadth
                  </span>
                  <span className="breadth-bar__pct">{summary.advPct}% Advancing</span>
                </div>
                <div className="breadth-track" role="img" aria-label={`${summary.advancing} advancing, ${summary.declining} declining`}>
                  <div className="breadth-track__up" style={{ width: `${summary.advPct}%` }} />
                  <div className="breadth-track__down" style={{ width: `${100 - summary.advPct}%` }} />
                </div>
                <div className="breadth-bar__foot">
                  <span>{summary.advancing} Advancing</span>
                  <span>
                    {summary.declining} Declining
                    {summary.unchanged > 0 ? ` · ${summary.unchanged} Flat` : ''}
                  </span>
                </div>
              </div>

              <button
                type="button"
                className="breadth-ribbon__mover"
                onClick={() => setSelectedStock(summary.topGainer)}
              >
                <span className="icon-tile icon-tile--primary">
                  <Icon name="trending_up" size={18} />
                </span>
                <div className="breadth-mover__text">
                  <span className="breadth-mover__label">Session Leader</span>
                  <span className="breadth-mover__name">{summary.topGainer.name}</span>
                </div>
                <ChangeBadge value={summary.topGainer.pChange} />
              </button>

              <button
                type="button"
                className="breadth-ribbon__mover"
                onClick={() => setSelectedStock(summary.topDecliner)}
              >
                <span className="icon-tile icon-tile--tertiary">
                  <Icon name="trending_down" size={18} />
                </span>
                <div className="breadth-mover__text">
                  <span className="breadth-mover__label">Steepest Pullback</span>
                  <span className="breadth-mover__name">{summary.topDecliner.name}</span>
                </div>
                <ChangeBadge value={summary.topDecliner.pChange} />
              </button>
            </section>
          )}

          <section className="controls-stack">
            <div className="controls-row">
              <div className="search-pill">
                <Icon name="search" size={19} className="search-pill__icon" />
                <input
                  type="text"
                  className="search-pill__input"
                  placeholder="Search stocks by name or symbol (e.g., Infosys, INFY)..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                />
                {searchTerm && (
                  <button
                    type="button"
                    className="search-pill__clear"
                    onClick={() => setSearchTerm('')}
                    aria-label="Clear search"
                  >
                    <Icon name="close" size={16} />
                  </button>
                )}
              </div>

              <div className="segmented" role="group" aria-label="Sort equities">
                {SORT_OPTIONS.map((opt) => (
                  <button
                    key={opt.id}
                    type="button"
                    className={`segmented__btn${sortBy === opt.id ? ' active' : ''}`}
                    onClick={() => setSortBy(opt.id)}
                  >
                    {opt.label}
                  </button>
                ))}
              </div>
            </div>

            <div className="chip-bar" role="tablist" aria-label="Filter by sector">
              {MARKET_CATEGORIES.map((cat) => {
                const count = categoryCounts[cat];
                return (
                  <button
                    key={cat}
                    type="button"
                    role="tab"
                    aria-selected={activeCategory === cat}
                    className={`filter-chip${activeCategory === cat ? ' active' : ''}`}
                    onClick={() => setActiveCategory(cat)}
                  >
                    <span>{cat}</span>
                    {!loading && count !== undefined && <span className="filter-chip__count">{count}</span>}
                  </button>
                );
              })}
            </div>
          </section>

          {loading ? (
            <div className="card">
              <Loader label="Loading live NSE quotes..." />
            </div>
          ) : filteredData.length > 0 ? (
            <MarketStockGrid data={filteredData} onSelectStock={(stock) => setSelectedStock(stock)} />
          ) : (
            <div className="card">
              <EmptyState
                icon="search_off"
                title="No matching stocks"
                description="Try clearing the search query or switching to All sectors."
              >
                <button
                  type="button"
                  className="btn btn--secondary"
                  onClick={() => {
                    setSearchTerm('');
                    setActiveCategory('All');
                  }}
                >
                  Reset Filters
                </button>
              </EmptyState>
            </div>
          )}
        </>
      )}
    </div>
  );
};

export default MarketOverviewDashboard;
