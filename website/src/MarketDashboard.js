import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css';
import TickerTape from './TickerTape';
import StockGrid from './StockGrid';
import StockDetail from './StockDetail';

// Static Sector Mapping for Nifty 50
const SECTOR_MAP = {
    'ADANIENT.NS': 'Commodities', 'ADANIPORTS.NS': 'Infra', 'APOLLOHOSP.NS': 'Healthcare', 'ASIANPAINT.NS': 'Consumer',
    'AXISBANK.NS': 'Financials', 'BAJAJ-AUTO.NS': 'Auto', 'BAJFINANCE.NS': 'Financials', 'BAJAJFINSV.NS': 'Financials',
    'BHARTIARTL.NS': 'Telecom', 'BPCL.NS': 'Energy', 'BRITANNIA.NS': 'Consumer', 'CIPLA.NS': 'Healthcare',
    'COALINDIA.NS': 'Energy', 'DIVISLAB.NS': 'Healthcare', 'DRREDDY.NS': 'Healthcare', 'EICHERMOT.NS': 'Auto',
    'GRASIM.NS': 'Commodities', 'HCLTECH.NS': 'IT', 'HDFCBANK.NS': 'Financials', 'HDFCLIFE.NS': 'Financials',
    'HEROMOTOCO.NS': 'Auto', 'HINDALCO.NS': 'Commodities', 'HINDUNILVR.NS': 'Consumer', 'ICICIBANK.NS': 'Financials',
    'INDUSINDBK.NS': 'Financials', 'INFY.NS': 'IT', 'ITC.NS': 'Consumer', 'JSWSTEEL.NS': 'Commodities',
    'KOTAKBANK.NS': 'Financials', 'LT.NS': 'Infra', 'LTIM.NS': 'IT', 'M&M.NS': 'Auto', 'MARUTI.NS': 'Auto',
    'NESTLEIND.NS': 'Consumer', 'NTPC.NS': 'Energy', 'ONGC.NS': 'Energy', 'POWERGRID.NS': 'Energy',
    'RELIANCE.NS': 'Energy', 'SBILIFE.NS': 'Financials', 'SBIN.NS': 'Financials', 'SUNPHARMA.NS': 'Healthcare',
    'TATACONSUM.NS': 'Consumer', 'TATAMOTORS.NS': 'Auto', 'TATASTEEL.NS': 'Commodities', 'TCS.NS': 'IT',
    'TECHM.NS': 'IT', 'TITAN.NS': 'Consumer', 'ULTRACEMCO.NS': 'Commodities', 'UPL.NS': 'Commodities', 'WIPRO.NS': 'IT'
};

const CATEGORIES = ['All', 'Financials', 'IT', 'Auto', 'Energy', 'Consumer', 'Healthcare', 'Commodities', 'Infra', 'Telecom'];

const MarketDashboard = () => {
    const [marketData, setMarketData] = useState([]);
    const [loading, setLoading] = useState(true);
    const [selectedStock, setSelectedStock] = useState(null);
    const [searchTerm, setSearchTerm] = useState('');
    const [activeCategory, setActiveCategory] = useState('All');

    useEffect(() => {
        fetchMarketData();
        const interval = setInterval(fetchMarketData, 60000);
        return () => clearInterval(interval);
    }, []);

    const fetchMarketData = async () => {
        try {
            const response = await axios.get('http://127.0.0.1:5000/get-market-data');
            setMarketData(response.data);
            setLoading(false);
        } catch (error) {
            console.error("Error fetching market data:", error);
            setLoading(false);
        }
    };

    // Filter Logic
    const filteredData = marketData.filter(stock => {
        const matchesSearch = stock.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
            stock.symbol.toLowerCase().includes(searchTerm.toLowerCase());
        const stockSector = SECTOR_MAP[stock.symbol] || 'Others';
        const matchesCategory = activeCategory === 'All' || stockSector === activeCategory;

        return matchesSearch && matchesCategory;
    });

    return (
        <div className="market-dashboard">
            {/* 1. Ticker Tape (Always Visible at Top) */}
            {!loading && marketData.length > 0 && (
                <TickerTape data={marketData} />
            )}

            {/* 2. Main Content Area */}
            <div className="market-content">

                {selectedStock ? (
                    /* Detail View */
                    <StockDetail
                        ticker={selectedStock.symbol}
                        name={selectedStock.name}
                        onBack={() => setSelectedStock(null)}
                    />
                ) : (
                    /* Grid View */
                    <div className="market-grid-container">
                        <div className="section-header">
                            <h2>Market Overview 🇮🇳</h2>
                            <p className="subtitle">Real-time prices for top NSE stocks</p>
                        </div>

                        {/* Search and Filters */}
                        <div className="market-controls">
                            <input
                                type="text"
                                className="market-search"
                                placeholder="Search stocks (e.g., Infosys, INFY)..."
                                value={searchTerm}
                                onChange={(e) => setSearchTerm(e.target.value)}
                            />

                            <div className="category-filters">
                                {CATEGORIES.map(cat => (
                                    <button
                                        key={cat}
                                        className={`filter-chip ${activeCategory === cat ? 'active' : ''}`}
                                        onClick={() => setActiveCategory(cat)}
                                    >
                                        {cat}
                                    </button>
                                ))}
                            </div>
                        </div>

                        {loading ? (
                            <div className="loading-spinner">Loading Market...</div>
                        ) : filteredData.length > 0 ? (
                            <StockGrid
                                data={filteredData}
                                onSelectStock={(stock) => setSelectedStock(stock)}
                            />
                        ) : (
                            <div className="no-results">No stocks found matching your criteria.</div>
                        )}
                    </div>
                )}
            </div>
        </div>
    );
};

export default MarketDashboard;
