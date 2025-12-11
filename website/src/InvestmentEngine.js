
import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { createChart, ColorType } from 'lightweight-charts';
import { format, parseISO } from 'date-fns';
import ReactMarkdown from 'react-markdown'; // Added Import
import './App.css'; // Reuse App css + new styles

// --- CHART COMPONENT (Simplified for Small Cards) ---
const EngineChart = ({ trendData, predictedPrice, width = 300, height = 200 }) => {
    const chartRef = useRef();
    const chartInstance = useRef();

    useEffect(() => {
        if (!trendData?.dates || !chartRef.current) return;

        const historyCutoff = trendData.history_cutoff;
        const lastActualPrice = parseFloat(trendData.prices[historyCutoff - 1]);
        const predictedPriceNum = parseFloat(predictedPrice);
        const isProfit = predictedPriceNum >= lastActualPrice;

        chartInstance.current = createChart(chartRef.current, {
            layout: {
                background: { type: ColorType.Solid, color: 'transparent' },
                textColor: '#B0BEC5',
            },
            grid: {
                vertLines: { visible: false },
                horzLines: { visible: false },
            },
            width: width,
            height: height,
            timeScale: { visible: false }, // Hide dates for cleaner look in cards
            rightPriceScale: { visible: true, borderVisible: false },
            handleScroll: false,
            handleScale: false,
        });

        const lineSeries = chartInstance.current.addLineSeries({
            color: '#2962FF',
            lineWidth: 2,
            crosshairMarkerVisible: false,
        });

        // Prepare Data
        const data = [];
        trendData.dates.forEach((d, i) => {
            const val = trendData.prices[i];
            if (val) data.push({ time: d, value: val });
        });

        lineSeries.setData(data);

        // Mark Forecast Area (Simple coloration technique or marker)
        // For simplicity, just one line.

        chartInstance.current.timeScale().fitContent();

        return () => {
            if (chartInstance.current) chartInstance.current.remove();
        };
    }, [trendData]);

    return <div ref={chartRef} className="engine-mini-chart" />;
};


const InvestmentEngine = () => {
    const [principal, setPrincipal] = useState(10000);
    const [withdrawalDate, setWithdrawalDate] = useState('');
    const [status, setStatus] = useState('IDLE'); // IDLE, PROCESSING, COMPLETE, ERROR
    const [completedStocks, setCompletedStocks] = useState([]);
    const [currentStock, setCurrentStock] = useState(null);
    const [results, setResults] = useState(null);
    const [error, setError] = useState(null);


    // Full Stock List matching Backend
    const stockList = [
        "Reliance", "TCS", "HDFC Bank", "Infosys", "ICICI Bank", "HUL", "SBI", "Bharti Airtel", "ITC",
        "Kotak Bank", "L&T", "Axis Bank", "Wipro", "HCL Tech", "Asian Paints", "Bajaj Finance",
        "Maruti", "Sun Pharma", "Titan", "Tata Steel", "NTPC", "Power Grid", "Tata Motors",
        "UltraTech", "Nestle Ind", "Adani Ent", "M&M", "ONGC", "Coal India", "Grasim"
    ];

    const startProcessing = async () => {
        if (!withdrawalDate) {
            alert("Please select a withdrawal date");
            return;
        }

        setStatus('PROCESSING');
        setCompletedStocks([]);
        setCurrentStock(stockList[0]);
        setError(null);

        // Visual Simulation: Growing List
        let idx = 0;
        let processInterval = setInterval(() => {
            idx++;
            if (idx < stockList.length) {
                setCompletedStocks(prev => [stockList[idx - 1], ...prev]); // Add to top of completed list
                setCurrentStock(stockList[idx]);
            } else if (idx === stockList.length) {
                // Final step: Move last stock to completed and show Finalizing (RUN ONCE)
                setCompletedStocks(prev => [stockList[stockList.length - 1], ...prev]);
                setCurrentStock("Finalizing AI Verdict...");
            }
        }, 1200); // 1.2s per stock

        try {
            const response = await axios.post('http://127.0.0.1:5000/investment-engine', {
                principal: principal,
                withdrawal_date: withdrawalDate
            });

            clearInterval(processInterval);
            setResults(response.data);
            setStatus('COMPLETE');

        } catch (err) {
            clearInterval(processInterval);
            console.error(err);
            setError("Analysis Failed: " + (err.response?.data?.error || err.message));
            setStatus('ERROR');
        }
    };

    // Helper to calculate total value
    const calculateTotal = (profitPct) => {
        const p = parseFloat(principal);
        const profit = p * (parseFloat(profitPct) / 100);
        return (p + profit).toLocaleString('en-IN', { style: 'currency', currency: 'INR', maximumFractionDigits: 0 });
    };

    return (
        <div className="investment-engine fade-in">
            {/* INPUT SECTION */}
            {status === 'IDLE' && (
                <div className="engine-input-card fade-in">
                    <h2 className="engine-title">AI Investment Engine</h2>
                    <p className="engine-subtitle">
                        Deep-dive analysis of NSE Top 30 stocks using <b>Ensemble AI (LSTM + Random Forest)</b>.
                    </p>

                    <div className="engine-form">
                        <div className="input-group">
                            <label>Principal Amount (₹)</label>
                            <input
                                type="number"
                                value={principal}
                                onChange={e => setPrincipal(e.target.value)}
                                className="engine-input"
                            />
                        </div>
                        <div className="input-group">
                            <label>Withdrawal Date</label>
                            <input
                                type="date"
                                value={withdrawalDate}
                                onChange={e => setWithdrawalDate(e.target.value)}
                                className="engine-input"
                                min={new Date().toISOString().split("T")[0]}
                            />
                        </div>
                        <button className="engine-btn" onClick={startProcessing}>
                            Start Analysis
                        </button>
                    </div>
                </div>
            )}

            {/* PROCESSING SECTION - NEW DESIGN */}
            {status === 'PROCESSING' && (
                <div className="processing-container fade-in">

                    {/* Active Processing Card */}
                    <div className="processing-card active-card glow-border">
                        <div className="scan-line"></div> {/* Scanner Animation */}

                        <div className="processing-header">
                            <div className="tech-spinner">
                                <div className="inner-circle"></div>
                            </div>
                            <div className="header-text">
                                <h2 className="active-stock-name">{currentStock}</h2>
                                <span className="processing-badge">AI ANALYZING</span>
                            </div>
                        </div>

                        <p className="processing-subtext">
                            Analyzing Volatility • Forecasting Trends • Sentiment Scan
                        </p>

                        {/* Progress Bar */}
                        <div className="progress-container">
                            <div className="progress-bar" style={{ width: `${(completedStocks.length / stockList.length) * 100}%` }}></div>
                        </div>
                        <div className="progress-label">
                            {Math.round((completedStocks.length / stockList.length) * 100)}% Complete
                        </div>
                    </div>

                    {/* Completed Stocks List */}
                    <div className="completed-stocks-list">
                        {completedStocks.map((stock, i) => (
                            <div key={i} className="completed-item fade-in-up">
                                <span className="check-icon">✓</span>
                                <span className="stock-name">{stock}</span>
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {/* RESULTS SECTION */}
            {status === 'COMPLETE' && results && (
                <div className="engine-results fade-in">
                    <button className="back-btn" onClick={() => setStatus('IDLE')}>← New Search</button>

                    <h2 className="results-header">Top 3 Investment Picks</h2>

                    <div className="top-cards-container">
                        {results.top_stocks.map((stock, index) => (
                            <div key={stock.ticker} className={`top-card rank-${index + 1} hover-scale`}>
                                <div className="card-badge">Rank #{index + 1}</div>
                                <div className="card-content">
                                    <h3>{stock.name}</h3>

                                    <div className="metric-row total-value-row" style={{ background: 'rgba(41, 98, 255, 0.05)', padding: '8px', borderRadius: '8px', marginBottom: '15px', border: 'none' }}>
                                        <span style={{ color: '#2962ff' }}>Est. Total Value</span>
                                        <span className="val" style={{ color: '#1565c0', fontSize: '1.1rem' }}>
                                            {calculateTotal(stock.profit_pct)}
                                        </span>
                                    </div>

                                    <div className="metric-row">
                                        <span>Profit Forecast</span>
                                        <span className={`val ${stock.profit_pct >= 0 ? 'pos' : 'neg'}`}>
                                            {stock.profit_pct > 0 ? '+' : ''}{stock.profit_pct}%
                                        </span>
                                    </div>
                                    <div className="metric-row">
                                        <span>Stability Score</span>
                                        <span className="val">{100 - stock.volatility}%</span>
                                    </div>
                                    <div className="metric-row">
                                        <span>Sentiment</span>
                                        <span className="val" style={{ color: stock.sentiment_score > 60 ? '#00E676' : '#FFEA00' }}>
                                            {stock.sentiment_score}/100
                                        </span>
                                    </div>
                                    <div className="metric-row">
                                        <span>NSE Rank</span>
                                        <span className="val">#{stock.nse_rank}</span>
                                    </div>
                                    <div className="chart-preview-wrapper">
                                        <EngineChart
                                            trendData={stock.trend_data}
                                            predictedPrice={stock.predicted_price}
                                        />
                                    </div>
                                </div>
                            </div>
                        ))}
                    </div>


                    <div className="advice-section glass-panel">
                        <img src="/chatbot_icon.png" alt="AI Bot" className="advisor-avatar-img" />
                        <div className="advice-content">
                            <h3>AI Investment Verdict</h3>
                            <div className="advice-markdown-container">
                                <ReactMarkdown>{results.ai_advice}</ReactMarkdown>
                            </div>
                        </div>
                    </div>
                </div>
            )}

            {status === 'ERROR' && (
                <div className="error-card">
                    <h3>Result Error</h3>
                    <p>{error}</p>
                    <button onClick={() => setStatus('IDLE')}>Try Again</button>
                </div>
            )}
        </div>
    );
};

export default InvestmentEngine;
