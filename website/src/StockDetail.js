import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { createChart, ColorType } from 'lightweight-charts';
import './App.css';

const StockDetail = ({ ticker, name, onBack }) => {
    const [details, setDetails] = useState(null);
    const [loading, setLoading] = useState(true);
    const chartContainerRef = useRef();

    useEffect(() => {
        fetchDetails();
    }, [ticker]);

    // Initialize Chart
    useEffect(() => {
        if (details && details.chart && chartContainerRef.current) {
            const chart = createChart(chartContainerRef.current, {
                layout: {
                    background: { type: ColorType.Solid, color: 'transparent' },
                    textColor: '#333',
                },
                width: chartContainerRef.current.clientWidth,
                height: 300,
                grid: {
                    vertLines: { color: 'rgba(0, 0, 0, 0.05)' },
                    horzLines: { color: 'rgba(0, 0, 0, 0.05)' },
                },
                rightPriceScale: {
                    borderColor: 'rgba(0, 0, 0, 0.1)',
                },
                timeScale: {
                    borderColor: 'rgba(0, 0, 0, 0.1)',
                },
            });

            const lineSeries = chart.addLineSeries({
                color: '#2962FF',
                lineWidth: 2,
            });

            // Ensure data is sorted by time
            const sortedData = [...details.chart].sort((a, b) => new Date(a.time) - new Date(b.time));
            lineSeries.setData(sortedData);
            chart.timeScale().fitContent();

            const handleResize = () => {
                chart.applyOptions({ width: chartContainerRef.current.clientWidth });
            };
            window.addEventListener('resize', handleResize);
            return () => {
                window.removeEventListener('resize', handleResize);
                chart.remove();
            };
        }
    }, [details]);

    const fetchDetails = async () => {
        try {
            const response = await axios.get(`http://127.0.0.1:5000/get-market-data?ticker=${ticker}`);
            setDetails(response.data);
            setLoading(false);
        } catch (error) {
            console.error("Error fetching available details:", error);
            setLoading(false);
        }
    };

    if (loading) return <div className="loading-spinner">Loading Analysis for {name}...</div>;
    if (!details) return <div className="error-text">Could not load data.</div>;

    const { info } = details;

    // Helper to render a metric card if value exists
    const MetricItem = ({ label, value, format = null }) => {
        if (value === undefined || value === null) return null;
        let displayVal = value;
        if (format === 'currency') displayVal = `₹${value.toLocaleString()}`;
        if (format === 'percent') displayVal = `${(value * 100).toFixed(2)}%`;
        if (format === 'large') displayVal = (value / 10000000).toFixed(2) + ' Cr'; // Convert to Crores approx

        return (
            <div className="detail-metric">
                <span className="dm-label">{label}</span>
                <span className="dm-value">{displayVal}</span>
            </div>
        );
    };

    return (
        <div className="stock-detail-view fade-in">
            <button className="back-btn" onClick={onBack}>← Back to Market</button>

            <div className="detail-header">
                <div>
                    <h1>{name} <span className="ticker-badge">{ticker}</span></h1>
                    <p className="sector-badge">{info.sector} | {info.industry}</p>
                </div>
                <div className="big-price">
                    ₹{info.currentPrice || info.regularMarketPrice}
                </div>
            </div>

            <div className="detail-chart-wrapper" ref={chartContainerRef}></div>

            <div className="detail-section">
                <h3>Key Financials</h3>
                <div className="detail-grid">
                    <MetricItem label="Market Cap" value={info.marketCap} format="large" />
                    <MetricItem label="P/E Ratio" value={info.trailingPE} />
                    <MetricItem label="Forward P/E" value={info.forwardPE} />
                    <MetricItem label="EPS (TTM)" value={info.trailingEps} />
                    <MetricItem label="Book Value" value={info.bookValue} />
                    <MetricItem label="Dividend Yield" value={info.dividendYield} format="percent" />
                    <MetricItem label="Return on Equity" value={info.returnOnEquity} format="percent" />
                    <MetricItem label="Total Revenue" value={info.totalRevenue} format="large" />
                    <MetricItem label="Profit Margins" value={info.profitMargins} format="percent" />
                </div>
            </div>

            <div className="detail-section">
                <h3>Price Statistics</h3>
                <div className="detail-grid">
                    <MetricItem label="52 Week High" value={info.fiftyTwoWeekHigh} format="currency" />
                    <MetricItem label="52 Week Low" value={info.fiftyTwoWeekLow} format="currency" />
                    <MetricItem label="50-Day SMA" value={info.fiftyDayAverage} format="currency" />
                    <MetricItem label="200-Day SMA" value={info.twoHundredDayAverage} format="currency" />
                    <MetricItem label="Beta (Volatility)" value={info.beta} />
                    <MetricItem label="Volume" value={info.volume} />
                    <MetricItem label="Avg Volume" value={info.averageVolume} />
                </div>
            </div>

            <div className="detail-section">
                <h3>Company Profile</h3>
                <p className="company-summary">{info.longBusinessSummary}</p>
            </div>
        </div>
    );
};

export default StockDetail;
