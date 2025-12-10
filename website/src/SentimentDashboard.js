import React, { useState } from 'react';
import axios from 'axios';
import SearchableSelect from './SearchableSelect';
import './App.css';

const SentimentPlaceholder = () => (
    <div className="sentiment-placeholder">
        <img src="/sentiment_placeholder.png" alt="Ready to Analyze" className="placeholder-img" />
        <div className="placeholder-text">
            <h3>Ready to Analyze Market Sentiment</h3>
            <p>Select a stock from the menu above and click "Analyze ⚡" to get real-time, AI-powered insights from the latest news headlines.</p>
        </div>
    </div>
);

const SentimentDashboard = () => {
    const [stock, setStock] = useState('Infosys');
    const [sentimentData, setSentimentData] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [hasAnalyzed, setHasAnalyzed] = useState(false); // Track if analysis ran

    // Full list of 50 NSE Stocks
    const stocks = [
        { value: "Adani Enterprises", label: "Adani Enterprises (ADANIENT.NS)" },
        { value: "Adani Ports", label: "Adani Ports (ADANIPORTS.NS)" },
        { value: "Apollo Hospitals", label: "Apollo Hospitals (APOLLOHOSP.NS)" },
        { value: "Asian Paints", label: "Asian Paints (ASIANPAINT.NS)" },
        { value: "Axis Bank", label: "Axis Bank (AXISBANK.NS)" },
        { value: "Bajaj Auto", label: "Bajaj Auto (BAJAJ-AUTO.NS)" },
        { value: "Bajaj Finance", label: "Bajaj Finance (BAJFINANCE.NS)" },
        { value: "Bajaj Finserv", label: "Bajaj Finserv (BAJAJFINSV.NS)" },
        { value: "Bharti Airtel", label: "Bharti Airtel (BHARTIARTL.NS)" },
        { value: "BPCL", label: "BPCL (BPCL.NS)" },
        { value: "Britannia", label: "Britannia (BRITANNIA.NS)" },
        { value: "Cipla", label: "Cipla (CIPLA.NS)" },
        { value: "Coal India", label: "Coal India (COALINDIA.NS)" },
        { value: "Divis Lab", label: "Divis Lab (DIVISLAB.NS)" },
        { value: "Dr Reddys Labs", label: "Dr Reddys Labs (DRREDDY.NS)" },
        { value: "Eicher Motors", label: "Eicher Motors (EICHERMOT.NS)" },
        { value: "Grasim", label: "Grasim (GRASIM.NS)" },
        { value: "HCL Tech", label: "HCL Tech (HCLTECH.NS)" },
        { value: "HDFC Bank", label: "HDFC Bank (HDFCBANK.NS)" },
        { value: "HDFC Life", label: "HDFC Life (HDFCLIFE.NS)" },
        { value: "Hero MotoCorp", label: "Hero MotoCorp (HEROMOTOCO.NS)" },
        { value: "Hindalco", label: "Hindalco (HINDALCO.NS)" },
        { value: "Hindustan Unilever", label: "Hindustan Unilever (HINDUNILVR.NS)" },
        { value: "ICICI Bank", label: "ICICI Bank (ICICIBANK.NS)" },
        { value: "IndusInd Bank", label: "IndusInd Bank (INDUSINDBK.NS)" },
        { value: "Infosys", label: "Infosys (INFY.NS)" },
        { value: "ITC", label: "ITC (ITC.NS)" },
        { value: "JSW Steel", label: "JSW Steel (JSWSTEEL.NS)" },
        { value: "Kotak Mahindra Bank", label: "Kotak Mahindra Bank (KOTAKBANK.NS)" },
        { value: "L&T", label: "L&T (LT.NS)" },
        { value: "LTIMindtree", label: "LTIMindtree (LTIM.NS)" },
        { value: "M&M", label: "M&M (M&M.NS)" },
        { value: "Maruti Suzuki", label: "Maruti Suzuki (MARUTI.NS)" },
        { value: "Nestle India", label: "Nestle India (NESTLEIND.NS)" },
        { value: "NTPC", label: "NTPC (NTPC.NS)" },
        { value: "ONGC", label: "ONGC (ONGC.NS)" },
        { value: "Power Grid Corp", label: "Power Grid Corp (POWERGRID.NS)" },
        { value: "Reliance", label: "Reliance (RELIANCE.NS)" },
        { value: "SBI Life", label: "SBI Life (SBILIFE.NS)" },
        { value: "SBI", label: "SBI (SBIN.NS)" },
        { value: "Sun Pharma", label: "Sun Pharma (SUNPHARMA.NS)" },
        { value: "Tata Consumer", label: "Tata Consumer (TATACONSUM.NS)" },
        { value: "Tata Motors", label: "Tata Motors (TATAMOTORS.NS)" },
        { value: "Tata Steel", label: "Tata Steel (TATASTEEL.NS)" },
        { value: "TCS", label: "TCS (TCS.NS)" },
        { value: "Tech Mahindra", label: "Tech Mahindra (TECHM.NS)" },
        { value: "Titan", label: "Titan (TITAN.NS)" },
        { value: "UltraTech Cement", label: "UltraTech Cement (ULTRACEMCO.NS)" },
        { value: "UPL", label: "UPL (UPL.NS)" },
        { value: "Wipro", label: "Wipro (WIPRO.NS)" }
    ];

    const fetchSentiment = async () => {
        setLoading(true);
        setError(null);
        setSentimentData(null);
        setHasAnalyzed(true); // Mark as analyzed to show results
        try {
            const response = await axios.post('http://127.0.0.1:5000/get-sentiment', {
                stock_name: stock
            });
            setSentimentData(response.data);
        } catch (err) {
            console.error("Sentiment Error:", err);
            setError("Failed to fetch AI sentiment. " + (err.response?.data?.error || err.message));
        } finally {
            setLoading(false);
        }
    };

    // Helper to determine color based on score
    const getScoreColor = (score) => {
        if (score >= 75) return '#10B981'; // Green
        if (score >= 40) return '#F59E0B'; // Yellow/Orange
        return '#EF4444'; // Red
    };

    const formatDate = (ts) => {
        try {
            if (!ts) return "Unknown Date";
            // Check if unix timestamp (number) or string
            const date = new Date(typeof ts === 'number' ? ts * 1000 : ts);
            return isNaN(date.getTime()) ? "Recent" : date.toLocaleString();
        } catch (e) {
            return "Recent";
        }
    };

    return (
        <div className="sentiment-dashboard">
            <div className="sentiment-header">
                {/* Header removed */}
                <div className="stock-selector">
                    <div style={{ width: '300px' }}>
                        <SearchableSelect
                            options={stocks}
                            value={stock}
                            onChange={setStock}
                            placeholder="Select stock..."
                        />
                    </div>
                    <button
                        className="refresh-btn"
                        onClick={fetchSentiment}
                    >
                        Analyze
                    </button>
                </div>
            </div>

            {!hasAnalyzed ? (
                <SentimentPlaceholder />
            ) : (
                <div className="sentiment-grid">
                    {/* Left: Gauge & Score */}
                    <div className="sentiment-card gauge-card">
                        <h3>Market Mood</h3>
                        {loading ? (
                            <div className="loading-spinner">Analyzing News...</div>
                        ) : error ? (
                            <div className="error-msg">{error}</div>
                        ) : sentimentData ? (
                            <div className="gauge-container">
                                <div className="sentiment-score" style={{ color: getScoreColor(sentimentData.score) }}>
                                    {sentimentData.score}
                                </div>
                                <div className="sentiment-label">
                                    {sentimentData.label}
                                </div>
                                <div className="sentiment-summary">
                                    "{sentimentData.summary}"
                                </div>
                                {/* Simple visual bar */}
                                <div className="score-bar-bg">
                                    <div
                                        className="score-bar-fill"
                                        style={{
                                            width: `${sentimentData.score}%`,
                                            backgroundColor: getScoreColor(sentimentData.score)
                                        }}
                                    ></div>
                                </div>
                                <div className="score-labels">
                                    <span>Bearish</span>
                                    <span>Neutral</span>
                                    <span>Bullish</span>
                                </div>
                            </div>
                        ) : null}
                    </div>

                    {/* Right: News Feed */}
                    <div className="sentiment-card news-card">
                        <h3>AI-Decoded News Feed 📰</h3>
                        {loading ? (
                            <div className="shimmer-loader"></div>
                        ) : sentimentData?.news?.length > 0 ? (
                            <div className="news-list">
                                {sentimentData.news.map((item, idx) => (
                                    <div key={idx} className="news-item">
                                        <div className="news-content">
                                            <a href={item.link} target="_blank" rel="noopener noreferrer" className="news-title">
                                                {item.title}
                                            </a>
                                            <div className="news-meta">
                                                {formatDate(item.time)}
                                            </div>
                                        </div>
                                        <span className="ai-tag">AI Analyzed</span>
                                    </div>
                                ))}
                            </div>
                        ) : (
                            !loading && <div className="no-data">No news found.</div>
                        )}
                    </div>
                </div>
            )}
        </div>
    );
};

export default SentimentDashboard;
