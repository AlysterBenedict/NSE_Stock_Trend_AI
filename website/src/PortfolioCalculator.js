import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { createChart, ColorType } from 'lightweight-charts';
import SearchableSelect from './SearchableSelect'; // Reuse the stock selector

const PortfolioCalculator = () => {
    // Inputs
    const [stock, setStock] = useState('Infosys'); // Default
    const [principal, setPrincipal] = useState(10000);
    const [targetDate, setTargetDate] = useState('');

    // State
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState(null);
    const [error, setError] = useState('');

    // Chart Ref
    const chartContainerRef = useRef(null);

    // Initial Date Default (1 Year from now)
    useEffect(() => {
        const d = new Date();
        d.setFullYear(d.getFullYear() + 1);
        setTargetDate(d.toISOString().split('T')[0]);
    }, []);

    // Helper to format currency
    const formatCurrency = (val) => {
        return new Intl.NumberFormat('en-IN', {
            style: 'currency',
            currency: 'INR',
            maximumFractionDigits: 0
        }).format(val);
    };

    const handleCalculate = async () => {
        if (!stock || principal <= 0 || !targetDate) {
            setError("Please fill all fields correctly.");
            return;
        }

        setError('');
        setLoading(true);
        setResult(null);

        try {
            const response = await axios.post('http://127.0.0.1:5000/calculate-portfolio', {
                stock_name: stock,
                principal: principal,
                future_date: targetDate
            });
            setResult(response.data);
        } catch (err) {
            console.error(err);
            setError(err.response?.data?.error || "Calculation failed. Try a different stock or date.");
        } finally {
            setLoading(false);
        }
    };

    // Render Chart
    useEffect(() => {
        if (result && chartContainerRef.current) {
            const chart = createChart(chartContainerRef.current, {
                layout: {
                    background: { type: ColorType.Solid, color: 'transparent' },
                    textColor: '#333', // Dark text for light theme
                },
                width: chartContainerRef.current.clientWidth,
                height: 300,
                grid: {
                    vertLines: { color: 'rgba(0, 0, 0, 0.06)' },
                    horzLines: { color: 'rgba(0, 0, 0, 0.06)' },
                },
                rightPriceScale: {
                    borderColor: 'rgba(0, 0, 0, 0.1)',
                },
                timeScale: {
                    borderColor: 'rgba(0, 0, 0, 0.1)',
                },
            });

            // projected Series (Area)
            const areaSeries = chart.addAreaSeries({
                lineColor: '#6366f1',
                topColor: 'rgba(99, 102, 241, 0.4)',
                bottomColor: 'rgba(99, 102, 241, 0.0)',
            });

            // Baseline (Principal) Line
            const baselineSeries = chart.addLineSeries({
                color: '#ef4444', // Red for baseline/cost
                lineWidth: 1,
                lineStyle: 1, // Dashed
                title: 'Principal'
            });

            if (result.chart_data) {
                // Prepare data
                const areaData = result.chart_data.map(d => ({ time: d.time, value: d.value }));

                // Create a constant line for principal
                // We need to match the start and end dates of the projected data
                if (areaData.length > 0) {
                    const baselineData = areaData.map(d => ({ time: d.time, value: principal }));

                    areaSeries.setData(areaData);
                    baselineSeries.setData(baselineData);
                }

                chart.timeScale().fitContent();
            }

            const handleResize = () => {
                chart.applyOptions({ width: chartContainerRef.current.clientWidth });
            };

            window.addEventListener('resize', handleResize);

            return () => {
                window.removeEventListener('resize', handleResize);
                chart.remove();
            };
        }
    }, [result, principal]);


    return (
        <div className="portfolio-container fade-in">
            <h2 className="section-title">AI Portfolio Projector</h2>
            <p className="subtitle">Estimate your future wealth with machine learning forecasts.</p>

            <div className="portfolio-grid">
                {/* Inputs Card */}
                <div className="card input-card">
                    <h3>Investment Details</h3>

                    <div className="form-group">
                        <label>Select Stock</label>
                        <SearchableSelect
                            options={[
                                { value: "Infosys", label: "Infosys (INFY.NS)" },
                                { value: "Yes Bank", label: "Yes Bank (YESBANK.NS)" },
                                { value: "TCS", label: "TCS (TCS.NS)" },
                                { value: "HDFC Bank", label: "HDFC Bank (HDFCBANK.NS)" },
                                { value: "ITC", label: "ITC (ITC.NS)" },
                                { value: "Power Grid Corp", label: "Power Grid Corp (POWERGRID.NS)" },
                                { value: "Bajaj Finserv", label: "Bajaj Finserv (BAJAJFINSV.NS)" },
                                { value: "Adani Ports", label: "Adani Ports (ADANIPORTS.NS)" },
                                { value: "Tata Steel", label: "Tata Steel (TATASTEEL.NS)" },
                                { value: "Asian Paints", label: "Asian Paints (ASIANPAINT.NS)" },
                                { value: "JSW Steel", label: "JSW Steel (JSWSTEEL.NS)" },
                                { value: "Bajaj Auto", label: "Bajaj Auto (BAJAJ-AUTO.NS)" },
                                { value: "Lupin", label: "Lupin (LUPIN.NS)" },
                                { value: "Hindalco", label: "Hindalco (HINDALCO.NS)" },
                                { value: "LTIMindtree", label: "LTIMindtree (LTIM.NS)" },
                                { value: "Grasim", label: "Grasim (GRASIM.NS)" },
                                { value: "Cipla", label: "Cipla (CIPLA.NS)" },
                                { value: "Tech Mahindra", label: "Tech Mahindra (TECHM.NS)" },
                                { value: "Wipro", label: "Wipro (WIPRO.NS)" },
                                { value: "Nestle India", label: "Nestle India (NESTLEIND.NS)" },
                                { value: "Adani Green", label: "Adani Green (ADANIGREEN.NS)" },
                                { value: "BEL", label: "BEL (BEL.NS)" },
                                { value: "Varun Beverages", label: "Varun Beverages (VBL.NS)" },
                                { value: "IndusInd Bank", label: "IndusInd Bank (INDUSINDBK.NS)" },
                                { value: "Tata Consumer", label: "Tata Consumer (TATACONSUM.NS)" },
                                { value: "Zomato", label: "Zomato (ZOMATO.NS)" },
                                { value: "Britannia", label: "Britannia (BRITANNIA.NS)" },
                                { value: "SBI Life", label: "SBI Life (SBILIFE.NS)" },
                                { value: "HAL", label: "HAL (HAL.NS)" },
                                { value: "Trent", label: "Trent (TRENT.NS)" }
                            ]}
                            onChange={setStock}
                            placeholder="Search (e.g., Infosys)"
                            initialValue={stock}
                        />
                    </div>

                    <div className="form-group">
                        <label>Principal Amount (₹)</label>
                        <input
                            type="number"
                            className="glass-input"
                            value={principal}
                            onChange={(e) => setPrincipal(e.target.value)}
                        />
                    </div>

                    <div className="form-group">
                        <label>Target Withdrawal Date</label>
                        <input
                            type="date"
                            className="glass-input"
                            value={targetDate}
                            onChange={(e) => setTargetDate(e.target.value)}
                        />
                    </div>

                    <button
                        className="predict-btn primary-btn"
                        onClick={handleCalculate}
                        disabled={loading}
                    >
                        {loading ? 'Calculating...' : 'Project Growth'}
                    </button>

                    {error && <p className="error-text">{error}</p>}
                </div>

                {/* Results Area */}
                <div className="card results-card-large">
                    {!result ? (
                        <div className="placeholder-content">
                            <img
                                src="/portfolio_placeholder_1765379821679.png"
                                alt="Financial Future"
                                className="placeholder-image"
                                style={{ maxWidth: '600px', opacity: 0.9, marginBottom: '20px' }}
                            />
                            <p>Enter details to see your financial future.</p>
                        </div>
                    ) : (
                        <div className="results-content">
                            {/* Key Metrics Row */}
                            <div className="metrics-row">
                                <div className="metric-box">
                                    <span className="metric-label">Projected Value</span>
                                    <span className="metric-value highlight">{formatCurrency(result.final_value)}</span>
                                    <span className={`metric-sub ${result.roi >= 0 ? 'success' : 'danger'}`}>
                                        {result.roi >= 0 ? '+' : ''}{result.roi}% ROI
                                    </span>
                                </div>
                                <div className="metric-box">
                                    <span className="metric-label">CAGR</span>
                                    <span className="metric-value">{result.cagr}%</span>
                                    <span className="metric-sub">Annual Growth</span>
                                </div>
                                <div className="metric-box">
                                    <span className="metric-label">Risk Level</span>
                                    <span className={`metric-tag ${result.risk_score.toLowerCase()}`}>{result.risk_score}</span>
                                    <span className="metric-sub">Vol: {result.risk_value}%</span>
                                </div>
                            </div>

                            {/* Chart Area */}
                            <div className="chart-wrapper-portfolio">
                                <h4>projected Wealth Curve</h4>
                                <div ref={chartContainerRef} className="chart-container-inner" />
                            </div>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};

export default PortfolioCalculator;
