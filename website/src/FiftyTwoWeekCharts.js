import React, { useEffect, useRef, useState } from 'react';
import { createChart, ColorType, CrosshairMode } from 'lightweight-charts';
import axios from 'axios';

const FiftyTwoWeekCharts = ({ stockName }) => {
    const chartContainerRef1 = useRef(); // Line Chart
    const chartContainerRef2 = useRef(); // Candle Chart
    const chartContainerRef3 = useRef(); // Volume & SMA Chart

    const chartInstance1 = useRef(null);
    const chartInstance2 = useRef(null);
    const chartInstance3 = useRef(null);

    // Refs to hold series instances
    const seriesRefs = useRef({
        line: null,
        candle: null,
        sma: null,
        volume: null
    });

    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState(null);
    const [metrics, setMetrics] = useState({ high: null, low: null });

    // State for series visibility (for Chart 3)
    const [visibility, setVisibility] = useState({
        sma: true,
        volume: true
    });

    // Sync State
    const [isSyncing, setIsSyncing] = useState(true);
    const isSyncingRef = useRef(true);

    // Toggle function for visibility
    const toggleSeries = (key) => {
        setVisibility(prev => ({ ...prev, [key]: !prev[key] }));
    };

    // Toggle function for sync
    const toggleSync = () => {
        setIsSyncing(prev => {
            const newState = !prev;
            isSyncingRef.current = newState;
            return newState;
        });
    };

    // Effect to update visibility when state changes
    useEffect(() => {
        if (seriesRefs.current.sma) {
            seriesRefs.current.sma.applyOptions({ visible: visibility.sma });
        }
        if (seriesRefs.current.volume) {
            seriesRefs.current.volume.applyOptions({ visible: visibility.volume });
        }
    }, [visibility]);

    useEffect(() => {
        const fetchDataAndRender = async () => {
            setIsLoading(true);
            setError(null);
            try {
                const response = await axios.get(`/historical-data?stock_name=${stockName}`);
                const fullData = response.data;
                console.log("API Response Data:", fullData);

                if (!fullData || !fullData.ohlc || fullData.ohlc.length === 0) {
                    console.error("No OHLC data received");
                    setError("No data available for this stock.");
                    return;
                }

                // --- Filter Data for Last 52 Weeks (approx 252 trading days) ---
                const cutoffIndex = Math.max(0, fullData.ohlc.length - 252);

                const data = {
                    ohlc: fullData.ohlc.slice(cutoffIndex),
                    volume: fullData.volume.slice(cutoffIndex),
                    sma: fullData.sma.slice(cutoffIndex),
                    rsi: fullData.rsi.slice(cutoffIndex) // Not used here but good to have
                };

                // --- Calculate 52-Week High & Low ---
                let high = -Infinity;
                let low = Infinity;
                data.ohlc.forEach(d => {
                    if (d.high > high) high = d.high;
                    if (d.low < low) low = d.low;
                });
                setMetrics({ high, low });

                if (chartContainerRef1.current && chartContainerRef2.current && chartContainerRef3.current) {
                    // Clean up previous charts
                    [chartInstance1, chartInstance2, chartInstance3].forEach(ref => {
                        if (ref.current) {
                            try { ref.current.remove(); } catch (e) { console.warn(e); }
                            ref.current = null;
                        }
                    });
                    seriesRefs.current = { line: null, candle: null, sma: null, volume: null };

                    const commonOptions = {
                        layout: {
                            background: { type: ColorType.Solid, color: '#FFFFFF' },
                            textColor: '#1A202C',
                        },
                        grid: {
                            vertLines: { color: 'rgba(0, 0, 0, 0.08)' },
                            horzLines: { color: 'rgba(0, 0, 0, 0.08)' },
                        },
                        crosshair: { mode: CrosshairMode.Normal },
                        rightPriceScale: { borderColor: 'rgba(0, 0, 0, 0.15)' },
                        timeScale: { borderColor: 'rgba(0, 0, 0, 0.15)', timeVisible: true },
                        height: 500, // Fixed height for each chart
                    };

                    // --- Chart 1: Line Chart (Close Price) ---
                    const chart1 = createChart(chartContainerRef1.current, {
                        ...commonOptions,
                        width: chartContainerRef1.current.clientWidth,
                    });
                    chartInstance1.current = chart1;

                    const lineSeries = chart1.addLineSeries({
                        color: '#2962FF',
                        lineWidth: 2,
                        title: 'Close Price',
                    });
                    // Convert OHLC to Line data (time, value=close)
                    const lineData = data.ohlc.map(d => ({ time: d.time, value: d.close }));
                    lineSeries.setData(lineData);
                    seriesRefs.current.line = lineSeries;


                    // --- Chart 2: Candle Chart ---
                    const chart2 = createChart(chartContainerRef2.current, {
                        ...commonOptions,
                        width: chartContainerRef2.current.clientWidth,
                    });
                    chartInstance2.current = chart2;

                    const candlestickSeries = chart2.addCandlestickSeries({
                        upColor: '#006400',
                        downColor: '#D50000',
                        borderVisible: false,
                        wickUpColor: '#006400',
                        wickDownColor: '#D50000',
                    });
                    candlestickSeries.setData(data.ohlc);
                    seriesRefs.current.candle = candlestickSeries;


                    // --- Chart 3: SMA & Volume ---
                    const chart3 = createChart(chartContainerRef3.current, {
                        ...commonOptions,
                        width: chartContainerRef3.current.clientWidth,
                    });
                    chartInstance3.current = chart3;

                    // SMA Series
                    const smaSeries = chart3.addLineSeries({
                        color: '#2962FF',
                        lineWidth: 2,
                        title: 'SMA 50',
                    });
                    smaSeries.setData(data.sma);
                    seriesRefs.current.sma = smaSeries;
                    smaSeries.applyOptions({ visible: visibility.sma });

                    // Volume Series
                    const volumeSeries = chart3.addHistogramSeries({
                        color: '#00897B',
                        priceFormat: { type: 'volume' },
                        priceScaleId: '', // Overlay
                        scaleMargins: { top: 0.7, bottom: 0 },
                    });
                    const volumeData = data.volume.map((v) => ({ ...v, color: '#00897Bcc' }));
                    volumeSeries.setData(volumeData);
                    seriesRefs.current.volume = volumeSeries;
                    volumeSeries.applyOptions({ visible: visibility.volume });


                    // --- Synchronization ---
                    const charts = [chart1, chart2, chart3];

                    charts.forEach((sourceChart, index) => {
                        sourceChart.timeScale().subscribeVisibleLogicalRangeChange((range) => {
                            if (isSyncingRef.current) {
                                charts.forEach((targetChart, targetIndex) => {
                                    if (index !== targetIndex) {
                                        targetChart.timeScale().setVisibleLogicalRange(range);
                                    }
                                });
                            }
                        });
                    });
                }
            } catch (err) {
                console.error("Error fetching historical data:", err);
                setError("Failed to load chart data.");
            } finally {
                setIsLoading(false);
            }
        };

        if (stockName) {
            fetchDataAndRender();
        }

        const handleResize = () => {
            [chartInstance1, chartInstance2, chartInstance3].forEach((inst, i) => {
                const container = [chartContainerRef1, chartContainerRef2, chartContainerRef3][i];
                if (inst.current && container.current) {
                    inst.current.applyOptions({ width: container.current.clientWidth });
                }
            });
        };

        const resizeObserver = new ResizeObserver(() => {
            handleResize();
        });

        [chartContainerRef1, chartContainerRef2, chartContainerRef3].forEach(ref => {
            if (ref.current) resizeObserver.observe(ref.current);
        });

        window.addEventListener('resize', handleResize);

        return () => {
            window.removeEventListener('resize', handleResize);
            resizeObserver.disconnect();
            [chartInstance1, chartInstance2, chartInstance3].forEach(ref => {
                if (ref.current) {
                    try { ref.current.remove(); } catch (e) { console.warn(e); }
                    ref.current = null;
                }
            });
        };

    }, [stockName, visibility]); // Re-run if stock or visibility changes

    return (
        <div className="advanced-chart-wrapper">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <h3>52-Week Analysis: {stockName}</h3>

                {/* Sync Toggle Button */}
                <button
                    onClick={toggleSync}
                    style={{
                        background: 'transparent',
                        border: '1px solid rgba(0, 0, 0, 0.1)',
                        borderRadius: '4px',
                        color: '#2D3748',
                        padding: '5px 10px',
                        cursor: 'pointer',
                        fontSize: '0.8rem',
                        display: 'flex',
                        alignItems: 'center',
                        gap: '5px',
                        transition: 'all 0.2s'
                    }}
                    title={isSyncing ? "Unlink Charts" : "Link Charts"}
                >
                    <span>{isSyncing ? '🔗' : '🔓'}</span> {isSyncing ? 'Sync On' : 'Sync Off'}
                </button>
            </div>

            {/* 52-Week Metrics Display */}
            {!isLoading && !error && metrics.high !== null && (
                <div style={{
                    display: 'flex',
                    gap: '20px',
                    marginBottom: '20px',
                    background: 'rgba(255, 255, 255, 0.5)',
                    padding: '15px',
                    borderRadius: '12px',
                    border: '1px solid rgba(0,0,0,0.05)'
                }}>
                    <div style={{ display: 'flex', flexDirection: 'column' }}>
                        <span style={{ fontSize: '0.9rem', color: '#718096' }}>52-Week High</span>
                        <span style={{ fontSize: '1.2rem', fontWeight: 'bold', color: '#00C853' }}>
                            ₹{metrics.high.toFixed(2)}
                        </span>
                    </div>
                    <div style={{ width: '1px', background: 'rgba(0,0,0,0.1)' }}></div>
                    <div style={{ display: 'flex', flexDirection: 'column' }}>
                        <span style={{ fontSize: '0.9rem', color: '#718096' }}>52-Week Low</span>
                        <span style={{ fontSize: '1.2rem', fontWeight: 'bold', color: '#D50000' }}>
                            ₹{metrics.low.toFixed(2)}
                        </span>
                    </div>
                </div>
            )}

            {isLoading && <div className="loading-indicator"><div className="spinner"></div><p>Loading 52-week data...</p></div>}
            {error && <div className="error-message">{error}</div>}

            <div className="charts-column" style={{ display: 'flex', flexDirection: 'column', gap: '20px', marginTop: '20px' }}>

                {/* Chart 1: Line Chart */}
                <div className="chart-container-wrapper">
                    <h4>Line Chart (Close Price)</h4>
                    <div ref={chartContainerRef1} className="chart-container" style={{ height: '500px' }} />
                </div>

                {/* Chart 2: Candle Chart */}
                <div className="chart-container-wrapper">
                    <h4>Candlestick Chart</h4>
                    <div ref={chartContainerRef2} className="chart-container" style={{ height: '500px' }} />
                </div>

                {/* Chart 3: Volume & SMA */}
                <div className="chart-container-wrapper" style={{ position: 'relative' }}>
                    <h4>Volume & SMA</h4>
                    <div ref={chartContainerRef3} className="chart-container" style={{ height: '500px' }} />

                    {/* Legend/Toggles */}
                    <div className="chart-legend" style={{ marginTop: '10px', justifyContent: 'center' }}>
                        <button
                            className={`legend-item ${!visibility.sma ? 'hidden' : ''}`}
                            onClick={() => toggleSeries('sma')}
                            style={{ color: '#2962FF' }}
                        >
                            <span className="legend-marker">—</span> SMA 50
                        </button>

                        <button
                            className={`legend-item ${!visibility.volume ? 'hidden' : ''}`}
                            onClick={() => toggleSeries('volume')}
                            style={{ color: '#00897B' }}
                        >
                            <span className="legend-marker">▮</span> Volume
                        </button>
                    </div>
                </div>

            </div>
        </div>
    );
};

export default FiftyTwoWeekCharts;
