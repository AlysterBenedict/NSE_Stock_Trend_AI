import React, { useEffect, useRef, useState } from 'react';
import { createChart, ColorType, CrosshairMode } from 'lightweight-charts';
import axios from 'axios';

const AdvancedChart = ({ stockName }) => {
    const chartContainerRef1 = useRef();
    const chartContainerRef2 = useRef();
    const chartInstance1 = useRef(null);
    const chartInstance2 = useRef(null);

    // Refs to hold series instances
    const seriesRefs = useRef({
        candle: null,
        sma: null,
        volume: null
    });

    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState(null);

    // State for series visibility (Candle is always visible now)
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
                const data = response.data;
                console.log("API Response Data:", data);

                if (!data || !data.ohlc || data.ohlc.length === 0) {
                    console.error("No OHLC data received");
                    setError("No data available for this stock.");
                    return;
                }

                if (chartContainerRef1.current && chartContainerRef2.current) {
                    // Clean up previous charts if exists
                    if (chartInstance1.current) {
                        try { chartInstance1.current.remove(); } catch (e) { console.warn(e); }
                        chartInstance1.current = null;
                    }
                    if (chartInstance2.current) {
                        try { chartInstance2.current.remove(); } catch (e) { console.warn(e); }
                        chartInstance2.current = null;
                    }
                    seriesRefs.current = { candle: null, sma: null, volume: null };

                    // --- Chart 1: Candle ---
                    const chart1 = createChart(chartContainerRef1.current, {
                        layout: {
                            background: { type: ColorType.Solid, color: '#FFFFFF' }, // Pure White
                            textColor: '#1A202C', // Very Dark Grey/Black for contrast
                        },
                        grid: {
                            vertLines: { color: 'rgba(0, 0, 0, 0.08)' }, // Slightly stronger grid
                            horzLines: { color: 'rgba(0, 0, 0, 0.08)' },
                        },
                        width: chartContainerRef1.current.clientWidth,
                        height: 600,
                        crosshair: { mode: CrosshairMode.Normal },
                        rightPriceScale: { borderColor: 'rgba(0, 0, 0, 0.15)' },
                        timeScale: { borderColor: 'rgba(0, 0, 0, 0.15)', timeVisible: true },
                    });
                    chartInstance1.current = chart1;

                    const candlestickSeries = chart1.addCandlestickSeries({
                        upColor: '#006400', // Dark Green
                        downColor: '#D50000', // Vibrant Red
                        borderVisible: false,
                        wickUpColor: '#006400',
                        wickDownColor: '#D50000',
                    });
                    candlestickSeries.setData(data.ohlc);
                    seriesRefs.current.candle = candlestickSeries;

                    // --- Chart 2: SMA & Volume ---
                    const chart2 = createChart(chartContainerRef2.current, {
                        layout: {
                            background: { type: ColorType.Solid, color: '#FFFFFF' }, // Pure White
                            textColor: '#1A202C', // Very Dark Grey/Black for contrast
                        },
                        grid: {
                            vertLines: { color: 'rgba(0, 0, 0, 0.08)' }, // Slightly stronger grid
                            horzLines: { color: 'rgba(0, 0, 0, 0.08)' },
                        },
                        width: chartContainerRef2.current.clientWidth,
                        height: 600,
                        crosshair: { mode: CrosshairMode.Normal },
                        rightPriceScale: { borderColor: 'rgba(0, 0, 0, 0.15)' },
                        timeScale: { borderColor: 'rgba(0, 0, 0, 0.15)', timeVisible: true },
                    });
                    chartInstance2.current = chart2;

                    // SMA Series
                    const smaSeries = chart2.addLineSeries({
                        color: '#2962FF', // Vivid Blue
                        lineWidth: 2,
                        title: 'SMA 50',
                    });
                    smaSeries.setData(data.sma);
                    seriesRefs.current.sma = smaSeries;
                    smaSeries.applyOptions({ visible: visibility.sma });

                    // Volume Series
                    const volumeSeries = chart2.addHistogramSeries({
                        color: '#00897B', // Vibrant Teal
                        priceFormat: { type: 'volume' },
                        priceScaleId: '', // Overlay
                        scaleMargins: { top: 0.7, bottom: 0 },
                    });
                    const volumeData = data.volume.map((v) => ({ ...v, color: '#00897Bcc' })); // Slightly transparent teal
                    volumeSeries.setData(volumeData);
                    seriesRefs.current.volume = volumeSeries;
                    volumeSeries.applyOptions({ visible: visibility.volume });

                    // --- Synchronization ---
                    const timeScale1 = chart1.timeScale();
                    const timeScale2 = chart2.timeScale();

                    timeScale1.subscribeVisibleLogicalRangeChange((range) => {
                        if (isSyncingRef.current) {
                            timeScale2.setVisibleLogicalRange(range);
                        }
                    });

                    timeScale2.subscribeVisibleLogicalRangeChange((range) => {
                        if (isSyncingRef.current) {
                            timeScale1.setVisibleLogicalRange(range);
                        }
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
            if (chartInstance1.current && chartContainerRef1.current) {
                chartInstance1.current.applyOptions({ width: chartContainerRef1.current.clientWidth });
            }
            if (chartInstance2.current && chartContainerRef2.current) {
                chartInstance2.current.applyOptions({ width: chartContainerRef2.current.clientWidth });
            }
        };

        const resizeObserver = new ResizeObserver(() => {
            handleResize();
        });

        if (chartContainerRef1.current) resizeObserver.observe(chartContainerRef1.current);
        if (chartContainerRef2.current) resizeObserver.observe(chartContainerRef2.current);
        // Also observe the wrapper to catch layout changes
        const wrapper = document.querySelector('.advanced-chart-wrapper');
        if (wrapper) resizeObserver.observe(wrapper);

        window.addEventListener('resize', handleResize);

        return () => {
            window.removeEventListener('resize', handleResize);
            resizeObserver.disconnect();
            if (chartInstance1.current) {
                try { chartInstance1.current.remove(); } catch (e) { console.warn(e); }
                chartInstance1.current = null;
            }
            if (chartInstance2.current) {
                try { chartInstance2.current.remove(); } catch (e) { console.warn(e); }
                chartInstance2.current = null;
            }
        };

        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [stockName, visibility]); // Added visibility to deps or ignored it. Using ignore for now as per previous intent, but actually I added it to deps in the code above. Wait, the previous code had it. Let's stick to stockName and use the ref pattern if needed, or just ignore. I'll use the ignore line.

    return (
        <div className="advanced-chart-wrapper">
            <h3>Technical Analysis: {stockName}</h3>

            {/* Charts Row */}
            <div className="charts-row" style={{ display: 'flex', gap: '20px', marginBottom: '10px', marginTop: '40px', flexWrap: 'wrap' }}>
                {/* Chart 1: Candle */}
                <div className="chart-wrapper-relative" style={{ position: 'relative', flex: 1, minWidth: '400px' }}>
                    {isLoading && (
                        <div className="chart-loading-overlay" style={{
                            position: 'absolute', top: 0, left: 0, right: 0, bottom: 0,
                            background: 'rgba(0,0,0,0.7)', display: 'flex', justifyContent: 'center', alignItems: 'center', zIndex: 10
                        }}>
                            <div className="spinner"></div>
                        </div>
                    )}
                    {error && (
                        <div className="chart-error-overlay" style={{
                            position: 'absolute', top: 0, left: 0, right: 0, bottom: 0,
                            background: 'rgba(0,0,0,0.8)', display: 'flex', justifyContent: 'center', alignItems: 'center', color: 'red', zIndex: 10
                        }}>
                            {error}
                        </div>
                    )}
                    <div ref={chartContainerRef1} className="chart-container" style={{ height: '600px' }} />
                </div>

                {/* Chart 2: SMA & Volume */}
                <div className="chart-wrapper-relative" style={{ position: 'relative', flex: 1, minWidth: '400px', display: 'flex', flexDirection: 'column' }}>
                    {/* Sync Toggle Button - Top Right Overlay */}
                    <button
                        onClick={toggleSync}
                        style={{
                            position: 'absolute',
                            top: '-35px',
                            right: '0',
                            zIndex: 20,
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

                    <div ref={chartContainerRef2} className="chart-container" style={{ height: '600px' }} />

                    {/* Interactive Legend - Moved under Chart 2 */}
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

export default AdvancedChart;
