import React, { useEffect, useRef } from 'react';
import { createChart, LineStyle } from 'lightweight-charts';
import { format, parseISO } from 'date-fns';
import { useTheme } from '../../context/ThemeContext';
import { getChartColors, getLightChartOptions } from '../../utils/chartHelpers';
import { isValidLineData } from '../../utils/formatters';

const PredictionTrendChart = ({ trendData, predictedPrice }) => {
  const chartRef = useRef();
  const chartInstance = useRef();
  const { theme } = useTheme();
  const colors = getChartColors(theme);

  const historyCutoff = trendData?.history_cutoff ?? 0;
  const lastActual =
    historyCutoff > 0 && historyCutoff <= (trendData?.prices?.length ?? 0)
      ? parseFloat(trendData.prices[historyCutoff - 1])
      : null;
  const isIncrease = lastActual !== null ? parseFloat(predictedPrice) >= lastActual : true;
  const forecastColor = isIncrease ? colors.primary : colors.tertiary;
  const hasBands = Array.isArray(trendData?.upper_band) && Array.isArray(trendData?.lower_band);

  useEffect(() => {
    if (!trendData?.dates || !trendData.prices || !chartRef.current) return undefined;
    if (historyCutoff <= 0 || historyCutoff > trendData.prices.length) return undefined;

    chartInstance.current = createChart(
      chartRef.current,
      getLightChartOptions(
        {
          width: chartRef.current.clientWidth,
          height: 300,
          timeScale: {
            borderVisible: false,
            timeVisible: true,
            secondsVisible: false,
            fixLeftEdge: true,
            fixRightEdge: true,
          },
        },
        theme
      )
    );

    try {
      const historyDataForChart = trendData.prices
        .slice(0, historyCutoff)
        .map((p, i) => ({ time: format(parseISO(trendData.dates[i]), 'yyyy-MM-dd'), value: p }))
        .filter(isValidLineData);

      if (historyDataForChart.length > 0) {
        const lineSeries = chartInstance.current.addLineSeries({
          color: colors.historical,
          lineWidth: 2,
        });
        lineSeries.setData(historyDataForChart);
      }

      const forecastDataForChart = [];
      const lastHistTimeStr = trendData.dates[historyCutoff - 1];
      const lastHistValue = trendData.prices[historyCutoff - 1];
      if (lastHistTimeStr && lastHistValue !== null && isFinite(lastHistValue)) {
        forecastDataForChart.push({
          time: format(parseISO(lastHistTimeStr), 'yyyy-MM-dd'),
          value: lastHistValue,
        });
      }
      for (let i = historyCutoff; i < trendData.dates.length; i += 1) {
        const futureTimeStr = trendData.dates[i];
        const futureValue = trendData.prices[i];
        if (futureTimeStr && futureValue !== null && isFinite(futureValue)) {
          forecastDataForChart.push({
            time: format(parseISO(futureTimeStr), 'yyyy-MM-dd'),
            value: futureValue,
          });
        }
      }

      if (forecastDataForChart.length > 0) {
        const forecastSeries = chartInstance.current.addLineSeries({
          color: forecastColor,
          lineWidth: 2.5,
          lineStyle: LineStyle.Dashed,
        });
        forecastSeries.setData(forecastDataForChart);
      }

      if (hasBands) {
        const upperData = trendData.upper_band
          .map((v, i) =>
            v !== null && v !== undefined && trendData.dates[i]
              ? { time: format(parseISO(trendData.dates[i]), 'yyyy-MM-dd'), value: Number(v) }
              : null
          )
          .filter(isValidLineData);
        const lowerData = trendData.lower_band
          .map((v, i) =>
            v !== null && v !== undefined && trendData.dates[i]
              ? { time: format(parseISO(trendData.dates[i]), 'yyyy-MM-dd'), value: Number(v) }
              : null
          )
          .filter(isValidLineData);

        if (upperData.length > 1) {
          const upperSeries = chartInstance.current.addLineSeries({
            color: colors.bandUpper,
            lineWidth: 1,
            lineStyle: LineStyle.Dotted,
          });
          upperSeries.setData(upperData);
        }
        if (lowerData.length > 1) {
          const lowerSeries = chartInstance.current.addLineSeries({
            color: colors.bandLower,
            lineWidth: 1,
            lineStyle: LineStyle.Dotted,
          });
          lowerSeries.setData(lowerData);
        }
      }

      chartInstance.current.timeScale().fitContent();
    } catch (chartError) {
      console.error('Chart error:', chartError);
    }

    const handleResize = () => {
      if (chartInstance.current && chartRef.current) {
        chartInstance.current.applyOptions({ width: chartRef.current.clientWidth });
      }
    };
    const observer = new ResizeObserver(handleResize);
    observer.observe(chartRef.current);
    window.addEventListener('resize', handleResize);

    return () => {
      observer.disconnect();
      window.removeEventListener('resize', handleResize);
      if (chartInstance.current) {
        try {
          chartInstance.current.remove();
        } catch (e) {
          console.warn('Chart disposal error:', e);
        }
        chartInstance.current = null;
      }
    };
  }, [trendData, predictedPrice, historyCutoff, forecastColor, hasBands, theme, colors]);

  return (
    <div className="chart-section">
      <div className="chart-section__head">
        <span className="chart-section__title">Prediction Trend (100D History + Forecast)</span>
        <div className="chart-legend-inline">
          <span className="legend-dot-item">
            <span className="legend-swatch legend-swatch--line" style={{ background: colors.historical }} />
            Historical
          </span>
          <span className="legend-dot-item">
            <span className="legend-swatch legend-swatch--dashed" style={{ borderColor: forecastColor }} />
            Forecast
          </span>
          {hasBands && (
            <span className="legend-dot-item">
              <span className="legend-swatch legend-swatch--dotted" style={{ borderColor: colors.bandUpper }} />
              90% Band
            </span>
          )}
        </div>
      </div>
      <div className="chart-well">
        <div className="chart-canvas" style={{ height: 300 }} ref={chartRef} />
      </div>
    </div>
  );
};

export default PredictionTrendChart;
