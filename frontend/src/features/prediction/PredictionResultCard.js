import React from 'react';
import { Badge, Callout, Card, ChangeBadge, EmptyState, Icon, Loader } from '../../components/common/ui';
import { exportPredictionBrief } from '../../utils/exportResearchBrief';
import { formatPrice, formatSigned } from '../../utils/formatters';
import ForecastXRayPanel from './ForecastXRayPanel';
import PredictionTrendChart from './PredictionTrendChart';

const PredictionResultCard = ({ result, error, isLoading = false, slotLabel = null }) => {
  if (isLoading) {
    return (
      <Card className="result-card result-card--state">
        <Loader label={slotLabel ? `Computing ${slotLabel} forecast...` : 'Computing model forecast...'} />
      </Card>
    );
  }

  if (error) {
    return (
      <Card className="result-card result-card--state">
        <Callout tone="error" icon="error" title="Forecast Request Error">
          {error}
        </Callout>
      </Card>
    );
  }

  if (!result || !result.trend_data || !result.trend_data.prices) {
    return (
      <Card className="result-card result-card--state">
        <EmptyState
          icon="query_stats"
          title={slotLabel ? `${slotLabel} Ready` : 'Ready to Forecast'}
          description="Choose a stock, forecasting algorithm, and target date, then run the prediction to inspect the projected closing price and 90% interval."
        />
      </Card>
    );
  }

  const {
    predicted_price: predictedPrice,
    stock_name: stockName,
    future_date: futureDate,
    algorithm_name: algorithmName,
    prediction_type: predictionType,
    warning,
    trend_data: trendData,
    confidence_interval: confidenceInterval,
    model_metrics: modelMetrics,
  } = result;

  const historyCutoff = trendData.history_cutoff;
  let lastActualPrice = null;
  let deltaPrice = null;
  let deltaPct = null;
  let directionTone = 'neutral';

  if (historyCutoff > 0 && historyCutoff <= trendData.prices.length) {
    const parsedLast = parseFloat(trendData.prices[historyCutoff - 1]);
    const parsedPred = parseFloat(predictedPrice);
    if (Number.isFinite(parsedLast) && Number.isFinite(parsedPred) && parsedLast > 0) {
      lastActualPrice = parsedLast;
      deltaPrice = parsedPred - parsedLast;
      deltaPct = (deltaPrice / parsedLast) * 100;
      if (deltaPrice > 0) directionTone = 'up';
      if (deltaPrice < 0) directionTone = 'down';
    }
  }

  return (
    <Card className="result-card">
      <div className="result-card__head">
        <div className="result-card__titles">
          <div className="result-card__badges">
            {slotLabel && <Badge tone="neutral">{slotLabel}</Badge>}
            <Badge tone={directionTone === 'down' ? 'down' : 'primary'}>{algorithmName}</Badge>
          </div>
          <h3 className="result-card__type">{predictionType}</h3>
        </div>

        <div className="result-card__actions">
          <Badge tone="neutral" icon="calendar_today">
            Target {futureDate}
          </Badge>
          <button
            type="button"
            className="btn btn--secondary btn--sm"
            onClick={() => exportPredictionBrief(result, 'md')}
            title="Download Markdown Research Brief"
          >
            <Icon name="download" size={14} />
            <span>.md Brief</span>
          </button>
          <button
            type="button"
            className="btn btn--secondary btn--sm"
            onClick={() => exportPredictionBrief(result, 'pdf')}
            title="Print / Save PDF Research Brief"
          >
            <Icon name="print" size={14} />
            <span>Print PDF</span>
          </button>
        </div>
      </div>

      {warning && (
        <Callout tone="secondary" icon="info" title="Model Note">
          {warning}
        </Callout>
      )}

      <div className={`forecast-hero forecast-hero--${directionTone}`}>
        <div className="forecast-hero__primary">
          <span className="forecast-hero__eyebrow">Projected Close · {stockName}</span>
          <div className="forecast-hero__value-row">
            <span className="forecast-hero__price">₹{formatPrice(predictedPrice)}</span>
            {deltaPct !== null && <ChangeBadge value={deltaPct} />}
          </div>
          {deltaPrice !== null && (
            <span className="forecast-hero__sub">
              {formatSigned(deltaPrice)} ₹ vs last close (₹{formatPrice(lastActualPrice)})
            </span>
          )}
        </div>

        {confidenceInterval && (
          <div className="forecast-hero__band">
            <span className="forecast-hero__eyebrow">90% Confidence Range</span>
            <span className="forecast-hero__range">
              ₹{formatPrice(confidenceInterval.lower)} – ₹{formatPrice(confidenceInterval.upper)}
            </span>
            <span className="forecast-hero__sub">Calibrated residual interval</span>
          </div>
        )}
      </div>

      {modelMetrics && (
        <div className="validation-strip" aria-label="Out-of-sample validation metrics">
          <div className="validation-item">
            <span className="validation-item__label">
              <Icon name="check_circle" size={14} /> Dir. Accuracy
            </span>
            <span className="validation-item__value">{modelMetrics.directional_accuracy_pct}%</span>
          </div>
          <div className="validation-item">
            <span className="validation-item__label">
              <Icon name="straighten" size={14} /> MAPE
            </span>
            <span className="validation-item__value">{modelMetrics.mape_pct}%</span>
          </div>
          <div className="validation-item">
            <span className="validation-item__label">
              <Icon name="trending_up" size={14} /> Skill vs Naive
            </span>
            <span
              className={`validation-item__value ${
                modelMetrics.skill_vs_naive_pct >= 0 ? 'is-up' : 'is-down'
              }`}
            >
              {modelMetrics.skill_vs_naive_pct >= 0 ? '+' : ''}
              {modelMetrics.skill_vs_naive_pct}%
            </span>
          </div>
          {modelMetrics.rank_ic !== undefined && (
            <div className="validation-item">
              <span className="validation-item__label">
                <Icon name="equalizer" size={14} /> Rank IC
              </span>
              <span
                className={`validation-item__value ${modelMetrics.rank_ic >= 0 ? 'is-up' : 'is-down'}`}
              >
                {modelMetrics.rank_ic >= 0 ? '+' : ''}
                {modelMetrics.rank_ic}
              </span>
            </div>
          )}
        </div>
      )}

      <PredictionTrendChart
        key={stockName + futureDate + algorithmName}
        trendData={trendData}
        predictedPrice={predictedPrice}
      />

      <ForecastXRayPanel explainability={result.explainability} />
    </Card>
  );
};

export default PredictionResultCard;
