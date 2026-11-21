import React from 'react';
import { Badge, Callout, Card, PageHeader, Stat } from '../../components/common/ui';
import { formatPrice } from '../../utils/formatters';
import ComparisonControlPanel from './ComparisonControlPanel';
import PredictionResultCard from './PredictionResultCard';

const getDirection = (res) => {
  const cutoff = res?.trend_data?.history_cutoff;
  if (!cutoff || cutoff <= 0 || cutoff > (res?.trend_data?.prices?.length ?? 0)) return 0;
  const last = parseFloat(res.trend_data.prices[cutoff - 1]);
  const pred = parseFloat(res.predicted_price);
  if (!Number.isFinite(last) || !Number.isFinite(pred)) return 0;
  return pred >= last ? 1 : -1;
};

const ModelComparisonDashboard = ({
  onPredict,
  isLoading,
  stock,
  setStock,
  algo1,
  setAlgo1,
  algo2,
  setAlgo2,
  date,
  setDate,
  result1,
  result2,
  error,
}) => {
  const hasBoth = Boolean(result1?.predicted_price && result2?.predicted_price);
  let midpoint = null;
  let spread = null;
  let spreadPct = null;
  let consensus = null;

  if (hasBoth) {
    const p1 = parseFloat(result1.predicted_price);
    const p2 = parseFloat(result2.predicted_price);
    if (Number.isFinite(p1) && Number.isFinite(p2)) {
      midpoint = (p1 + p2) / 2;
      spread = Math.abs(p1 - p2);
      spreadPct = midpoint > 0 ? (spread / midpoint) * 100 : 0;
      const d1 = getDirection(result1);
      const d2 = getDirection(result2);
      if (d1 === 1 && d2 === 1) consensus = { label: 'Bullish Consensus', tone: 'up' };
      else if (d1 === -1 && d2 === -1) consensus = { label: 'Bearish Consensus', tone: 'down' };
      else consensus = { label: 'Divergent Signals', tone: 'secondary' };
    }
  }

  return (
    <div className="page-stack">
      <PageHeader
        eyebrow="Dual-Model Benchmark"
        icon="compare_arrows"
        title="Model Comparison"
        description="Evaluate two forecasting architectures side by side on the same equity and target date to measure signal consensus and forecast dispersion."
      />

      <ComparisonControlPanel
        onPredict={onPredict}
        isLoading={isLoading}
        stock={stock}
        setStock={setStock}
        algo1={algo1}
        setAlgo1={setAlgo1}
        algo2={algo2}
        setAlgo2={setAlgo2}
        date={date}
        setDate={setDate}
      />

      {error && (
        <Callout tone="error" icon="error" title="Comparison Failed">
          {error}
        </Callout>
      )}

      {hasBoth && midpoint !== null && (
        <Card className="consensus-strip">
          <div className="consensus-strip__lead">
            <span className="eyebrow-inline">Dual-Model Synthesis</span>
            <Badge tone={consensus.tone} icon="balance">
              {consensus.label}
            </Badge>
          </div>
          <div className="kpi-grid kpi-grid--3">
            <Stat
              label="Ensemble Midpoint"
              value={`₹${formatPrice(midpoint)}`}
              hint={`Mean of ${result1.algorithm_name} & ${result2.algorithm_name}`}
              icon="adjust"
            />
            <Stat
              label="Model Spread"
              value={`₹${formatPrice(spread)}`}
              hint={`${spreadPct.toFixed(2)}% relative dispersion`}
              icon="unfold_more"
            />
            <Stat
              label="Higher Target"
              value={
                parseFloat(result1.predicted_price) >= parseFloat(result2.predicted_price)
                  ? result1.algorithm_name
                  : result2.algorithm_name
              }
              hint={`Target date ${result1.future_date}`}
              icon="north_east"
            />
          </div>
        </Card>
      )}

      <div className="grid-2 comparison-grid">
        <PredictionResultCard result={result1} error={null} isLoading={isLoading} slotLabel="Model A" />
        <PredictionResultCard result={result2} error={null} isLoading={isLoading} slotLabel="Model B" />
      </div>
    </div>
  );
};

export default ModelComparisonDashboard;
