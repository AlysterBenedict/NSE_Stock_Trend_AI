import React from 'react';
import { PageHeader } from '../../components/common/ui';
import PredictionControlPanel from './PredictionControlPanel';
import PredictionResultCard from './PredictionResultCard';
import TechnicalPriceChart from './TechnicalPriceChart';

const PredictionDashboard = ({
  onPredict,
  isLoading,
  stock,
  setStock,
  algorithm,
  setAlgorithm,
  date,
  setDate,
  predictionResult,
  error,
}) => (
  <div className="page-stack">
    <PageHeader
      eyebrow="Machine Learning Forecast"
      icon="query_stats"
      title="Price Prediction"
      description="Estimate future closing prices with calibrated 90% confidence ranges, out-of-sample validation metrics, and synchronized technical history."
    />

    <div className="bento-grid">
      <div className="bento-aside">
        <PredictionControlPanel
          onPredict={onPredict}
          isLoading={isLoading}
          stock={stock}
          setStock={setStock}
          algorithm={algorithm}
          setAlgorithm={setAlgorithm}
          date={date}
          setDate={setDate}
        />
      </div>

      <div className="bento-main">
        <PredictionResultCard result={predictionResult} error={error} isLoading={isLoading} />
      </div>
    </div>

    <TechnicalPriceChart stockName={stock} />
  </div>
);

export default PredictionDashboard;
