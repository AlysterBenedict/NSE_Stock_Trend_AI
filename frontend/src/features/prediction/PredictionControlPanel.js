import React from 'react';
import { format } from 'date-fns';
import SearchableSelect from '../../components/common/SearchableSelect';
import { Card, CardHeader, Field, Icon, Spinner } from '../../components/common/ui';
import { ALGORITHM_OPTIONS, PREDICTION_STOCK_OPTIONS } from '../../constants/stocks';

const PredictionControlPanel = ({
  onPredict,
  isLoading,
  stock,
  setStock,
  algorithm,
  setAlgorithm,
  date,
  setDate,
}) => (
  <Card className="control-panel sticky-panel">
    <CardHeader
      icon="tune"
      iconTone="primary"
      title="Forecast Parameters"
      subtitle="Select equity, model, and target date"
    />

    <div className="form-stack">
      <Field label="Equity" hint="NSE" htmlFor="stock-select">
        <SearchableSelect
          options={PREDICTION_STOCK_OPTIONS}
          value={stock}
          onChange={setStock}
          placeholder="Select stock..."
        />
      </Field>

      <Field label="Forecasting Model" htmlFor="algo-select">
        <SearchableSelect
          options={ALGORITHM_OPTIONS}
          value={algorithm}
          onChange={setAlgorithm}
          placeholder="Search or select an algorithm..."
        />
      </Field>

      <Field label="Target Date" htmlFor="future-date">
        <input
          type="date"
          id="future-date"
          className="input"
          value={date}
          onChange={(e) => setDate(e.target.value)}
          min={format(new Date(), 'yyyy-MM-dd')}
        />
      </Field>

      <button
        id="predict-btn"
        type="button"
        className="btn btn--primary btn--block"
        onClick={() => onPredict(stock, algorithm, date)}
        disabled={isLoading || !stock}
      >
        {isLoading ? <Spinner size="sm" /> : <Icon name="auto_graph" size={18} />}
        <span>{isLoading ? 'Running Forecast...' : 'Predict Trend'}</span>
      </button>
    </div>
  </Card>
);

export default PredictionControlPanel;
