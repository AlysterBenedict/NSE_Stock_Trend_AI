import React from 'react';
import { format } from 'date-fns';
import SearchableSelect from '../../components/common/SearchableSelect';
import { Card, Field, Icon, Spinner } from '../../components/common/ui';
import { ALGORITHM_OPTIONS, PREDICTION_STOCK_OPTIONS } from '../../constants/stocks';

const ComparisonControlPanel = ({
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
}) => (
  <Card className="comparison-bar">
    <div className="comparison-bar__grid">
      <Field label="Equity" hint="NSE">
        <SearchableSelect
          options={PREDICTION_STOCK_OPTIONS}
          value={stock}
          onChange={setStock}
          placeholder="Select stock..."
        />
      </Field>

      <Field label="Model A">
        <SearchableSelect
          options={ALGORITHM_OPTIONS}
          value={algo1}
          onChange={setAlgo1}
          placeholder="Select Model A..."
        />
      </Field>

      <Field label="Model B">
        <SearchableSelect
          options={ALGORITHM_OPTIONS}
          value={algo2}
          onChange={setAlgo2}
          placeholder="Select Model B..."
        />
      </Field>

      <Field label="Target Date">
        <input
          type="date"
          className="input"
          value={date}
          onChange={(e) => setDate(e.target.value)}
          min={format(new Date(), 'yyyy-MM-dd')}
        />
      </Field>

      <div className="comparison-bar__cta">
        <button
          type="button"
          className="btn btn--primary btn--block"
          onClick={() => onPredict(stock, algo1, algo2, date)}
          disabled={isLoading || !stock}
        >
          {isLoading ? <Spinner size="sm" /> : <Icon name="compare_arrows" size={18} />}
          <span>{isLoading ? 'Comparing...' : 'Run Comparison'}</span>
        </button>
      </div>
    </div>
  </Card>
);

export default ComparisonControlPanel;
