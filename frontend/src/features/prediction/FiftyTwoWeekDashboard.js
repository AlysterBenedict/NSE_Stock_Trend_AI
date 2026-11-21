import React from 'react';
import SearchableSelect from '../../components/common/SearchableSelect';
import { Field, PageHeader } from '../../components/common/ui';
import { PREDICTION_STOCK_OPTIONS } from '../../constants/stocks';
import FiftyTwoWeekAnalysisChart from './FiftyTwoWeekAnalysisChart';

const FiftyTwoWeekDashboard = ({ stock, setStock }) => (
  <div className="page-stack">
    <PageHeader
      eyebrow="252-Session Technical Structure"
      icon="candlestick_chart"
      title="52-Week Analysis"
      description="Inspect trailing one-year price extremes, OHLC candlestick structure, 50-day moving average, and volume with synchronized pan and zoom."
      actions={
        <div className="header-select-box">
          <Field label="Equity">
            <SearchableSelect
              options={PREDICTION_STOCK_OPTIONS}
              value={stock}
              onChange={setStock}
              placeholder="Select stock..."
            />
          </Field>
        </div>
      }
    />

    <FiftyTwoWeekAnalysisChart stockName={stock} />
  </div>
);

export default FiftyTwoWeekDashboard;
