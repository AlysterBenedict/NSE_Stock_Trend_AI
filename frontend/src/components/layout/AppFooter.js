import React from 'react';
import { Icon } from '../common/ui';

const FEATURES = [
  { icon: 'query_stats', label: 'ML forecasts' },
  { icon: 'candlestick_chart', label: 'Interactive technical charts' },
  { icon: 'forum', label: 'Conversational market insights' },
];

const AppFooter = ({ showFeatures }) => (
  <footer className="app-footer">
    <div className="app-footer__inner">
      <div className="app-footer__lead">
        <span className="status-dot" />
        <span>© {new Date().getFullYear()} NeuroStock. All rights reserved.</span>
      </div>

      {showFeatures ? (
        <ul className="app-footer__features">
          {FEATURES.map((f) => (
            <li key={f.label}>
              <Icon name={f.icon} size={16} />
              {f.label}
            </li>
          ))}
        </ul>
      ) : (
        <p className="app-footer__note">
          Forecasts are statistical estimates for research purposes and do not constitute investment advice.
        </p>
      )}
    </div>
  </footer>
);

export default AppFooter;
