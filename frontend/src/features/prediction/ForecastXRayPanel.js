import React from 'react';
import { Badge, Icon } from '../../components/common/ui';
import { formatPrice, formatSigned } from '../../utils/formatters';

const toneClassFor = (val) => (Number(val) >= 0 ? 'up' : 'down');

const ForecastXRayPanel = ({ explainability }) => {
  if (!explainability) return null;

  const {
    anchor_horizon: anchorHorizon,
    consensus,
    model_votes: modelVotes = [],
    top_drivers: topDrivers = [],
    term_structure: termStructure = [],
  } = explainability;

  const maxVoteAbs = Math.max(
    0.25,
    ...modelVotes.map((m) => Math.abs(Number(m.predicted_return_pct) || 0))
  );
  const maxDriverAbs = Math.max(
    0.05,
    ...topDrivers.map((d) => Math.abs(Number(d.impact_pct) || 0))
  );

  const convictionTone =
    consensus?.agreement_pct >= 75
      ? consensus?.direction === 'bullish'
        ? 'up'
        : 'down'
      : 'secondary';

  return (
    <div className="xray-shell">
      <div className="xray-header">
        <div className="xray-header__title">
          <span className="icon-tile icon-tile--primary">
            <Icon name="psychology" size={18} />
          </span>
          <div>
            <h4>Explainable AI Forecast X-Ray</h4>
            <p>
              Model vote breakdown, live factor attributions, and cross-horizon term structure (h = {anchorHorizon}D)
            </p>
          </div>
        </div>

        {consensus && (
          <Badge tone={convictionTone} icon="verified">
            {consensus.label} · {consensus.bullish_count}/{consensus.total_models} Bullish ({consensus.agreement_pct}%)
          </Badge>
        )}
      </div>

      <div className="xray-grid">
        {modelVotes.length > 0 && (
          <div className="xray-box">
            <div className="xray-box__head">
              <h5>7-Model Consensus Votes</h5>
              <span>Calibrated Return &amp; Stacker Weight</span>
            </div>
            <div className="vote-list">
              {modelVotes.map((vote) => {
                const pct = Number(vote.predicted_return_pct) || 0;
                const widthPct = Math.min(100, Math.max(4, (Math.abs(pct) / maxVoteAbs) * 100));
                const tone = toneClassFor(pct);
                return (
                  <div key={vote.algorithm} className="vote-row">
                    <div className="vote-row__label">
                      <strong>{vote.algorithm}</strong>
                      <span>w={Math.round((vote.weight || 0) * 100)}%</span>
                    </div>
                    <div className="vote-row__track">
                      <div
                        className={`vote-row__fill vote-row__fill--${tone}`}
                        style={{ width: `${widthPct}%` }}
                      />
                    </div>
                    <span className={`vote-row__val is-${tone}`}>
                      {pct >= 0 ? '+' : ''}
                      {pct.toFixed(2)}%
                    </span>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {topDrivers.length > 0 && (
          <div className="xray-box">
            <div className="xray-box__head">
              <h5>Live Factor Attribution (SHAP)</h5>
              <span>Return contribution drivers</span>
            </div>
            <div className="driver-list">
              {topDrivers.map((drv) => {
                const imp = Number(drv.impact_pct) || 0;
                const widthPct = Math.min(100, Math.max(6, (Math.abs(imp) / maxDriverAbs) * 100));
                const tone = drv.direction === 'bullish' ? 'up' : 'down';
                return (
                  <div key={drv.feature} className="driver-row">
                    <div className="driver-row__top">
                      <span className="driver-row__name">{drv.label}</span>
                      <span className="driver-row__cat">{drv.category}</span>
                    </div>
                    <div className="driver-row__bar-wrap">
                      <div className="driver-row__track">
                        <div
                          className={`driver-row__fill driver-row__fill--${tone}`}
                          style={{ width: `${widthPct}%` }}
                        />
                      </div>
                      <span className={`driver-row__val is-${tone}`}>
                        {imp >= 0 ? '+' : ''}
                        {imp.toFixed(2)}%
                      </span>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </div>

      {termStructure.length > 0 && (
        <div className="term-strip">
          <div className="xray-box__head">
            <h5>Multi-Horizon Ensemble Term Structure</h5>
            <span>1D to 63D calibrated path &amp; 90% cone</span>
          </div>
          <div className="term-grid">
            {termStructure.map((pt) => {
              const ret = Number(pt.predicted_return_pct) || 0;
              const tone = toneClassFor(ret);
              const isActive = Number(pt.horizon) === Number(anchorHorizon);
              return (
                <div key={pt.horizon} className={`term-tile${isActive ? ' is-active' : ''}`}>
                  <div className="term-tile__head">
                    <span className="term-tile__horizon">{pt.label}</span>
                    <span className={`term-tile__ret is-${tone}`}>{formatSigned(ret)}%</span>
                  </div>
                  <div className="term-tile__price">₹{formatPrice(pt.predicted_price)}</div>
                  <div className="term-tile__band">
                    ₹{formatPrice(pt.lower_price)} – ₹{formatPrice(pt.upper_price)}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
};

export default ForecastXRayPanel;
