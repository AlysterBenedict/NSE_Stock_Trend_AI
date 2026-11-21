import React, { useEffect, useRef, useState } from 'react';
import SearchableSelect from '../../components/common/SearchableSelect';
import {
  Badge,
  Callout,
  Card,
  CardHeader,
  EmptyState,
  Field,
  Icon,
  Loader,
  PageHeader,
  Spinner,
} from '../../components/common/ui';
import { SENTIMENT_STOCK_OPTIONS } from '../../constants/stocks';
import { getStockSentiment } from '../../services/apiClient';
import { formatUnixOrIsoDate, isCancelledError } from '../../utils/formatters';

const scoreTone = (score) => {
  if (score >= 65) return 'up';
  if (score >= 40) return 'secondary';
  return 'down';
};

const scoreStroke = (score) => {
  if (score >= 65) return 'var(--primary)';
  if (score >= 40) return 'var(--secondary)';
  return 'var(--tertiary)';
};

/** 180-degree semicircular gauge calibrated to 0–100. */
const SentimentGauge = ({ score }) => {
  const clamped = Math.max(0, Math.min(100, Number(score) || 0));
  const radius = 78;
  const arcLength = Math.PI * radius;
  const offset = arcLength * (1 - clamped / 100);

  return (
    <div className="sentiment-arc">
      <svg viewBox="0 0 200 116" className="sentiment-arc__svg" aria-hidden="true">
        <path
          d="M 22 100 A 78 78 0 0 1 178 100"
          fill="none"
          stroke="var(--surface-high)"
          strokeWidth="14"
          strokeLinecap="round"
        />
        <path
          d="M 22 100 A 78 78 0 0 1 178 100"
          fill="none"
          stroke={scoreStroke(clamped)}
          strokeWidth="14"
          strokeLinecap="round"
          strokeDasharray={arcLength}
          strokeDashoffset={offset}
          style={{ transition: 'stroke-dashoffset 600ms cubic-bezier(0.2, 0.7, 0.2, 1)' }}
        />
      </svg>
      <div className="sentiment-arc__readout">
        <span className="sentiment-arc__score" style={{ color: scoreStroke(clamped) }}>
          {clamped}
        </span>
        <span className="sentiment-arc__scale">out of 100</span>
      </div>
    </div>
  );
};

const NewsSentimentDashboard = () => {
  const [stock, setStock] = useState('');
  const [sentimentData, setSentimentData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [hasAnalyzed, setHasAnalyzed] = useState(false);
  const abortControllerRef = useRef(null);

  useEffect(() => () => abortControllerRef.current?.abort(), []);

  const fetchSentiment = async () => {
    if (!stock) return;
    abortControllerRef.current?.abort();
    const controller = new AbortController();
    abortControllerRef.current = controller;

    setLoading(true);
    setError(null);
    setSentimentData(null);
    setHasAnalyzed(true);
    try {
      const data = await getStockSentiment(stock, controller.signal);
      if (controller.signal.aborted) return;
      setSentimentData(data);
    } catch (err) {
      if (controller.signal.aborted || isCancelledError(err)) {
        return;
      }
      console.error('Sentiment Error:', err);
      setError(`Failed to fetch AI sentiment. ${err.response?.data?.error || err.message}`);
    } finally {
      if (abortControllerRef.current === controller) {
        setLoading(false);
      }
    }
  };

  return (
    <div className="page-stack">
      <PageHeader
        eyebrow="FinBERT News Intelligence"
        icon="newspaper"
        title="Market Sentiment"
        description="Analyze recent financial headlines with FinBERT scoring to gauge bullish, neutral, or bearish narrative momentum."
      />

      <Card className="sentiment-toolbar">
        <div className="sentiment-toolbar__row">
          <div className="sentiment-toolbar__select">
            <Field label="Equity" hint="NSE">
              <SearchableSelect
                options={SENTIMENT_STOCK_OPTIONS}
                value={stock}
                onChange={setStock}
                placeholder="Select stock..."
              />
            </Field>
          </div>
          <button
            type="button"
            className="btn btn--primary"
            onClick={fetchSentiment}
            disabled={loading || !stock}
          >
            {loading ? <Spinner size="sm" /> : <Icon name="analytics" size={18} />}
            <span>{loading ? 'Analyzing...' : 'Analyze Sentiment'}</span>
          </button>
        </div>
      </Card>

      {!hasAnalyzed ? (
        <Card className="result-card--state">
          <EmptyState
            icon="newspaper"
            title="Ready to Analyze Market Sentiment"
            description="Select an equity above and run analysis to decode live news headlines and compute an aggregate FinBERT mood score."
          />
        </Card>
      ) : (
        <div className="bento-grid">
          <div className="bento-aside">
            <Card className="sticky-panel">
              <CardHeader
                icon="speed"
                iconTone="primary"
                title="Market Mood"
                subtitle={stock}
              />

              {loading ? (
                <Loader label="Scoring recent headlines..." />
              ) : error ? (
                <Callout tone="error" icon="error">
                  {error}
                </Callout>
              ) : sentimentData ? (
                <div className="mood-body">
                  <SentimentGauge score={sentimentData.score} />

                  <div className="mood-badge-row">
                    <Badge tone={scoreTone(sentimentData.score)}>{sentimentData.label}</Badge>
                  </div>

                  <div className="mood-bar-group">
                    <div className="mood-bar-track">
                      <div
                        className="mood-bar-fill"
                        style={{
                          width: `${Math.max(0, Math.min(100, sentimentData.score))}%`,
                          background: scoreStroke(sentimentData.score),
                        }}
                      />
                    </div>
                    <div className="mood-bar-labels">
                      <span>Bearish</span>
                      <span>Neutral</span>
                      <span>Bullish</span>
                    </div>
                  </div>

                  {sentimentData.summary && (
                    <blockquote className="mood-quote">{sentimentData.summary}</blockquote>
                  )}
                </div>
              ) : null}
            </Card>
          </div>

          <div className="bento-main">
            <Card>
              <CardHeader
                icon="feed"
                iconTone="secondary"
                title="FinBERT-Scored News Feed"
                subtitle="Latest coverage ranked by publication time"
                actions={
                  sentimentData?.news?.length > 0 && (
                    <Badge tone="neutral">{sentimentData.news.length} Headlines</Badge>
                  )
                }
              />

              {loading ? (
                <div className="shimmer-stack">
                  <span className="shimmer-line" />
                  <span className="shimmer-line" />
                  <span className="shimmer-line" />
                  <span className="shimmer-line" />
                </div>
              ) : sentimentData?.news?.length > 0 ? (
                <div className="news-list">
                  {sentimentData.news.map((item, idx) => {
                    const hasScore = item.sentiment_score !== undefined;
                    return (
                      <article key={idx} className="news-row">
                        <div className="news-row__body">
                          <a
                            href={item.link}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="news-row__title"
                          >
                            <span>{item.title}</span>
                            <Icon name="open_in_new" size={14} className="news-row__ext" />
                          </a>
                          <span className="news-row__time">
                            <Icon name="schedule" size={14} />
                            {formatUnixOrIsoDate(item.time)}
                          </span>
                        </div>
                        <Badge tone={hasScore ? scoreTone(item.sentiment_score) : 'neutral'}>
                          {item.sentiment_label
                            ? `${item.sentiment_label} (${item.sentiment_score})`
                            : 'FinBERT Scored'}
                        </Badge>
                      </article>
                    );
                  })}
                </div>
              ) : (
                !loading && (
                  <EmptyState
                    icon="article"
                    title="No headlines available"
                    description="No recent news items were returned for this stock."
                  />
                )
              )}
            </Card>
          </div>
        </div>
      )}
    </div>
  );
};

export default NewsSentimentDashboard;
