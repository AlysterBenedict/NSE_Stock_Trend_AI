import React, { useState } from 'react';
import { BrandMark, Callout, Field, Icon, NvidiaMark, Spinner } from '../../components/common/ui';

const CONFIG_HIGHLIGHTS = [
  {
    icon: 'dns',
    title: 'Local LLM Primary Routing',
    detail: 'Every AI request checks your local LLM server first for zero-latency private inference.',
  },
  {
    icon: 'cloud_sync',
    title: 'Automatic NVIDIA NIM failover (Nemotron 550B)',
    detail: 'When your local LLM is offline, requests seamlessly route to nvidia/nemotron-3-ultra-550b-a55b.',
  },
  {
    icon: 'enhanced_encryption',
    title: 'Encrypted & tied to your account',
    detail: 'Stored with per-user AES-256-GCM encryption so future logins jump straight into the workspace.',
  },
];

const LlmConfigCard = ({ user, llmSettings, onSaveKey, onSkip }) => {
  const [apiKey, setApiKey] = useState('');
  const [showKey, setShowKey] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    const trimmed = apiKey.trim();
    if (trimmed.length < 20 || /\s/.test(trimmed)) {
      setError('Enter a valid NVIDIA API key (20-256 characters, no spaces).');
      return;
    }

    setError('');
    setSaving(true);
    try {
      await onSaveKey(trimmed);
    } catch (err) {
      setError(
        err?.response?.data?.error ||
          err?.message ||
          'Failed to save NVIDIA API key. Please verify and try again.'
      );
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="auth-layout">
      <section className="auth-intro">
        <div className="eyebrow">
          <span className="eyebrow__icon">
            <Icon name="hub" fill size={15} />
          </span>
          <span>Hybrid LLM Routing</span>
        </div>
        <div className="auth-brand-hero">
          <div className="auth-brand-hero__logo">
            <BrandMark size={200} />
          </div>
        </div>
        <p className="auth-intro__lead">
          Configure an optional NVIDIA API key for cloud failover whenever your local LLM is offline.
        </p>

        <ul className="auth-highlights">
          {CONFIG_HIGHLIGHTS.map((item) => (
            <li key={item.title} className="auth-highlight">
              <span className="icon-tile icon-tile--primary">
                <Icon name={item.icon} size={18} />
              </span>
              <div>
                <strong>{item.title}</strong>
                <p>{item.detail}</p>
              </div>
            </li>
          ))}
        </ul>
      </section>

      <section className="card auth-card llm-auth-card" aria-label="NVIDIA LLM Configuration">
        <div className="llm-auth-card__brand">
          <span className="llm-auth-card__logo-badge">
            <NvidiaMark size={32} />
          </span>
          <span className={`badge ${llmSettings?.lm_studio_available ? 'badge--up' : 'badge--neutral'}`}>
            <span
              className={`llm-status-dot ${
                llmSettings?.lm_studio_available ? 'is-green' : 'is-muted'
              }`}
            />
            {llmSettings?.lm_studio_available ? 'Local LLM Online' : 'LM Studio Offline'}
          </span>
        </div>

        <div className="auth-card__header">
          <h2>NVIDIA LLM Setup</h2>
          <p>
            Signed in as <strong>{user?.email || 'Account'}</strong>. Add your NVIDIA API key for{' '}
            <code>Nemotron 550B</code> fallback, or skip to continue.
          </p>
        </div>

        <form className="auth-form" onSubmit={handleSubmit}>
          <Field
            label="NVIDIA API Key"
            hint="nvidia/nemotron-3-ultra-550b-a55b"
            htmlFor="onboarding-nvidia-key"
          >
            <div className="input-shell">
              <Icon name="key" size={18} className="input-shell__icon" />
              <input
                type={showKey ? 'text' : 'password'}
                id="onboarding-nvidia-key"
                className="input input--with-icon input--with-action"
                value={apiKey}
                onChange={(e) => setApiKey(e.target.value)}
                autoComplete="off"
                spellCheck={false}
                placeholder="nvapi-..."
              />
              <button
                type="button"
                className="input-shell__toggle"
                onClick={() => setShowKey((prev) => !prev)}
                aria-label={showKey ? 'Hide API key' : 'Show API key'}
              >
                <Icon name={showKey ? 'visibility_off' : 'visibility'} size={18} />
              </button>
            </div>
          </Field>

          <p className="llm-security-note">
            <Icon name="lock" size={14} />
            <span>AES-256-GCM encrypted &amp; bound to your account. Future logins open directly.</span>
          </p>

          {error && (
            <Callout tone="error" icon="error">
              {error}
            </Callout>
          )}

          <div className="auth-card__actions">
            <button
              type="submit"
              className="btn btn--primary btn--block"
              disabled={saving || !apiKey.trim()}
            >
              {saving ? <Spinner size="sm" /> : <Icon name="verified_user" size={18} />}
              <span>{saving ? 'Encrypting & Saving...' : 'Save & Continue'}</span>
            </button>
            <button
              type="button"
              className="btn btn--secondary btn--block"
              onClick={onSkip}
              disabled={saving}
            >
              <span>Skip</span>
              <Icon name="arrow_forward" size={17} />
            </button>
          </div>
        </form>
      </section>
    </div>
  );
};

export default LlmConfigCard;
