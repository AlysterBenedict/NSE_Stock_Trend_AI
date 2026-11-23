import React, { useEffect, useRef, useState } from 'react';
import { Callout, Field, Icon, NvidiaMark, Spinner } from './ui';

const LlmSettingsPopup = ({
  isOpen,
  anchorRef,
  llmSettings,
  onClose,
  onSaveKey,
  onDeleteKey,
}) => {
  const popupRef = useRef(null);
  const [apiKey, setApiKey] = useState('');
  const [showKey, setShowKey] = useState(false);
  const [saving, setSaving] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [error, setError] = useState('');
  const [feedback, setFeedback] = useState('');

  useEffect(() => {
    if (!isOpen) {
      setError('');
      setFeedback('');
      setApiKey('');
      setShowKey(false);
      return undefined;
    }

    const handleKeyDown = (e) => {
      if (e.key === 'Escape') {
        onClose();
      }
    };
    const handlePointerDown = (e) => {
      if (
        popupRef.current &&
        !popupRef.current.contains(e.target) &&
        !(anchorRef?.current && anchorRef.current.contains(e.target))
      ) {
        onClose();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    document.addEventListener('mousedown', handlePointerDown);
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      document.removeEventListener('mousedown', handlePointerDown);
    };
  }, [isOpen, onClose, anchorRef]);

  if (!isOpen) return null;

  const lmOnline = Boolean(llmSettings?.lm_studio_available);
  const hasApiKey = Boolean(llmSettings?.has_api_key);
  const nvidiaOnline = Boolean(hasApiKey || llmSettings?.nvidia_llm_online);
  const isAnyOnline = Boolean(lmOnline || nvidiaOnline || llmSettings?.llm_configured);

  const activeDisplay = lmOnline
    ? llmSettings?.local_model_display || 'Local LLM'
    : nvidiaOnline
    ? llmSettings?.nvidia_model_display || 'Nemotron 550B'
    : 'LLM Offline';

  const handleSave = async (e) => {
    e.preventDefault();
    const trimmed = apiKey.trim();
    if (trimmed.length < 20 || /\s/.test(trimmed)) {
      setError('Enter a valid NVIDIA API key (20-256 characters, no spaces).');
      setFeedback('');
      return;
    }

    setError('');
    setFeedback('');
    setSaving(true);
    try {
      await onSaveKey(trimmed);
      setApiKey('');
      setFeedback('NVIDIA API key encrypted & saved.');
    } catch (err) {
      setError(
        err?.response?.data?.error ||
          err?.message ||
          'Unable to save NVIDIA API key.'
      );
    } finally {
      setSaving(false);
    }
  };

  const handleDelete = async () => {
    setError('');
    setFeedback('');
    setDeleting(true);
    try {
      await onDeleteKey();
      setFeedback('Saved NVIDIA API key removed.');
    } catch (err) {
      setError(err?.response?.data?.error || 'Unable to remove saved NVIDIA API key.');
    } finally {
      setDeleting(false);
    }
  };

  return (
    <section
      ref={popupRef}
      className="card llm-popup-card"
      role="dialog"
      aria-label="LLM Connection Settings"
    >
      <header className="llm-popup-card__head">
        <div className="llm-popup-card__brand">
          <span className="llm-popup-card__logo-wrap">
            <NvidiaMark size={24} />
          </span>
          <div>
            <h3 className="llm-popup-card__title">LLM Connection</h3>
            <span className="llm-popup-card__sub">LM Studio Primary · NVIDIA Failover</span>
          </div>
        </div>

        <div className="llm-popup-card__head-right">
          <span className={`badge ${isAnyOnline ? 'badge--up' : 'badge--error'}`}>
            <span className={`llm-status-dot ${isAnyOnline ? 'is-green' : 'is-red'}`} />
            {activeDisplay}
          </span>
          <button
            type="button"
            className="icon-btn"
            onClick={onClose}
            aria-label="Close LLM settings"
          >
            <Icon name="close" size={18} />
          </button>
        </div>
      </header>

      <div className="llm-status-list">
        <div className="llm-status-item">
          <div className="llm-status-item__label">
            <span className={`llm-status-dot ${lmOnline ? 'is-green' : 'is-red'}`} />
            <div>
              <strong>1. Local LM Studio</strong>
              <small>{llmSettings?.local_model_display || 'Local LLM'} ({llmSettings?.local_model || 'google/gemma-4-26b-a4b'})</small>
            </div>
          </div>
          <span className={`llm-status-item__state ${lmOnline ? 'is-online' : 'is-offline'}`}>
            {lmOnline ? 'Online' : 'Offline'}
          </span>
        </div>

        <div className="llm-status-item">
          <div className="llm-status-item__label">
            <span className={`llm-status-dot ${nvidiaOnline ? 'is-green' : 'is-red'}`} />
            <div>
              <strong>2. NVIDIA NIM Fallback</strong>
              <small>
                {llmSettings?.nvidia_model_display || 'Nemotron 550B'} ({llmSettings?.nvidia_model || 'nvidia/nemotron-3-ultra-550b-a55b'})
              </small>
            </div>
          </div>
          <span className={`llm-status-item__state ${nvidiaOnline ? 'is-online' : 'is-offline'}`}>
            {hasApiKey
              ? llmSettings?.masked_key || 'Key Saved'
              : nvidiaOnline
              ? 'Env Key'
              : 'Not Set'}
          </span>
        </div>
      </div>

      <form className="llm-popup-form" onSubmit={handleSave}>
        <Field
          label="NVIDIA API Key"
          hint={hasApiKey ? 'Update saved key' : 'Tied to your account'}
          htmlFor="popup-nvidia-key"
        >
          <div className="input-shell">
            <Icon name="key" size={17} className="input-shell__icon" />
            <input
              type={showKey ? 'text' : 'password'}
              id="popup-nvidia-key"
              className="input input--with-icon input--with-action"
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              autoComplete="off"
              spellCheck={false}
              placeholder={hasApiKey ? 'Enter new key to replace saved key...' : 'nvapi-...'}
            />
            <button
              type="button"
              className="input-shell__toggle"
              onClick={() => setShowKey((prev) => !prev)}
              aria-label={showKey ? 'Hide API key' : 'Show API key'}
            >
              <Icon name={showKey ? 'visibility_off' : 'visibility'} size={17} />
            </button>
          </div>
        </Field>

        {error && (
          <Callout tone="error" icon="error">
            {error}
          </Callout>
        )}

        {feedback && !error && (
          <Callout tone="neutral" icon="check_circle">
            {feedback}
          </Callout>
        )}

        <div className="llm-popup-form__actions">
          <button
            type="submit"
            className="btn btn--primary btn--sm"
            disabled={saving || deleting || !apiKey.trim()}
          >
            {saving ? <Spinner size="sm" /> : <Icon name="save" size={15} />}
            <span>{hasApiKey ? 'Update Key' : 'Save API Key'}</span>
          </button>

          {hasApiKey && (
            <button
              type="button"
              className="btn btn--secondary btn--sm"
              onClick={handleDelete}
              disabled={saving || deleting}
            >
              {deleting ? <Spinner size="sm" /> : <Icon name="delete_outline" size={15} />}
              <span>Remove Key</span>
            </button>
          )}
        </div>
      </form>
    </section>
  );
};

export default LlmSettingsPopup;
