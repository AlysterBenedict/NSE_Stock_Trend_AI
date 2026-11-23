import React, { useEffect, useRef, useState } from 'react';
import { useTheme } from '../../context/ThemeContext';
import LlmSettingsPopup from '../common/LlmSettingsPopup';
import { BrandMark, Icon, NvidiaMark } from '../common/ui';

/** NSE cash session: Mon–Fri, 09:15–15:30 IST. */
function getNseSession(now = new Date()) {
  const ist = new Date(now.getTime() + (now.getTimezoneOffset() + 330) * 60000);
  const day = ist.getDay();
  const minutes = ist.getHours() * 60 + ist.getMinutes();
  const open = day >= 1 && day <= 5 && minutes >= 555 && minutes <= 930;
  const time = ist.toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit', hour12: false });
  return { open, time };
}

const AppHeader = ({
  user,
  onLogout,
  tabGroups = [],
  activeTab,
  onTabChange,
  onOpenCommandPalette,
  onOpenWatchlist,
  watchlistCount = 0,
  llmSettings = null,
  onSaveLlmKey,
  onDeleteLlmKey,
  onRefreshLlmSettings,
}) => {
  const [session, setSession] = useState(getNseSession());
  const [llmPopupOpen, setLlmPopupOpen] = useState(false);
  const llmPillRef = useRef(null);
  const { isDark, toggleTheme } = useTheme();

  useEffect(() => {
    const id = setInterval(() => setSession(getNseSession()), 30000);
    return () => clearInterval(id);
  }, []);

  useEffect(() => {
    if (!user) {
      setLlmPopupOpen(false);
    }
  }, [user]);

  const initial = (user?.email || 'U').charAt(0).toUpperCase();
  const lmOnline = Boolean(llmSettings?.lm_studio_available);
  const nvidiaOnline = Boolean(llmSettings?.has_api_key || llmSettings?.nvidia_llm_online);
  const isLlmOnline = Boolean(lmOnline || nvidiaOnline || llmSettings?.llm_configured);
  const llmDisplayName = lmOnline
    ? llmSettings?.local_model_display || 'Local LLM'
    : nvidiaOnline
    ? llmSettings?.nvidia_model_display || 'Nemotron 550B'
    : 'LLM Offline';

  const handleToggleLlmPopup = () => {
    const nextOpen = !llmPopupOpen;
    setLlmPopupOpen(nextOpen);
    if (nextOpen) {
      onRefreshLlmSettings?.();
    }
  };

  return (
    <header className="app-header">
      <div className="app-header__bar">
        <div className="brand">
          <BrandMark />
          <div className="brand__text">
            <span className="brand__name">NeuroStock</span>
            <span className="brand__tagline">NSE forecasting suite</span>
          </div>
        </div>

        <div className="app-header__right">
          <div className={`session-pill ${session.open ? 'is-open' : 'is-closed'}`}>
            <span className="session-pill__dot" />
            <span className="session-pill__label">NSE {session.open ? 'Open' : 'Closed'}</span>
            <span className="session-pill__sep" />
            <Icon name="schedule" size={15} />
            <span className="session-pill__time">{session.time} IST</span>
          </div>

          <button
            type="button"
            className="theme-toggle"
            onClick={toggleTheme}
            aria-label={isDark ? 'Switch to light mode' : 'Switch to dark mode'}
            aria-pressed={isDark}
            title={isDark ? 'Switch to light mode' : 'Switch to dark mode'}
          >
            <Icon name={isDark ? 'light_mode' : 'dark_mode'} size={18} />
          </button>

          {user && (
            <>
              <button
                type="button"
                className="header-pill-btn"
                onClick={onOpenCommandPalette}
                aria-label="Open command search (Ctrl+K)"
                title="Quick Search (Ctrl+K)"
              >
                <Icon name="search" size={16} />
                <span className="header-pill-btn__label">Search</span>
                <kbd className="header-pill-btn__kbd">Ctrl+K</kbd>
              </button>

              <button
                type="button"
                className="header-pill-btn"
                onClick={onOpenWatchlist}
                aria-label="Open watchlist"
                title="Watchlist & Price Alerts"
              >
                <Icon name="star" size={16} fill={watchlistCount > 0} />
                <span className="header-pill-btn__label">Watchlist</span>
                {watchlistCount > 0 && <span className="header-pill-btn__count">{watchlistCount}</span>}
              </button>

              <div className="llm-pill-anchor">
                <button
                  ref={llmPillRef}
                  type="button"
                  className={`header-pill-btn llm-pill-btn${isLlmOnline ? ' is-online' : ' is-offline'}`}
                  onClick={handleToggleLlmPopup}
                  aria-expanded={llmPopupOpen}
                  aria-haspopup="dialog"
                  aria-label={`LLM Connection Settings: ${llmDisplayName} (${isLlmOnline ? 'Online' : 'Offline'})`}
                  title={`LLM Connection: ${llmDisplayName} (${isLlmOnline ? 'Online' : 'Click to configure NVIDIA API key'})`}
                >
                  <NvidiaMark size={16} className="llm-pill-btn__logo" />
                  <span className="llm-pill-btn__label">{llmDisplayName}</span>
                  <span
                    className={`llm-pill-btn__dot ${isLlmOnline ? 'llm-pill-btn__dot--green' : 'llm-pill-btn__dot--red'}`}
                    aria-hidden="true"
                  />
                </button>

                <LlmSettingsPopup
                  isOpen={llmPopupOpen}
                  anchorRef={llmPillRef}
                  llmSettings={llmSettings}
                  onClose={() => setLlmPopupOpen(false)}
                  onSaveKey={onSaveLlmKey}
                  onDeleteKey={onDeleteLlmKey}
                />
              </div>

              <div className="user-chip">
                <span className="user-chip__avatar">{initial}</span>
                <span className="user-chip__email" title={user.email}>
                  {user.email || 'User'}
                </span>
                <button type="button" className="icon-btn" onClick={onLogout} title="Log out" aria-label="Log out">
                  <Icon name="logout" size={18} />
                </button>
              </div>
            </>
          )}
        </div>
      </div>

      {user && tabGroups.length > 0 && (
        <nav className="app-nav" aria-label="Primary">
          <div className="app-nav__track" role="tablist">
            {tabGroups.map((group, gi) => (
              <React.Fragment key={group.label}>
                {gi > 0 && <span className="app-nav__divider" aria-hidden="true" />}
                <div className="app-nav__group" aria-label={group.label}>
                  {group.tabs.map((tab) => {
                    const active = activeTab === tab.id;
                    return (
                      <button
                        key={tab.id}
                        type="button"
                        role="tab"
                        aria-selected={active}
                        className={`tab-item${active ? ' active' : ''}`}
                        onClick={() => onTabChange(tab.id)}
                      >
                        <Icon name={tab.icon} size={18} fill={active} />
                        <span>{tab.label}</span>
                      </button>
                    );
                  })}
                </div>
              </React.Fragment>
            ))}
          </div>
        </nav>
      )}
    </header>
  );
};

export default AppHeader;
