import React, { useEffect, useState } from 'react';
import { useTheme } from '../../context/ThemeContext';
import { BrandMark, Icon } from '../common/ui';

/** NSE cash session: Mon–Fri, 09:15–15:30 IST. */
function getNseSession(now = new Date()) {
  const ist = new Date(now.getTime() + (now.getTimezoneOffset() + 330) * 60000);
  const day = ist.getDay();
  const minutes = ist.getHours() * 60 + ist.getMinutes();
  const open = day >= 1 && day <= 5 && minutes >= 555 && minutes <= 930;
  const time = ist.toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit', hour12: false });
  return { open, time };
}

const AppHeader = ({ user, onLogout, tabGroups = [], activeTab, onTabChange }) => {
  const [session, setSession] = useState(getNseSession());
  const { isDark, toggleTheme } = useTheme();

  useEffect(() => {
    const id = setInterval(() => setSession(getNseSession()), 30000);
    return () => clearInterval(id);
  }, []);

  const initial = (user?.email || 'U').charAt(0).toUpperCase();

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
            <div className="user-chip">
              <span className="user-chip__avatar">{initial}</span>
              <span className="user-chip__email" title={user.email}>
                {user.email || 'User'}
              </span>
              <button type="button" className="icon-btn" onClick={onLogout} title="Log out" aria-label="Log out">
                <Icon name="logout" size={18} />
              </button>
            </div>
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
