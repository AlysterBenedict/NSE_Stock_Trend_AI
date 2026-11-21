import React, { useState } from 'react';
import { BrandMark, Callout, Field, Icon, Spinner } from '../../components/common/ui';
import { auth, signInWithEmailAndPassword } from '../../services/firebaseAuth';

const HIGHLIGHTS = [
  {
    icon: 'query_stats',
    title: 'Multi-model price forecasts and comparison',
    detail: 'Stacking Ensemble, AttentionNet, LightGBM, CatBoost, and XGBoost with 90% confidence intervals.',
  },
  {
    icon: 'account_balance_wallet',
    title: 'Portfolio simulation and Investment recommendation engine',
    detail: 'Portfolio Optimization, Monte Carlo Simulation, Sharpe Ratio, Max Drawdown, Scenario Analysis.',
  },
  {
    icon: 'newspaper',
    title: 'News sentiment',
    detail: 'Real-time headline scoring paired with technical charts and an interactive AI assistant.',
  },
];

const LoginPage = ({ toggleForm }) => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handleLogin = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);
    try {
      await signInWithEmailAndPassword(auth, email, password);
    } catch (err) {
      if (
        err.code === 'auth/invalid-credential' ||
        err.code === 'auth/invalid-email' ||
        err.code === 'auth/wrong-password'
      ) {
        setError('Invalid email or password.');
      } else {
        setError('Login failed. Please try again.');
      }
      console.error('Login failed:', err.code, err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="auth-layout">
      <section className="auth-intro">
        <div className="eyebrow">
          <span className="eyebrow__icon">
            <Icon name="insights" fill size={15} />
          </span>
          <span>Quantitative NSE Analytics</span>
        </div>
        <div className="auth-brand-hero">
          <div className="auth-brand-hero__logo">
            <BrandMark size={200} />
          </div>
          
        </div>
        <p className="auth-intro__lead">
          Track live NSE equities, predict stock prices, compare ML forecast models side by side, simulate portfolio
          trajectories, and chat with TrendAI in one workspace.
        </p>

        <ul className="auth-highlights">
          {HIGHLIGHTS.map((item) => (
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

      <section className="card auth-card">
        <div className="auth-card__header">
          <h2>Login</h2>
          <p>Sign in with your account to access forecasting tools.</p>
        </div>

        <form className="auth-form" onSubmit={handleLogin}>
          <Field label="Email" htmlFor="login-email">
            <div className="input-shell">
              <Icon name="mail" size={18} className="input-shell__icon" />
              <input
                type="email"
                id="login-email"
                className="input input--with-icon"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
                autoComplete="email"
                placeholder="name@example.com"
              />
            </div>
          </Field>

          <Field label="Password" htmlFor="login-password">
            <div className="input-shell">
              <Icon name="lock" size={18} className="input-shell__icon" />
              <input
                type="password"
                id="login-password"
                className="input input--with-icon"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
                autoComplete="current-password"
                placeholder="Enter your password"
              />
            </div>
          </Field>

          {error && (
            <Callout tone="error" icon="error">
              {error}
            </Callout>
          )}

          <button type="submit" className="btn btn--primary btn--block" disabled={loading}>
            {loading ? <Spinner size="sm" /> : <Icon name="login" size={18} />}
            <span>{loading ? 'Logging in...' : 'Login'}</span>
          </button>
        </form>

        <p className="auth-toggle">
          Don&apos;t have an account?{' '}
          <button type="button" className="link-btn" onClick={() => toggleForm('signup')}>
            Sign Up
          </button>
        </p>
      </section>
    </div>
  );
};

export default LoginPage;
