import React, { useState } from 'react';
import { BrandMark, Callout, Field, Icon, Spinner } from '../../components/common/ui';
import { auth, createUserWithEmailAndPassword } from '../../services/firebaseAuth';

const HIGHLIGHTS = [
  {
    icon: 'storefront',
    title: '50+ tracked NSE constituents',
    detail: 'Live price cards, sector filtering, intraday breadth, and company fundamentals.',
  },
  {
    icon: 'compare_arrows',
    title: 'Dual-model benchmark comparisons',
    detail: 'Evaluate model consensus, confidence bounds, and out-of-sample validation metrics.',
  },
  {
    icon: 'school',
    title: 'Integrated FinTeach & TrendAI',
    detail: 'Ask follow-up questions on stock dynamics or brush up on financial concepts anytime.',
  },
];

const SignupPage = ({ toggleForm }) => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSignup = async (e) => {
    e.preventDefault();
    setError('');
    if (password !== confirmPassword) {
      setError('Passwords do not match.');
      return;
    }
    if (password.length < 8 || !/[A-Za-z]/.test(password) || !/\d/.test(password)) {
      setError('Password must be at least 8 characters and include a letter and a number.');
      return;
    }
    setLoading(true);
    try {
      await createUserWithEmailAndPassword(auth, email, password);
    } catch (err) {
      if (err.code === 'auth/invalid-email') {
        setError('Please enter a valid email address.');
      } else if (err.code === 'auth/weak-password') {
        setError('Password is too weak. Use at least 8 characters with letters and numbers.');
      } else {
        setError('Unable to create account with these details. Please verify your email or log in.');
      }
      if (process.env.NODE_ENV !== 'production') {
        console.error('Signup failed:', err.code);
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="auth-layout">
      <section className="auth-intro">
        <div className="eyebrow">
          <span className="eyebrow__icon">
            <Icon name="auto_awesome" fill size={15} />
          </span>
          <span>Create Your Workspace</span>
        </div>
        <div className="auth-brand-hero">
          <div className="auth-brand-hero__logo">
            <BrandMark size={200} />
          </div>
        </div>
        <p className="auth-intro__lead">
          Set up your NeuroStock account to run ensemble forecasts, screen NSE bluechips, and analyze portfolio risk
          distributions.
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
          <h2>Sign Up</h2>
          <p>Create an account with your email and a password.</p>
        </div>

        <form className="auth-form" onSubmit={handleSignup}>
          <Field label="Email" htmlFor="signup-email">
            <div className="input-shell">
              <Icon name="mail" size={18} className="input-shell__icon" />
              <input
                type="email"
                id="signup-email"
                className="input input--with-icon"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
                autoComplete="email"
                placeholder="name@example.com"
              />
            </div>
          </Field>

          <Field label="Password" hint="Min. 8 chars (letter + number)" htmlFor="signup-password">
            <div className="input-shell">
              <Icon name="lock" size={18} className="input-shell__icon" />
              <input
                type="password"
                id="signup-password"
                className="input input--with-icon"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
                minLength="8"
                autoComplete="new-password"
                placeholder="Create a strong password"
              />
            </div>
          </Field>

          <Field label="Confirm Password" htmlFor="confirm-password">
            <div className="input-shell">
              <Icon name="verified_user" size={18} className="input-shell__icon" />
              <input
                type="password"
                id="confirm-password"
                className="input input--with-icon"
                value={confirmPassword}
                onChange={(e) => setConfirmPassword(e.target.value)}
                required
                autoComplete="new-password"
                placeholder="Re-enter your password"
              />
            </div>
          </Field>

          {error && (
            <Callout tone="error" icon="error">
              {error}
            </Callout>
          )}

          <button type="submit" className="btn btn--primary btn--block" disabled={loading}>
            {loading ? <Spinner size="sm" /> : <Icon name="person_add" size={18} />}
            <span>{loading ? 'Signing up...' : 'Sign Up'}</span>
          </button>
        </form>

        <p className="auth-toggle">
          Already have an account?{' '}
          <button type="button" className="link-btn" onClick={() => toggleForm('login')}>
            Login
          </button>
        </p>
      </section>
    </div>
  );
};

export default SignupPage;
