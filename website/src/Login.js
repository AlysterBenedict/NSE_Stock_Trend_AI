import React, { useState } from 'react';
import { auth, signInWithEmailAndPassword } from './firebase'; // Import from your firebase.js

const Login = ({ toggleForm }) => {
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
      // Login successful, App.js's onAuthStateChanged will handle the redirect/state change
    } catch (err) {
      // Provide more user-friendly error messages
      if (err.code === 'auth/invalid-credential' || err.code === 'auth/invalid-email' || err.code === 'auth/wrong-password') {
        setError('Invalid email or password.');
      } else {
        setError('Login failed. Please try again.');
      }
      console.error("Login failed:", err.code, err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="auth-container">
      <div className="card auth-card">
        <h2>Login</h2>
        <form onSubmit={handleLogin}>
          <div className="form-group">
            <label htmlFor="login-email">Email:</label>
            <input
              type="email"
              id="login-email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
              placeholder="Enter your email"
            />
          </div>
          <div className="form-group">
            <label htmlFor="login-password">Password:</label>
            <input
              type="password"
              id="login-password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
              placeholder="Enter your password"
            />
          </div>
          {error && <p className="error-message auth-error">{error}</p>}
          <button type="submit" className="cta-button auth-button" disabled={loading}>
            {loading ? (
              <svg className="button-spinner mini-spinner" viewBox="0 0 50 50"> {/* Added mini-spinner class */}
                 <circle className="path" cx="25" cy="25" r="20" fill="none" strokeWidth="5"></circle>
              </svg>
            ) : null}
            <span>{loading ? 'Logging in...' : 'Login'}</span>
          </button>
        </form>
        <p className="auth-toggle">
          Don't have an account? <button type="button" onClick={() => toggleForm('signup')}>Sign Up</button>
        </p>
      </div>
    </div>
  );
};

export default Login;