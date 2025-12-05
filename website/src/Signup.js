import React, { useState } from 'react';
import { auth, createUserWithEmailAndPassword } from './firebase'; // Import from your firebase.js

const Signup = ({ toggleForm }) => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSignup = async (e) => {
    e.preventDefault();
    setError('');
    if (password !== confirmPassword) {
      setError("Passwords do not match.");
      return;
    }
    if (password.length < 6) {
        setError("Password must be at least 6 characters long.");
        return;
    }
    setLoading(true);
    try {
      await createUserWithEmailAndPassword(auth, email, password);
      // Signup successful, App.js's onAuthStateChanged will handle the state change
    } catch (err) {
      // Provide more user-friendly error messages
      if (err.code === 'auth/email-already-in-use') {
        setError('This email address is already registered.');
      } else if (err.code === 'auth/invalid-email') {
        setError('Please enter a valid email address.');
      } else {
        setError('Signup failed. Please try again.');
      }
      console.error("Signup failed:", err.code, err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="auth-container">
      <div className="card auth-card">
        <h2>Sign Up</h2>
        <form onSubmit={handleSignup}>
          <div className="form-group">
            <label htmlFor="signup-email">Email:</label>
            <input
              type="email"
              id="signup-email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
              placeholder="Enter your email"
            />
          </div>
          <div className="form-group">
            <label htmlFor="signup-password">Password (min. 6 characters):</label>
            <input
              type="password"
              id="signup-password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
              minLength="6"
              placeholder="Create a password"
            />
          </div>
           <div className="form-group">
            <label htmlFor="confirm-password">Confirm Password:</label>
            <input
              type="password"
              id="confirm-password"
              value={confirmPassword}
              onChange={(e) => setConfirmPassword(e.target.value)}
              required
              placeholder="Confirm your password"
            />
          </div>
          {error && <p className="error-message auth-error">{error}</p>}
          <button type="submit" className="cta-button auth-button" disabled={loading}>
            {loading ? (
              <svg className="button-spinner mini-spinner" viewBox="0 0 50 50"> {/* Added mini-spinner class */}
                 <circle className="path" cx="25" cy="25" r="20" fill="none" strokeWidth="5"></circle>
              </svg>
            ) : null}
            <span>{loading ? 'Signing up...' : 'Sign Up'}</span>
          </button>
        </form>
         <p className="auth-toggle">
          Already have an account? <button type="button" onClick={() => toggleForm('login')}>Login</button>
        </p>
      </div>
    </div>
  );
};

export default Signup;