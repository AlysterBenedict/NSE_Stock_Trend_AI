import React from 'react';
import { Callout, Card, Icon } from './ui';

class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError() {
    return { hasError: true };
  }

  componentDidCatch(err) {
    if (process.env.NODE_ENV !== 'production') {
      console.error('View crashed:', err);
    }
  }

  handleReset = () => {
    this.setState({ hasError: false });
    if (typeof this.props.onReset === 'function') {
      this.props.onReset();
    }
  };

  render() {
    if (this.state.hasError) {
      return (
        <Card className="result-card--state">
          <Callout tone="error" icon="error" title="Unexpected Display Error">
            This dashboard view encountered a rendering error. Resetting the view usually recovers cleanly.
          </Callout>
          <div style={{ marginTop: 16 }}>
            <button type="button" className="btn btn--secondary" onClick={this.handleReset}>
              <Icon name="refresh" size={18} />
              <span>Reload View</span>
            </button>
          </div>
        </Card>
      );
    }
    return this.props.children;
  }
}

export default ErrorBoundary;
