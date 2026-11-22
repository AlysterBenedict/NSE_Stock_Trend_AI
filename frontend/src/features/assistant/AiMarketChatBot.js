import React, { useRef, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Icon } from '../../components/common/ui';
import { useChatSession } from './useChatSession';

const DEFAULT_SUGGESTIONS = [
  'What is the current price of ',
  'Predict the stock price for ',
  'Should I invest in ',
  'Compare Infosys vs TCS',
  'Analyze the trend for ',
];

const AiMarketChatBot = ({ onGetInsights, stockName, activeContext = null }) => {
  const [isOpen, setIsOpen] = useState(false);
  const [isFullScreen, setIsFullScreen] = useState(false);
  const inputRef = useRef(null);

  const focusedStock =
    activeContext?.stockName || (stockName && stockName !== 'Market Assistant' ? stockName : '');

  const suggestions = focusedStock
    ? [
        `Explain the AI forecast & key drivers for ${focusedStock}`,
        `Bull vs bear case & volatility risk for ${focusedStock}`,
        `Latest FinBERT news catalysts for ${focusedStock}`,
        'Compare Infosys vs TCS',
      ]
    : DEFAULT_SUGGESTIONS;

  const {
    messages,
    input,
    setInput,
    isLoading,
    messagesEndRef,
    handleSend,
    handleKeyPress,
  } = useChatSession(
    "Hello! I'm TrendAI, your market research assistant. Ask me about stock prices, trend forecasts, or company comparisons.",
    onGetInsights,
    'I encountered an error while fetching insights. Please try again.',
    isOpen
  );

  const toggleChat = () => {
    if (isOpen && isFullScreen) {
      setIsFullScreen(false);
    }
    setIsOpen(!isOpen);
  };

  const toggleFullScreen = (e) => {
    e.stopPropagation();
    setIsFullScreen(!isFullScreen);
  };

  const handleSuggestionClick = (text) => {
    setInput(text);
    inputRef.current?.focus();
  };

  return (
    <div className={`assistant-widget${isFullScreen ? ' is-fullscreen' : ''}`}>
      {isOpen && (
        <section
          className={`assistant-panel${isFullScreen ? ' is-fullscreen' : ''}`}
          aria-label="TrendAI Market Assistant"
        >
          <header className="assistant-panel__head">
            <div className="assistant-panel__brand">
              <span className="assistant-panel__avatar">
                <Icon name="auto_awesome" size={18} />
              </span>
              <div>
                <h3 className="assistant-panel__title">TrendAI</h3>
                <span className="assistant-panel__context">
                  Context · {stockName || 'Market Assistant'}
                </span>
              </div>
            </div>

            <div className="assistant-panel__actions">
              <button
                type="button"
                className="icon-btn"
                onClick={toggleFullScreen}
                title={isFullScreen ? 'Exit Fullscreen' : 'Expand'}
                aria-label={isFullScreen ? 'Exit Fullscreen' : 'Expand'}
              >
                <Icon name={isFullScreen ? 'close_fullscreen' : 'open_in_full'} size={18} />
              </button>
              <button
                type="button"
                className="icon-btn"
                onClick={toggleChat}
                title="Close"
                aria-label="Close assistant"
              >
                <Icon name="close" size={19} />
              </button>
            </div>
          </header>

          <div className="assistant-panel__body">
            {messages.map((msg, index) => (
              <div key={index} className={`chat-row chat-row--compact chat-row--${msg.role}`}>
                <div className={`chat-bubble-card chat-bubble-card--compact chat-bubble-card--${msg.role}`}>
                  <div className="markdown-body">
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.text}</ReactMarkdown>
                  </div>
                </div>
              </div>
            ))}

            {isLoading && (
              <div className="chat-row chat-row--compact chat-row--bot">
                <div className="chat-bubble-card chat-bubble-card--compact chat-bubble-card--bot">
                  <div className="typing-dots" aria-label="Thinking">
                    <span />
                    <span />
                    <span />
                  </div>
                </div>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>

          <div className="assistant-panel__suggestions">
            {suggestions.map((s) => (
              <button
                key={s}
                type="button"
                className="suggestion-pill suggestion-pill--compact"
                onClick={() => handleSuggestionClick(s)}
              >
                {s}
              </button>
            ))}
          </div>

          <div className="assistant-panel__foot">
            <div className="composer-bar composer-bar--compact">
              <input
                ref={inputRef}
                type="text"
                className="composer-bar__input"
                placeholder={`Ask about ${stockName || 'stocks'}...`}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyPress}
                disabled={isLoading}
              />
              <button
                type="button"
                className="composer-bar__send composer-bar__send--sm"
                onClick={() => handleSend(input)}
                disabled={isLoading || !input.trim()}
                aria-label="Send message"
              >
                <Icon name="arrow_upward" size={18} />
              </button>
            </div>
          </div>
        </section>
      )}

      {!isFullScreen && (
        <button
          type="button"
          className={`assistant-fab${isOpen ? ' is-open' : ''}`}
          onClick={toggleChat}
          aria-label={isOpen ? 'Close TrendAI' : 'Open TrendAI assistant'}
          title={isOpen ? 'Close TrendAI' : 'Ask TrendAI'}
        >
          <Icon name={isOpen ? 'close' : 'auto_awesome'} size={22} />
          {!isOpen && <span className="assistant-fab__label">TrendAI</span>}
        </button>
      )}
    </div>
  );
};

export default AiMarketChatBot;
