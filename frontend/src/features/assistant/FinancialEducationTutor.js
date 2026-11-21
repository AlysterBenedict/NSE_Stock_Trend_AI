import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Badge, Card, Icon, PageHeader } from '../../components/common/ui';
import { getGeneralFinancialKnowledge } from '../../services/apiClient';
import { useChatSession } from './useChatSession';

const SUGGESTIONS = [
  'What is a P/E Ratio?',
  'Explain Inflation briefly',
  'How to start investing?',
  'Difference between Stocks and Bonds',
  'What is Risk Management?',
];

const FinancialEducationTutor = () => {
  const {
    messages,
    input,
    setInput,
    isLoading,
    messagesEndRef,
    handleSend,
    handleKeyPress,
  } = useChatSession(
    "Hello! I'm FinTeach. I can help you understand financial concepts, investment strategies, and market terminology.\n\nWhat would you like to learn today?",
    async (question, signal) => {
      const data = await getGeneralFinancialKnowledge(question, signal);
      return data.answer;
    },
    'I encountered a connection error. Please check if the server is running.'
  );

  return (
    <div className="page-stack">
      <PageHeader
        eyebrow="Interactive Curriculum"
        icon="school"
        title="FinTeach Financial Tutor"
        description="Ask questions on valuation ratios, portfolio construction, macroeconomics, and risk principles. Your session stays preserved as you switch tabs."
        actions={
          <Badge tone="primary" icon="verified">
            Session Persistent
          </Badge>
        }
      />

      <Card className="tutor-shell">
        <div className="tutor-messages">
          {messages.map((msg, index) => (
            <div key={index} className={`chat-row chat-row--${msg.role}`}>
              <span className={`chat-avatar chat-avatar--${msg.role}`}>
                <Icon name={msg.role === 'user' ? 'person' : 'school'} size={18} />
              </span>
              <div className={`chat-bubble-card chat-bubble-card--${msg.role}`}>
                <div className="markdown-body">
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.text}</ReactMarkdown>
                </div>
              </div>
            </div>
          ))}

          {isLoading && (
            <div className="chat-row chat-row--bot">
              <span className="chat-avatar chat-avatar--bot">
                <Icon name="school" size={18} />
              </span>
              <div className="chat-bubble-card chat-bubble-card--bot">
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

        <div className="tutor-composer">
          <div className="suggestion-strip" role="group" aria-label="Suggested topics">
            {SUGGESTIONS.map((s) => (
              <button
                key={s}
                type="button"
                className="suggestion-pill"
                onClick={() => handleSend(s)}
                disabled={isLoading}
              >
                <Icon name="lightbulb" size={15} />
                <span>{s}</span>
              </button>
            ))}
          </div>

          <div className="composer-bar">
            <input
              type="text"
              className="composer-bar__input"
              placeholder="Ask a financial question..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyPress}
              disabled={isLoading}
            />
            <button
              type="button"
              className="composer-bar__send"
              onClick={() => handleSend(input)}
              disabled={isLoading || !input.trim()}
              aria-label="Send message"
            >
              <Icon name="arrow_upward" size={20} />
            </button>
          </div>
        </div>
      </Card>
    </div>
  );
};

export default FinancialEducationTutor;
