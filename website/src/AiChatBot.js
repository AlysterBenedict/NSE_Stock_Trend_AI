import React, { useState, useEffect, useRef } from 'react';
import './App.css';

const AiChatBot = ({ onGetInsights, stockName }) => {
    const [isOpen, setIsOpen] = useState(false);
    const [messages, setMessages] = useState([
        { role: 'bot', text: "Hello! I'm TrendAI, your professional stock advisor. Ask me anything about the selected stock." }
    ]);
    const [input, setInput] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const messagesEndRef = useRef(null);

    const suggestions = [
        "Analyze the overall trend",
        "What are the key support levels?",
        "Is the stock volatile?",
        "Short-term outlook?"
    ];

    const toggleChat = () => setIsOpen(!isOpen);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };

    useEffect(() => {
        if (isOpen) scrollToBottom();
    }, [messages, isOpen]);

    const handleSend = async (text) => {
        if (!text.trim() || isLoading) return;

        const userMessage = { role: 'user', text };
        setMessages(prev => [...prev, userMessage]);
        setInput('');
        setIsLoading(true);

        try {
            // Call the parent function which calls the API
            // We expect onGetInsights to return the response string
            const response = await onGetInsights(text);

            const botMessage = { role: 'bot', text: response };
            setMessages(prev => [...prev, botMessage]);
        } catch (error) {
            const errorMessage = { role: 'bot', text: "⚠️ I encountered an error analyzing the data. Please try again." };
            setMessages(prev => [...prev, errorMessage]);
        } finally {
            setIsLoading(false);
        }
    };

    const handleKeyPress = (e) => {
        if (e.key === 'Enter') handleSend(input);
    };

    return (
        <div className="ai-chatbot-container">
            {/* Chat Window */}
            {isOpen && (
                <div className="chat-window">
                    <div className="chat-header">
                        <div className="chat-title">
                            <img src="/chatbot_icon.png" alt="Bot" style={{ width: '24px', height: '24px', borderRadius: '50%' }} /> TrendAI
                        </div>
                        <button className="close-chat" onClick={toggleChat}>×</button>
                    </div>

                    <div className="chat-body">
                        {messages.map((msg, index) => (
                            <div key={index} className={`chat-message ${msg.role}`}>
                                <p>{msg.text}</p>
                            </div>
                        ))}

                        {isLoading && (
                            <div className="chat-message bot loading">
                                <div className="typing-indicator">
                                    <span></span><span></span><span></span>
                                </div>
                            </div>
                        )}

                        <div ref={messagesEndRef} />
                    </div>

                    {/* Suggestions */}
                    <div className="chat-suggestions">
                        {suggestions.map((s, i) => (
                            <button key={i} className="suggestion-chip" onClick={() => handleSend(s)}>
                                {s}
                            </button>
                        ))}
                    </div>

                    <div className="chat-footer">
                        <div className="input-group">
                            <input
                                type="text"
                                placeholder={`Ask about ${stockName}...`}
                                value={input}
                                onChange={(e) => setInput(e.target.value)}
                                onKeyPress={handleKeyPress}
                                disabled={isLoading}
                            />
                            <button
                                className="send-btn"
                                onClick={() => handleSend(input)}
                                disabled={isLoading || !input.trim()}
                            >
                                ➤
                            </button>
                        </div>
                    </div>
                </div>
            )}

            {/* Chat Bubble Button */}
            <button className="chat-bubble" onClick={toggleChat}>
                {isOpen ? '✕' : <img src="/chatbot_icon.png" alt="Chat" style={{ width: '100%', height: '100%', objectFit: 'cover', borderRadius: '50%' }} />}
            </button>
        </div>
    );
};

export default AiChatBot;
