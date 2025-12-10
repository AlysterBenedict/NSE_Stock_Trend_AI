import React, { useState, useEffect, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import axios from 'axios';
import './App.css';

const FinancialTeacher = () => {
    const [messages, setMessages] = useState([
        { role: 'bot', text: "🎓 Hello! I'm FinTeach. I can help you understand financial concepts, investment strategies, and market terms. \n\nWhat would you like to learn today?" }
    ]);
    const [input, setInput] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const messagesEndRef = useRef(null);

    const suggestions = [
        "What is a P/E Ratio?",
        "Explain Inflation briefly",
        "How to start investing?",
        "Difference between Stocks and Bonds",
        "What is Risk Management?"
    ];

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    const handleSend = async (text) => {
        if (!text.trim() || isLoading) return;

        const userMessage = { role: 'user', text };
        setMessages(prev => [...prev, userMessage]);
        setInput('');
        setIsLoading(true);

        try {
            // Call the dedicated Financial Teacher endpoint
            const response = await axios.post('http://127.0.0.1:5000/get-general-knowledge', {
                user_question: text
            });

            const botMessage = { role: 'bot', text: response.data.answer };
            setMessages(prev => [...prev, botMessage]);
        } catch (error) {
            console.error("FinTeach Error:", error);
            const errorMessage = { role: 'bot', text: "⚠️ I lost my train of thought. Please check if the server is running." };
            setMessages(prev => [...prev, errorMessage]);
        } finally {
            setIsLoading(false);
        }
    };

    const handleSuggestionClick = (text) => {
        handleSend(text);
    };

    const handleKeyPress = (e) => {
        if (e.key === 'Enter') handleSend(input);
    };

    return (
        <div className="financial-teacher-container">
            <div className="teacher-chat-window">
                <div className="teacher-header">
                    <span style={{ fontSize: '24px' }}>🎓</span>
                    <h2>FinTeach</h2>
                </div>

                <div className="teacher-chat-body">
                    {messages.map((msg, index) => (
                        <div key={index} className={`teacher-message ${msg.role}`}>
                            <div className="message-content">
                                <ReactMarkdown remarkPlugins={[remarkGfm]}>
                                    {msg.text}
                                </ReactMarkdown>
                            </div>
                        </div>
                    ))}

                    {isLoading && (
                        <div className="teacher-message bot loading">
                            <div className="typing-indicator">
                                <span></span><span></span><span></span>
                            </div>
                        </div>
                    )}
                    <div ref={messagesEndRef} />
                </div>

                <div className="teacher-suggestions">
                    {suggestions.map((s, i) => (
                        <button key={i} className="teacher-suggestion-chip" onClick={() => handleSuggestionClick(s)}>
                            {s}
                        </button>
                    ))}
                </div>

                <div className="teacher-footer">
                    <div className="input-group">
                        <input
                            type="text"
                            placeholder="Ask a financial question..."
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
        </div>
    );
};

export default FinancialTeacher;
