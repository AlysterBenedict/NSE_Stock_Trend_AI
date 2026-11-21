import { useEffect, useRef, useState } from 'react';
import { isCancelledError } from '../../utils/formatters';

export function useChatSession(initialGreeting, sendMessageFn, errorFallbackText, autoScroll = true) {
  const [messages, setMessages] = useState([{ role: 'bot', text: initialGreeting }]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);
  const abortControllerRef = useRef(null);
  const isMountedRef = useRef(true);

  useEffect(() => {
    isMountedRef.current = true;
    return () => {
      isMountedRef.current = false;
      abortControllerRef.current?.abort();
    };
  }, []);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    if (autoScroll) {
      scrollToBottom();
    }
  }, [messages, autoScroll]);

  const handleSend = async (text) => {
    if (!text.trim() || isLoading) return;

    abortControllerRef.current?.abort();
    const controller = new AbortController();
    abortControllerRef.current = controller;

    setMessages((prev) => [...prev, { role: 'user', text }]);
    setInput('');
    setIsLoading(true);

    try {
      const reply = await sendMessageFn(text, controller.signal);
      if (!isMountedRef.current || controller.signal.aborted) return;
      setMessages((prev) => [...prev, { role: 'bot', text: reply }]);
    } catch (err) {
      if (!isMountedRef.current || controller.signal.aborted || isCancelledError(err)) {
        return;
      }
      setMessages((prev) => [...prev, { role: 'bot', text: errorFallbackText }]);
    } finally {
      if (isMountedRef.current && abortControllerRef.current === controller) {
        setIsLoading(false);
      }
    }
  };

  const handleKeyPress = (event) => {
    if (event.key === 'Enter') handleSend(input);
  };

  return {
    messages,
    input,
    setInput,
    isLoading,
    messagesEndRef,
    handleSend,
    handleKeyPress,
  };
}
