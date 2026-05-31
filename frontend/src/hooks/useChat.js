import { useState, useCallback, useRef } from 'react';
import { sendMessage } from '../lib/api';

/**
 * useChat — manages conversation state, loading, errors,
 * and session continuity for the AI terminal.
 */
export function useChat() {
  const [messages, setMessages] = useState([
    {
      id: 'init',
      role: 'assistant',
      content: 'FinSight AI initialized. I can analyze stocks, parse financial filings, generate research reports, and answer market questions. What would you like to explore?',
      timestamp: new Date(),
    }
  ]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const sessionIdRef = useRef(null);

  const sendChatMessage = useCallback(async (content) => {
    if (!content.trim() || isLoading) return;

    const userMsg = {
      id: `user-${Date.now()}`,
      role: 'user',
      content: content.trim(),
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMsg]);
    setIsLoading(true);
    setError(null);

    try {
      const data = await sendMessage(content, {
        sessionId: sessionIdRef.current,
      });

      // Persist session_id for conversation continuity
      if (data.session_id) {
        sessionIdRef.current = data.session_id;
      }

      const assistantMsg = {
        id: `ai-${Date.now()}`,
        role: 'assistant',
        content: data.response || data.message || 'No response received.',
        timestamp: new Date(),
        sources: data.sources || [],
        metadata: data.metadata || {},
      };

      setMessages(prev => [...prev, assistantMsg]);
    } catch (err) {
      const errMsg = {
        id: `err-${Date.now()}`,
        role: 'error',
        content: `Connection error: ${err.message}. Ensure the backend is running at ${import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000'}.`,
        timestamp: new Date(),
      };
      setMessages(prev => [...prev, errMsg]);
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  }, [isLoading]);

  const clearConversation = useCallback(() => {
    sessionIdRef.current = null;
    setMessages([{
      id: 'init-' + Date.now(),
      role: 'assistant',
      content: 'Session cleared. Ready for new analysis.',
      timestamp: new Date(),
    }]);
    setError(null);
  }, []);

  return {
    messages,
    isLoading,
    error,
    sessionId: sessionIdRef.current,
    sendMessage: sendChatMessage,
    clearConversation,
  };
}
