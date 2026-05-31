import { useRef, useEffect, useState } from 'react';
import { useChat } from '../hooks/useChat';
import ChatMessage from './ChatMessage';

const QUICK_PROMPTS = [
  'Analyze AAPL fundamentals',
  'Compare MSFT vs GOOGL',
  'Summarize latest Fed decision',
  'What are key risks in NVDA?',
  'Explain P/E ratio in simple terms',
  'Show me sectors outperforming YTD',
];

export default function AITerminal({ compact = false }) {
  const { messages, isLoading, sendMessage, clearConversation } = useChat();
  const [input, setInput] = useState('');
  const [history, setHistory] = useState([]);
  const [historyIdx, setHistoryIdx] = useState(-1);
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  // Auto-scroll to bottom on new messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  const handleSend = async () => {
    const trimmed = input.trim();
    if (!trimmed || isLoading) return;
    setHistory(prev => [trimmed, ...prev.slice(0, 49)]);
    setHistoryIdx(-1);
    setInput('');
    await sendMessage(trimmed);
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
    // Arrow up/down for history
    if (e.key === 'ArrowUp') {
      e.preventDefault();
      const next = Math.min(historyIdx + 1, history.length - 1);
      setHistoryIdx(next);
      setInput(history[next] || '');
    }
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      const next = Math.max(historyIdx - 1, -1);
      setHistoryIdx(next);
      setInput(next === -1 ? '' : history[next]);
    }
  };

  const chatHeight = compact ? 'h-72' : 'h-[480px]';

  return (
    <div className="panel flex flex-col">
      {/* Panel header */}
      <div className="panel-header justify-between">
        <div className="flex items-center gap-2">
          <span className="status-dot" />
          <span>FINSIGHT AI TERMINAL</span>
          <span className="px-1.5 py-0.5 text-[9px] border border-terminal text-muted">
            GROK-2 / LANGCHAIN
          </span>
        </div>
        <div className="flex items-center gap-3">
          <span className="text-muted text-[10px]">
            {messages.length - 1} msg{messages.length !== 2 ? 's' : ''}
          </span>
          <button
            onClick={clearConversation}
            className="mono text-[10px] text-muted hover:text-red-t transition-colors"
            title="Clear conversation"
          >
            CLR
          </button>
        </div>
      </div>

      {/* Messages area */}
      <div className={`${chatHeight} overflow-y-auto flex flex-col gap-px bg-terminal-bg`}>
        {messages.map((msg, i) => (
          <ChatMessage
            key={msg.id}
            message={msg}
            isLatest={i === messages.length - 1}
          />
        ))}

        {/* Loading state */}
        {isLoading && (
          <div className="msg-ai p-4 animate-fade-in">
            <div className="flex items-center gap-2 mb-2">
              <span className="mono text-[10px] font-600 tracking-widest" style={{ color: 'var(--green)' }}>
                FINSIGHT AI
              </span>
            </div>
            <div className="flex items-center gap-2">
              <div className="flex gap-1">
                {[0, 1, 2].map((i) => (
                  <span
                    key={i}
                    className="w-1.5 h-1.5 rounded-full animate-bounce"
                    style={{
                      background: 'var(--green)',
                      animationDelay: `${i * 0.15}s`,
                    }}
                  />
                ))}
              </div>
              <span className="mono text-[11px] text-dim">Analyzing query...</span>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Quick prompts */}
      {!compact && (
        <div className="px-3 py-2 border-t border-terminal flex gap-2 overflow-x-auto">
          {QUICK_PROMPTS.map((p) => (
            <button
              key={p}
              onClick={() => { setInput(p); inputRef.current?.focus(); }}
              className="mono text-[10px] px-2.5 py-1 border border-terminal text-muted hover:text-dim hover:border-muted transition-all whitespace-nowrap flex-shrink-0"
            >
              {p}
            </button>
          ))}
        </div>
      )}

      {/* Input area */}
      <div className="border-t border-terminal p-3">
        <div className="flex items-center gap-2 bg-surface border border-terminal px-3 py-2 focus-within:border-accent/40 transition-colors">
          <span className="mono text-xs text-accent flex-shrink-0">❯</span>
          <input
            ref={inputRef}
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask about stocks, markets, filings..."
            disabled={isLoading}
            className="flex-1 bg-transparent outline-none mono text-sm text-terminal placeholder-muted disabled:opacity-50"
          />
          {input && (
            <button
              onClick={() => setInput('')}
              className="mono text-muted hover:text-dim text-xs flex-shrink-0"
            >
              ✕
            </button>
          )}
          <button
            onClick={handleSend}
            disabled={!input.trim() || isLoading}
            className="mono text-[11px] px-3 py-1 flex-shrink-0 transition-all disabled:opacity-30"
            style={{
              color: 'var(--accent)',
              border: '1px solid rgba(0,212,255,0.3)',
              background: 'rgba(0,212,255,0.05)',
            }}
          >
            {isLoading ? '...' : 'SEND'}
          </button>
        </div>
        <div className="flex items-center gap-4 mt-1.5 px-1">
          <span className="mono text-[9px] text-muted">↵ ENTER to send</span>
          <span className="mono text-[9px] text-muted">↑↓ History</span>
          <span className="mono text-[9px] text-muted">SHIFT+↵ Newline</span>
        </div>
      </div>
    </div>
  );
}
