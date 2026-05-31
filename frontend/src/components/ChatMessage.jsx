import { useState } from 'react';

function formatTime(date) {
  return new Date(date).toLocaleTimeString('en-US', {
    hour12: false,
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  });
}

function TypedText({ text }) {
  return (
    <div className="mono text-[13px] leading-relaxed whitespace-pre-wrap"
         style={{ color: 'var(--text)' }}>
      {text}
    </div>
  );
}

export default function ChatMessage({ message, isLatest }) {
  const [copied, setCopied] = useState(false);
  const { role, content, timestamp, sources } = message;

  const isUser = role === 'user';
  const isError = role === 'error';

  const copyToClipboard = () => {
    navigator.clipboard.writeText(content);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  };

  const containerClass = isUser ? 'msg-user' : isError ? 'msg-error' : 'msg-ai';
  const roleLabel = isUser ? 'YOU' : isError ? 'ERROR' : 'FINSIGHT AI';
  const roleLabelColor = isUser ? 'var(--accent)' : isError ? 'var(--red)' : 'var(--green)';

  return (
    <div className={`relative animate-slide-up p-4 group ${containerClass}`}>
      {/* Header row */}
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <span className="mono text-[10px] font-600 tracking-widest"
                style={{ color: roleLabelColor }}>
            {roleLabel}
          </span>
          {!isUser && !isError && (
            <span className="mono text-[9px] px-1.5 py-0.5 border"
                  style={{ borderColor: 'var(--border)', color: 'var(--dim)' }}>
              GPT-4o / Grok
            </span>
          )}
        </div>
        <div className="flex items-center gap-2">
          <span className="mono text-[10px] text-muted tabular-nums">
            {formatTime(timestamp)}
          </span>
          <button
            onClick={copyToClipboard}
            className="opacity-0 group-hover:opacity-100 transition-opacity mono text-[10px] text-muted hover:text-dim px-1"
          >
            {copied ? '✓' : '⌗'}
          </button>
        </div>
      </div>

      {/* Content */}
      <TypedText text={content} />

      {/* Sources */}
      {sources && sources.length > 0 && (
        <div className="mt-3 pt-3 border-t border-terminal">
          <div className="mono text-[9px] text-muted tracking-widest mb-1.5">SOURCES</div>
          <div className="flex flex-wrap gap-2">
            {sources.map((src, i) => (
              <span key={i} className="mono text-[10px] px-2 py-0.5 border border-terminal text-dim">
                {src}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Active cursor for latest AI message */}
      {isLatest && !isUser && !isError && (
        <span className="inline-block w-2 h-3.5 ml-1 animate-blink"
              style={{ background: 'var(--green)', opacity: 0.8 }} />
      )}
    </div>
  );
}
