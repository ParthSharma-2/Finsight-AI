import { useState } from 'react';
import { Copy, Check, Sparkles, User, AlertCircle } from 'lucide-react';

function formatTime(date) {
  return new Date(date).toLocaleTimeString('en-US', {
    hour12: false,
    hour: '2-digit',
    minute: '2-digit',
  });
}

export default function ChatMessage({ message }) {
  const [copied, setCopied] = useState(false);
  const { role, content, timestamp } = message;

  const isUser = role === 'user';
  const isError = role === 'error';

  const copyToClipboard = () => {
    navigator.clipboard.writeText(content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className={`p-4 border-b border-zinc-100 dark:border-zinc-800/80 transition-colors ${
      isUser ? 'bg-zinc-50/40 dark:bg-zinc-900/30' : 'bg-white dark:bg-zinc-950'
    }`}>
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          {isUser ? (
            <div className="w-5 h-5 rounded-md bg-zinc-200 dark:bg-zinc-800 flex items-center justify-center text-zinc-700 dark:text-zinc-300">
              <User className="w-3 h-3" />
            </div>
          ) : isError ? (
            <div className="w-5 h-5 rounded-md bg-rose-100 dark:bg-rose-950/60 flex items-center justify-center text-rose-600 dark:text-rose-400">
              <AlertCircle className="w-3 h-3" />
            </div>
          ) : (
            <div className="w-5 h-5 rounded-md bg-indigo-100 dark:bg-indigo-950/60 flex items-center justify-center text-indigo-600 dark:text-indigo-400">
              <Sparkles className="w-3 h-3" />
            </div>
          )}

          <span className="text-xs font-semibold text-zinc-900 dark:text-zinc-100 font-mono">
            {isUser ? 'YOU' : isError ? 'ERROR' : 'FINSIGHT AI'}
          </span>

          {!isUser && !isError && (
            <span className="text-[10px] font-mono text-zinc-400 bg-zinc-100 dark:bg-zinc-900 px-1.5 py-0.5 rounded border border-zinc-200/60 dark:border-zinc-800/60">
              Groq • GPT-OSS 120B
            </span>
          )}
        </div>

        <div className="flex items-center gap-2">
          <span className="text-[11px] font-mono text-zinc-400">
            {formatTime(timestamp)}
          </span>
          <button
            onClick={copyToClipboard}
            className="p-1 text-zinc-400 hover:text-zinc-600 dark:hover:text-zinc-200 transition-colors rounded"
            title="Copy response"
          >
            {copied ? <Check className="w-3.5 h-3.5 text-emerald-500" /> : <Copy className="w-3.5 h-3.5" />}
          </button>
        </div>
      </div>

      <div className={`text-sm leading-relaxed whitespace-pre-wrap pl-7 ${
        isError ? 'text-rose-600 dark:text-rose-400 font-mono text-xs' : 'text-zinc-800 dark:text-zinc-200'
      }`}>
        {content}
      </div>
    </div>
  );
}
