export default function MetricCard({ label, value, change, changePct, sub, accent = false }) {
  const isPositive = change > 0 || (typeof change === 'string' && change.startsWith('+'));
  const isNegative = change < 0 || (typeof change === 'string' && change.startsWith('-'));

  return (
    <div className="fin-card p-4 sm:p-5 flex flex-col justify-between">
      <div className="text-[11px] font-mono font-medium text-zinc-500 dark:text-zinc-400 tracking-wider uppercase mb-2">
        {label}
      </div>

      <div className={`text-2xl font-bold font-mono tracking-tight ${
        accent ? 'text-indigo-600 dark:text-indigo-400' : 'text-zinc-900 dark:text-zinc-100'
      }`}>
        {value}
      </div>

      {(change !== undefined || changePct !== undefined || sub) && (
        <div className="flex items-center gap-2 mt-2 pt-2 border-t border-zinc-100 dark:border-zinc-800/80 text-xs font-mono">
          {(change !== undefined || changePct !== undefined) && (
            <span
              className={`font-semibold flex items-center gap-0.5 ${
                isPositive
                  ? 'text-emerald-600 dark:text-emerald-400'
                  : isNegative
                  ? 'text-rose-600 dark:text-rose-400'
                  : 'text-zinc-500'
              }`}
            >
              {isPositive ? '▲' : isNegative ? '▼' : '•'} {change} {changePct && `(${changePct})`}
            </span>
          )}
          {sub && <span className="text-zinc-400 dark:text-zinc-500 text-[11px] ml-auto">{sub}</span>}
        </div>
      )}
    </div>
  );
}
