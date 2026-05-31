export default function MetricCard({ label, value, change, changePct, sub, accent = false, loading = false }) {
  const isPositive = change > 0 || (typeof change === 'string' && change.startsWith('+'));
  const isNegative = change < 0 || (typeof change === 'string' && change.startsWith('-'));

  const changeColor = isPositive ? 'var(--green)' : isNegative ? 'var(--red)' : 'var(--dim)';
  const changePrefix = isPositive ? '▲' : isNegative ? '▼' : '—';

  if (loading) {
    return (
      <div className="metric-card">
        <div className="skeleton h-3 w-20 mb-3 rounded" />
        <div className="skeleton h-7 w-28 mb-2 rounded" />
        <div className="skeleton h-3 w-16 rounded" />
      </div>
    );
  }

  return (
    <div className="metric-card group cursor-default">
      {/* Label */}
      <div className="mono text-[10px] tracking-widest text-dim uppercase mb-2">
        {label}
      </div>

      {/* Value */}
      <div
        className="font-display text-xl font-600 mb-1.5 transition-all duration-200"
        style={{ color: accent ? 'var(--accent)' : 'var(--text)' }}
      >
        {value}
      </div>

      {/* Change */}
      {(change !== undefined || changePct !== undefined) && (
        <div className="flex items-center gap-2">
          <span className="mono text-[11px] font-500" style={{ color: changeColor }}>
            {changePrefix} {change}
          </span>
          {changePct && (
            <span className="mono text-[10px]" style={{ color: changeColor }}>
              ({changePct})
            </span>
          )}
        </div>
      )}

      {/* Sub label */}
      {sub && (
        <div className="mono text-[10px] text-muted mt-1">{sub}</div>
      )}

      {/* Hover accent line */}
      <div
        className="absolute bottom-0 left-0 h-px w-0 group-hover:w-full transition-all duration-300"
        style={{ background: 'var(--accent)', opacity: 0.4 }}
      />
    </div>
  );
}
