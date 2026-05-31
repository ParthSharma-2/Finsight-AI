import { useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { useBackendStatus } from '../hooks/useBackendStatus';

const NAV_ITEMS = [
  { label: 'HOME', path: '/' },
  { label: 'TERMINAL', path: '/terminal' },
  { label: 'MARKETS', path: '/markets' },
  { label: 'RESEARCH', path: '/research' },
];

export default function Nav() {
  const location = useLocation();
  const { status } = useBackendStatus();
  const [mobileOpen, setMobileOpen] = useState(false);

  const statusColor = {
    online: 'var(--green)',
    offline: 'var(--red)',
    checking: 'var(--amber)',
  }[status];

  const statusLabel = {
    online: 'CONNECTED',
    offline: 'OFFLINE',
    checking: 'CONNECTING',
  }[status];

  return (
    <nav className="fixed top-0 left-0 right-0 z-50 border-b border-terminal bg-terminal-bg/90 backdrop-blur-sm">
      {/* Top bar */}
      <div className="flex items-center justify-between px-4 md:px-6 h-12">
        {/* Logo */}
        <Link to="/" className="flex items-center gap-3 group">
          <div className="relative">
            <div className="w-7 h-7 border border-accent/50 flex items-center justify-center"
                 style={{ clipPath: 'polygon(50% 0%, 100% 25%, 100% 75%, 50% 100%, 0% 75%, 0% 25%)' }}>
              <span className="mono text-xs font-bold text-accent">F</span>
            </div>
          </div>
          <div>
            <span className="font-display font-700 text-sm tracking-widest text-white">
              FIN<span className="text-accent">SIGHT</span>
            </span>
            <span className="mono text-[9px] text-dim ml-1">AI</span>
          </div>
        </Link>

        {/* Desktop nav links */}
        <div className="hidden md:flex items-center gap-1">
          {NAV_ITEMS.map((item) => {
            const active = location.pathname === item.path;
            return (
              <Link
                key={item.path}
                to={item.path}
                className="relative mono text-[11px] px-4 py-1.5 tracking-wider transition-all duration-200"
                style={{
                  color: active ? 'var(--accent)' : 'var(--dim)',
                  borderBottom: active ? '1px solid var(--accent)' : '1px solid transparent',
                }}
              >
                {active && (
                  <span className="absolute left-1 top-1/2 -translate-y-1/2 text-accent text-[8px]">▶</span>
                )}
                {item.label}
              </Link>
            );
          })}
        </div>

        {/* Right side */}
        <div className="flex items-center gap-4">
          {/* Backend status */}
          <div className="hidden sm:flex items-center gap-2 mono text-[10px] px-3 py-1 border"
               style={{ borderColor: statusColor + '40', background: statusColor + '08' }}>
            <span className="w-1.5 h-1.5 rounded-full animate-pulse-slow"
                  style={{ background: statusColor, boxShadow: `0 0 6px ${statusColor}` }} />
            <span style={{ color: statusColor }}>{statusLabel}</span>
          </div>

          {/* Time */}
          <LiveClock />

          {/* Mobile hamburger */}
          <button
            className="md:hidden mono text-dim text-xs"
            onClick={() => setMobileOpen(!mobileOpen)}
          >
            {mobileOpen ? '✕' : '☰'}
          </button>
        </div>
      </div>

      {/* Mobile menu */}
      {mobileOpen && (
        <div className="md:hidden border-t border-terminal bg-terminal-bg">
          {NAV_ITEMS.map((item) => (
            <Link
              key={item.path}
              to={item.path}
              onClick={() => setMobileOpen(false)}
              className="block mono text-[11px] px-6 py-3 border-b border-terminal tracking-widest"
              style={{ color: location.pathname === item.path ? 'var(--accent)' : 'var(--dim)' }}
            >
              {location.pathname === item.path ? '▶ ' : '  '}{item.label}
            </Link>
          ))}
        </div>
      )}
    </nav>
  );
}

function LiveClock() {
  const [time, setTime] = useState(new Date());

  // Simple clock without useEffect for SSR safety
  if (typeof window !== 'undefined') {
    setInterval(() => setTime(new Date()), 1000);
  }

  return (
    <span className="mono text-[10px] text-dim hidden md:block tabular-nums">
      {time.toLocaleTimeString('en-US', { hour12: false, timeZone: 'America/New_York' })} ET
    </span>
  );
}
