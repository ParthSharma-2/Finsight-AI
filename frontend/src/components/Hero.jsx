import { Link } from 'react-router-dom';

const FEATURES = [
  { icon: '◈', label: 'AI Stock Analysis', desc: 'Deep fundamental & technical analysis powered by LLMs' },
  { icon: '◉', label: 'RAG on Filings', desc: 'Query 10-K, 10-Q, and earnings transcripts instantly' },
  { icon: '◐', label: 'Market Intelligence', desc: 'Real-time data from Yahoo Finance, Alpha Vantage & more' },
  { icon: '◑', label: 'Multi-Agent Research', desc: 'LangGraph agents that reason across tools and memory' },
];

const STATS = [
  { value: '10K+', label: 'SEC Filings Indexed' },
  { value: '<200ms', label: 'API Latency' },
  { value: '50+', label: 'Financial Metrics' },
  { value: '24/7', label: 'Market Coverage' },
];

export default function Hero() {
  return (
    <section className="relative min-h-screen flex flex-col items-center justify-center px-4 pt-20 pb-10 overflow-hidden grid-bg">
      {/* Ambient glow blobs */}
      <div className="absolute top-1/4 left-1/4 w-96 h-96 rounded-full pointer-events-none blur-3xl"
           style={{ background: 'radial-gradient(circle, rgba(0,212,255,0.06) 0%, transparent 70%)' }} />
      <div className="absolute bottom-1/3 right-1/4 w-80 h-80 rounded-full pointer-events-none blur-3xl"
           style={{ background: 'radial-gradient(circle, rgba(0,255,136,0.04) 0%, transparent 70%)' }} />

      {/* System badge */}
      <div className="flex items-center gap-2 mb-8 px-4 py-2 border border-terminal mono text-[10px] text-dim tracking-widest">
        <span className="status-dot" />
        SYS:ONLINE — FINSIGHT AI v0.1.0 — INSTITUTIONAL RESEARCH PLATFORM
      </div>

      {/* Main headline */}
      <div className="text-center max-w-4xl mb-6">
        <h1 className="font-display font-800 leading-none mb-4" style={{ fontSize: 'clamp(2.5rem, 7vw, 5.5rem)' }}>
          <span className="block text-white">INSTITUTIONAL</span>
          <span className="block animate-glow" style={{ color: 'var(--accent)' }}>FINANCIAL INTELLIGENCE</span>
          <span className="block text-terminal" style={{ fontSize: '55%', fontWeight: 400, letterSpacing: '0.15em' }}>
            POWERED BY AI
          </span>
        </h1>

        <p className="font-body text-base md:text-lg text-dim max-w-2xl mx-auto leading-relaxed">
          Institutional-grade research platform combining LangGraph multi-agent AI with real-time
          market data. Analyze stocks, query filings, and generate reports in seconds.
        </p>
      </div>

      {/* CTA buttons */}
      <div className="flex flex-wrap items-center justify-center gap-4 mb-16">
        <Link to="/terminal">
          <button className="btn-primary flex items-center gap-2 text-sm">
            <span>▶</span> LAUNCH TERMINAL
          </button>
        </Link>
        <Link to="/markets">
          <button className="btn-secondary flex items-center gap-2 text-sm">
            <span>◈</span> VIEW MARKETS
          </button>
        </Link>
      </div>

      {/* Stats bar */}
      <div className="w-full max-w-3xl grid grid-cols-2 md:grid-cols-4 border border-terminal mb-16">
        {STATS.map((stat, i) => (
          <div key={i} className={`p-5 text-center ${i < 3 ? 'border-r border-terminal' : ''}`}>
            <div className="font-display font-700 text-xl md:text-2xl mb-1"
                 style={{ color: 'var(--accent)' }}>
              {stat.value}
            </div>
            <div className="mono text-[10px] tracking-widest text-dim uppercase">{stat.label}</div>
          </div>
        ))}
      </div>

      {/* Feature grid */}
      <div className="w-full max-w-4xl grid grid-cols-1 sm:grid-cols-2 gap-px bg-border">
        {FEATURES.map((f, i) => (
          <div key={i} className="bg-panel p-6 hover:bg-surface transition-colors duration-200 group cursor-default">
            <div className="flex items-start gap-4">
              <span className="text-2xl mt-0.5 transition-all duration-300 group-hover:scale-110"
                    style={{ color: 'var(--accent)' }}>
                {f.icon}
              </span>
              <div>
                <div className="font-display font-600 text-sm text-white mb-1 tracking-wide">{f.label}</div>
                <div className="font-body text-sm text-dim leading-relaxed">{f.desc}</div>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Scroll indicator */}
      <div className="absolute bottom-6 left-1/2 -translate-x-1/2 flex flex-col items-center gap-1 animate-bounce">
        <div className="w-px h-8 bg-gradient-to-b from-border to-transparent" />
        <span className="mono text-[9px] text-muted tracking-widest">SCROLL</span>
      </div>
    </section>
  );
}
