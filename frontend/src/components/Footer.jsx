import { Link } from 'react-router-dom';

export default function Footer() {
  return (
    <footer className="border-t border-terminal bg-surface mt-auto">
      <div className="max-w-7xl mx-auto px-6 py-8">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-8 mb-8">
          {/* Brand */}
          <div>
            <div className="font-display font-700 text-sm tracking-widest text-white mb-3">
              FIN<span style={{ color: 'var(--accent)' }}>SIGHT</span> <span className="mono font-400 text-dim">AI</span>
            </div>
            <p className="mono text-[11px] text-dim leading-relaxed">
              Institutional-grade AI financial research. Not investment advice.
            </p>
          </div>

          {/* Platform */}
          <div>
            <div className="mono text-[10px] tracking-widest text-muted uppercase mb-3">Platform</div>
            <ul className="space-y-2">
              {['Terminal', 'Markets', 'Research', 'API Docs'].map(item => (
                <li key={item}>
                  <Link to={`/${item.toLowerCase().replace(' ', '-')}`}
                        className="mono text-[11px] text-dim hover:text-accent transition-colors">
                    {item}
                  </Link>
                </li>
              ))}
            </ul>
          </div>

          {/* Stack */}
          <div>
            <div className="mono text-[10px] tracking-widest text-muted uppercase mb-3">Stack</div>
            <ul className="space-y-2">
              {['React + Vite', 'FastAPI', 'LangGraph', 'Grok API (xAI)', 'PostgreSQL'].map(item => (
                <li key={item}>
                  <span className="mono text-[11px] text-dim">{item}</span>
                </li>
              ))}
            </ul>
          </div>

          {/* Status */}
          <div>
            <div className="mono text-[10px] tracking-widest text-muted uppercase mb-3">System</div>
            <div className="space-y-2">
              {[
                { label: 'API', status: 'online' },
                { label: 'LLM', status: 'online' },
                { label: 'Market Data', status: 'pending' },
                { label: 'RAG', status: 'pending' },
              ].map(({ label, status }) => (
                <div key={label} className="flex items-center justify-between">
                  <span className="mono text-[11px] text-dim">{label}</span>
                  <span className="mono text-[10px]" style={{
                    color: status === 'online' ? 'var(--green)' : 'var(--amber)'
                  }}>
                    {status === 'online' ? '● LIVE' : '○ SOON'}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="border-t border-terminal pt-4 flex flex-col md:flex-row items-center justify-between gap-2">
          <span className="mono text-[10px] text-muted">
            © 2025 FinSight AI — FOR RESEARCH PURPOSES ONLY. NOT FINANCIAL ADVICE.
          </span>
          <span className="mono text-[10px] text-muted">
            {new Date().toISOString().split('T')[0]} {new Date().toTimeString().slice(0, 8)} UTC
          </span>
        </div>
      </div>
    </footer>
  );
}
