import Hero from '../components/Hero';
import AITerminal from '../components/AITerminal';
import ChartPanel from '../components/ChartPanel';
import MetricCard from '../components/MetricCard';

const MARKET_METRICS = [
  { label: 'S&P 500', value: '5,277', change: '+14', changePct: '+0.27%', sub: 'SPY' },
  { label: 'NASDAQ', value: '16,428', change: '+72', changePct: '+0.44%', sub: 'QQQ' },
  { label: 'VIX', value: '13.82', change: '-0.45', changePct: '-3.15%', sub: 'Volatility Index' },
  { label: 'USD/JPY', value: '155.34', change: '+0.21', changePct: '+0.14%', sub: 'Forex' },
];

export default function Home() {
  return (
    <div className="flex flex-col min-h-screen">
      {/* Hero section */}
      <Hero />

      {/* Live AI section */}
      <section className="px-4 md:px-6 py-12 max-w-7xl mx-auto w-full">
        <SectionHeader
          label="AI RESEARCH TERMINAL"
          desc="Ask anything about markets, stocks, or financial concepts"
          tag="LIVE"
        />
        <AITerminal />
      </section>

      {/* Market Overview section */}
      <section className="px-4 md:px-6 py-12 max-w-7xl mx-auto w-full">
        <SectionHeader
          label="MARKET OVERVIEW"
          desc="Key indices and indicators at a glance"
        />
        <div className="grid grid-cols-2 md:grid-cols-4 gap-px bg-border mb-6">
          {MARKET_METRICS.map((m, i) => (
            <MetricCard key={i} {...m} accent={i === 0} />
          ))}
        </div>
        <ChartPanel />
      </section>

      {/* Architecture section */}
      <section className="px-4 md:px-6 py-12 max-w-7xl mx-auto w-full">
        <SectionHeader
          label="PLATFORM ARCHITECTURE"
          desc="How FinSight AI processes your queries"
        />
        <ArchitectureDiagram />
      </section>
    </div>
  );
}

function SectionHeader({ label, desc, tag }) {
  return (
    <div className="flex items-start justify-between mb-6">
      <div>
        <div className="flex items-center gap-3 mb-1">
          <span className="w-4 h-px" style={{ background: 'var(--accent)' }} />
          <span className="mono text-[10px] tracking-widest text-accent uppercase">{label}</span>
          {tag && (
            <span className="mono text-[9px] px-2 py-0.5 border animate-pulse-slow"
                  style={{ borderColor: 'var(--green)', color: 'var(--green)' }}>
              {tag}
            </span>
          )}
        </div>
        {desc && <p className="font-body text-sm text-dim ml-7">{desc}</p>}
      </div>
    </div>
  );
}

function ArchitectureDiagram() {
  const nodes = [
    { id: 'ui', label: 'React Frontend', sub: 'Vite + Tailwind', color: 'var(--accent)', x: 'left' },
    { id: 'api', label: 'FastAPI Backend', sub: 'REST + CORS', color: 'var(--amber)', x: 'center' },
    { id: 'llm', label: 'Grok API (xAI)', sub: 'LangChain/LangGraph', color: 'var(--green)', x: 'right' },
  ];

  return (
    <div className="panel p-6">
      <div className="flex items-center justify-between gap-4 relative">
        {nodes.map((node, i) => (
          <div key={node.id} className="flex-1 flex flex-col items-center">
            {/* Node box */}
            <div className="w-full border p-4 text-center"
                 style={{ borderColor: node.color + '40', background: node.color + '08' }}>
              <div className="mono text-xs font-600 mb-1" style={{ color: node.color }}>
                {node.label}
              </div>
              <div className="mono text-[10px] text-dim">{node.sub}</div>
            </div>

            {/* Arrow (except last) */}
            {i < nodes.length - 1 && (
              <div className="absolute top-1/2 -translate-y-1/2"
                   style={{ left: `${((i + 1) / nodes.length) * 100 - 100 / (nodes.length * 2)}%` }}>
                <span className="mono text-dim text-xl">→</span>
              </div>
            )}
          </div>
        ))}
      </div>

      {/* Data flow layers */}
      <div className="mt-6 grid grid-cols-3 gap-3 text-center">
        {[
          { label: 'RAG Layer', items: ['SEC Edgar', 'Annual Reports', 'Earnings Calls'] },
          { label: 'Memory Layer', items: ['Conversation History', 'Session Context', 'DynamoDB'] },
          { label: 'Data Layer', items: ['Yahoo Finance', 'Alpha Vantage', 'Polygon.io'] },
        ].map(({ label, items }) => (
          <div key={label} className="border border-terminal p-3">
            <div className="mono text-[10px] tracking-widest text-muted mb-2 uppercase">{label}</div>
            {items.map(item => (
              <div key={item} className="mono text-[10px] text-dim py-0.5">{item}</div>
            ))}
          </div>
        ))}
      </div>
    </div>
  );
}
