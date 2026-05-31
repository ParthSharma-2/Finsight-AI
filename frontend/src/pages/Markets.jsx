import ChartPanel from '../components/ChartPanel';
import MetricCard from '../components/MetricCard';

const SECTOR_PERFORMANCE = [
  { sector: 'Technology', chg: '+2.14%', ytd: '+28.4%', up: true },
  { sector: 'Healthcare', chg: '+0.87%', ytd: '+12.1%', up: true },
  { sector: 'Financials', chg: '+1.23%', ytd: '+18.7%', up: true },
  { sector: 'Energy', chg: '-0.45%', ytd: '+5.2%', up: false },
  { sector: 'Consumer Disc.', chg: '-0.12%', ytd: '+9.8%', up: false },
  { sector: 'Industrials', chg: '+0.56%', ytd: '+11.3%', up: true },
  { sector: 'Utilities', chg: '-0.78%', ytd: '-2.1%', up: false },
  { sector: 'Real Estate', chg: '-1.02%', ytd: '-4.7%', up: false },
  { sector: 'Materials', chg: '+0.33%', ytd: '+7.5%', up: true },
  { sector: 'Comm. Services', chg: '+1.78%', ytd: '+22.1%', up: true },
];

const TOP_MOVERS = {
  gainers: [
    { symbol: 'SMCI', price: '884.20', chg: '+12.4%' },
    { symbol: 'ARM', price: '134.55', chg: '+8.7%' },
    { symbol: 'PLTR', price: '28.34', chg: '+6.2%' },
    { symbol: 'MSTR', price: '1,542', chg: '+5.9%' },
    { symbol: 'COIN', price: '248.77', chg: '+5.4%' },
  ],
  losers: [
    { symbol: 'PARA', price: '10.23', chg: '-7.8%' },
    { symbol: 'WBD', price: '8.44', chg: '-5.1%' },
    { symbol: 'MPW', price: '4.15', chg: '-4.7%' },
    { symbol: 'NYCB', price: '3.88', chg: '-4.2%' },
    { symbol: 'VFC', price: '14.32', chg: '-3.8%' },
  ],
};

const MACRO_METRICS = [
  { label: 'Fed Funds Rate', value: '5.25–5.50%', sub: 'Hold — FOMC June 2025', accent: false },
  { label: '10Y Treasury', value: '4.52%', change: '+0.02', changePct: '+0.44%', sub: 'Yield' },
  { label: 'CPI (YoY)', value: '3.4%', change: '-0.1', sub: 'Core: 3.6%' },
  { label: 'GDP Growth', value: '2.9%', change: '+0.3', changePct: 'QoQ Annlzd', sub: 'Q1 2025' },
];

export default function Markets() {
  return (
    <div className="min-h-screen pt-12 bg-terminal-bg">
      <div className="max-w-7xl mx-auto px-4 md:px-6 py-8">
        {/* Page header */}
        <div className="flex items-center justify-between mb-8">
          <div>
            <div className="flex items-center gap-2 mb-1">
              <span className="w-3 h-px bg-accent" />
              <span className="mono text-[10px] tracking-widest text-accent">MARKET OVERVIEW</span>
              <span className="mono text-[9px] px-2 py-0.5 border border-green/30 text-green animate-pulse-slow">● LIVE</span>
            </div>
            <h1 className="font-display font-700 text-2xl text-white">Market Dashboard</h1>
          </div>
          <div className="mono text-[11px] text-dim hidden md:block">
            {new Date().toLocaleDateString('en-US', { weekday: 'long', month: 'long', day: 'numeric', year: 'numeric' })}
          </div>
        </div>

        {/* Macro metrics */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-px bg-border mb-6">
          {MACRO_METRICS.map((m, i) => <MetricCard key={i} {...m} />)}
        </div>

        {/* Main chart */}
        <div className="mb-6">
          <ChartPanel />
        </div>

        {/* Two columns: Sectors + Movers */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Sector performance */}
          <div className="panel lg:col-span-2">
            <div className="panel-header">
              <span style={{ color: 'var(--amber)' }}>◑</span>
              SECTOR PERFORMANCE (1D)
            </div>
            <div className="divide-y divide-terminal">
              {SECTOR_PERFORMANCE.map((s) => (
                <div key={s.sector} className="flex items-center px-4 py-3 hover:bg-surface/50 transition-colors">
                  <span className="mono text-xs text-dim flex-1">{s.sector}</span>
                  {/* Bar */}
                  <div className="flex-1 mx-4 hidden sm:block">
                    <div className="h-1 rounded-full overflow-hidden bg-border">
                      <div
                        className="h-full rounded-full transition-all"
                        style={{
                          width: `${Math.min(Math.abs(parseFloat(s.chg)) * 15, 100)}%`,
                          background: s.up ? 'var(--green)' : 'var(--red)',
                        }}
                      />
                    </div>
                  </div>
                  <span className="mono text-xs font-500 w-16 text-right"
                        style={{ color: s.up ? 'var(--green)' : 'var(--red)' }}>
                    {s.chg}
                  </span>
                  <span className="mono text-xs text-dim w-16 text-right">{s.ytd}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Top movers */}
          <div className="space-y-4">
            {/* Gainers */}
            <div className="panel">
              <div className="panel-header">
                <span style={{ color: 'var(--green)' }}>▲</span>
                TOP GAINERS
              </div>
              <div className="divide-y divide-terminal">
                {TOP_MOVERS.gainers.map((s) => (
                  <div key={s.symbol} className="flex items-center justify-between px-4 py-2.5 hover:bg-surface transition-colors">
                    <span className="mono text-xs font-600 text-white">{s.symbol}</span>
                    <span className="mono text-xs text-terminal">{s.price}</span>
                    <span className="mono text-xs font-500" style={{ color: 'var(--green)' }}>{s.chg}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* Losers */}
            <div className="panel">
              <div className="panel-header">
                <span style={{ color: 'var(--red)' }}>▼</span>
                TOP LOSERS
              </div>
              <div className="divide-y divide-terminal">
                {TOP_MOVERS.losers.map((s) => (
                  <div key={s.symbol} className="flex items-center justify-between px-4 py-2.5 hover:bg-surface transition-colors">
                    <span className="mono text-xs font-600 text-white">{s.symbol}</span>
                    <span className="mono text-xs text-terminal">{s.price}</span>
                    <span className="mono text-xs font-500" style={{ color: 'var(--red)' }}>{s.chg}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>

        <p className="mono text-[10px] text-muted mt-6 text-center">
          ⚠ Market data is simulated for demonstration. Connect live APIs (Yahoo Finance, Alpha Vantage, Polygon) for real-time data.
        </p>
      </div>
    </div>
  );
}
