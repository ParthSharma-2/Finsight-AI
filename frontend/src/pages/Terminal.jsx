import { useState } from 'react';
import AITerminal from '../components/AITerminal';
import ChartPanel from '../components/ChartPanel';
import MetricCard from '../components/MetricCard';

const WATCHLIST = [
  { symbol: 'AAPL', name: 'Apple Inc.', price: '189.47', chg: '+1.23', pct: '+0.65%', up: true },
  { symbol: 'MSFT', name: 'Microsoft', price: '415.32', chg: '+3.11', pct: '+0.75%', up: true },
  { symbol: 'NVDA', name: 'NVIDIA', price: '875.20', chg: '-12.40', pct: '-1.40%', up: false },
  { symbol: 'GOOGL', name: 'Alphabet', price: '175.58', chg: '+2.89', pct: '+1.67%', up: true },
  { symbol: 'META', name: 'Meta Platforms', price: '524.13', chg: '+7.02', pct: '+1.36%', up: true },
  { symbol: 'AMZN', name: 'Amazon', price: '198.44', chg: '-0.87', pct: '-0.44%', up: false },
  { symbol: 'TSLA', name: 'Tesla', price: '181.29', chg: '+4.65', pct: '+2.64%', up: true },
  { symbol: 'JPM', name: 'JPMorgan Chase', price: '213.05', chg: '+1.85', pct: '+0.88%', up: true },
];

const SCREENER_DATA = [
  { symbol: 'NVDA', pe: '65.2', eps: '13.44', mktcap: '2.16T', rev: '+265%', sector: 'Tech' },
  { symbol: 'AAPL', pe: '28.7', eps: '6.57', mktcap: '2.91T', rev: '+3%', sector: 'Tech' },
  { symbol: 'MSFT', pe: '35.4', eps: '11.73', mktcap: '3.08T', rev: '+17%', sector: 'Tech' },
  { symbol: 'META', pe: '27.1', eps: '19.35', mktcap: '1.34T', rev: '+27%', sector: 'Tech' },
  { symbol: 'AMZN', pe: '53.8', eps: '3.68', mktcap: '2.09T', rev: '+13%', sector: 'Consumer' },
];

const TABS = ['CHAT', 'WATCHLIST', 'SCREENER', 'CHARTS'];

export default function Terminal() {
  const [activeTab, setActiveTab] = useState('CHAT');

  return (
    <div className="min-h-screen pt-12 bg-terminal-bg flex flex-col">
      {/* Terminal header bar */}
      <div className="border-b border-terminal px-6 py-2 flex items-center justify-between bg-surface">
        <div className="flex items-center gap-4">
          <span className="mono text-[11px] tracking-widest text-accent">FINSIGHT TERMINAL</span>
          <span className="mono text-[10px] text-muted">SESSION:{' '}
            <span className="text-dim">{Math.random().toString(36).slice(2, 10).toUpperCase()}</span>
          </span>
        </div>
        <div className="flex items-center gap-3 mono text-[10px] text-dim">
          <span>NYSE: OPEN</span>
          <span className="text-border">|</span>
          <span>NASDAQ: OPEN</span>
          <span className="text-border">|</span>
          <span className="text-green">● LIVE</span>
        </div>
      </div>

      <div className="flex-1 flex overflow-hidden">
        {/* Left sidebar — watchlist */}
        <aside className="hidden lg:flex flex-col w-56 border-r border-terminal bg-surface flex-shrink-0">
          <div className="panel-header">
            <span style={{ color: 'var(--accent)' }}>◉</span>
            WATCHLIST
          </div>
          <div className="flex-1 overflow-y-auto">
            {WATCHLIST.map((stock) => (
              <div key={stock.symbol} className="px-3 py-2.5 border-b border-terminal hover:bg-panel cursor-pointer transition-colors group">
                <div className="flex items-center justify-between mb-0.5">
                  <span className="mono text-xs font-600 text-white">{stock.symbol}</span>
                  <span className="mono text-xs font-500" style={{ color: stock.up ? 'var(--green)' : 'var(--red)' }}>
                    {stock.price}
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="mono text-[9px] text-muted truncate">{stock.name}</span>
                  <span className="mono text-[10px]" style={{ color: stock.up ? 'var(--green)' : 'var(--red)' }}>
                    {stock.pct}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </aside>

        {/* Main content area */}
        <main className="flex-1 flex flex-col overflow-hidden">
          {/* Tab bar */}
          <div className="flex border-b border-terminal bg-surface">
            {TABS.map((tab) => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                className="mono text-[11px] px-5 py-3 tracking-widest transition-all border-r border-terminal"
                style={{
                  color: activeTab === tab ? 'var(--accent)' : 'var(--dim)',
                  background: activeTab === tab ? 'var(--panel)' : 'transparent',
                  borderBottom: activeTab === tab ? '2px solid var(--accent)' : '2px solid transparent',
                }}
              >
                {tab}
              </button>
            ))}
          </div>

          {/* Tab content */}
          <div className="flex-1 overflow-auto p-4">
            {activeTab === 'CHAT' && <AITerminal />}

            {activeTab === 'WATCHLIST' && (
              <div className="panel">
                <div className="panel-header">
                  <span style={{ color: 'var(--accent)' }}>◉</span> FULL WATCHLIST
                </div>
                <table className="w-full">
                  <thead>
                    <tr className="border-b border-terminal">
                      {['SYMBOL', 'COMPANY', 'PRICE', 'CHANGE', '%', 'ACTION'].map(h => (
                        <th key={h} className="mono text-[10px] text-muted text-left px-4 py-2.5 tracking-wider">
                          {h}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {WATCHLIST.map((stock) => (
                      <tr key={stock.symbol} className="border-b border-terminal hover:bg-surface transition-colors cursor-pointer">
                        <td className="px-4 py-3 mono text-xs font-600 text-white">{stock.symbol}</td>
                        <td className="px-4 py-3 mono text-xs text-dim">{stock.name}</td>
                        <td className="px-4 py-3 mono text-xs font-500 text-terminal">{stock.price}</td>
                        <td className="px-4 py-3 mono text-xs" style={{ color: stock.up ? 'var(--green)' : 'var(--red)' }}>
                          {stock.chg}
                        </td>
                        <td className="px-4 py-3 mono text-xs" style={{ color: stock.up ? 'var(--green)' : 'var(--red)' }}>
                          {stock.pct}
                        </td>
                        <td className="px-4 py-3">
                          <button className="mono text-[10px] px-2 py-0.5 border border-terminal text-muted hover:text-accent hover:border-accent/40 transition-all">
                            ANALYZE
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}

            {activeTab === 'SCREENER' && (
              <div className="panel">
                <div className="panel-header">
                  <span style={{ color: 'var(--amber)' }}>◑</span> EQUITY SCREENER
                  <span className="ml-auto mono text-[10px] text-muted">Tech Sector | Large Cap</span>
                </div>
                <table className="w-full">
                  <thead>
                    <tr className="border-b border-terminal">
                      {['TICKER', 'P/E', 'EPS', 'MKT CAP', 'REV GROWTH', 'SECTOR'].map(h => (
                        <th key={h} className="mono text-[10px] text-muted text-left px-4 py-2.5 tracking-wider">
                          {h}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {SCREENER_DATA.map((row) => (
                      <tr key={row.symbol} className="border-b border-terminal hover:bg-surface transition-colors">
                        <td className="px-4 py-3 mono text-xs font-600 text-accent">{row.symbol}</td>
                        <td className="px-4 py-3 mono text-xs text-terminal">{row.pe}</td>
                        <td className="px-4 py-3 mono text-xs text-terminal">${row.eps}</td>
                        <td className="px-4 py-3 mono text-xs text-terminal">{row.mktcap}</td>
                        <td className="px-4 py-3 mono text-xs" style={{ color: 'var(--green)' }}>{row.rev}</td>
                        <td className="px-4 py-3 mono text-xs text-dim">{row.sector}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                <div className="px-4 pb-3 mono text-[9px] text-muted">
                  * Static demo data — connect Alpha Vantage / Polygon API for live screener
                </div>
              </div>
            )}

            {activeTab === 'CHARTS' && (
              <div className="space-y-4">
                <ChartPanel />
                <div className="grid grid-cols-2 gap-px bg-border">
                  {[
                    { label: 'MARKET CAP', value: '$2.91T', change: '+23B', changePct: '+0.8%' },
                    { label: '52W HIGH', value: '$199.62', sub: 'AAPL' },
                    { label: '52W LOW', value: '$164.08', sub: 'AAPL' },
                    { label: 'AVG VOLUME', value: '58.3M', sub: '30-day avg' },
                  ].map((m, i) => (
                    <MetricCard key={i} {...m} />
                  ))}
                </div>
              </div>
            )}
          </div>
        </main>

        {/* Right panel — quick metrics */}
        <aside className="hidden xl:flex flex-col w-52 border-l border-terminal bg-surface flex-shrink-0">
          <div className="panel-header">
            <span style={{ color: 'var(--green)' }}>◐</span>
            INDICES
          </div>
          <div className="p-3 space-y-3">
            {[
              { n: 'S&P 500', v: '5,277', c: '+0.27%', up: true },
              { n: 'DJIA', v: '39,069', c: '+0.10%', up: true },
              { n: 'NASDAQ', v: '16,428', c: '+0.44%', up: true },
              { n: 'VIX', v: '13.82', c: '-3.15%', up: false },
              { n: 'BTC', v: '$67,240', c: '+1.86%', up: true },
              { n: 'GOLD', v: '$2,347', c: '-0.14%', up: false },
              { n: 'OIL WTI', v: '$79.82', c: '+0.32%', up: true },
              { n: 'EUR/USD', v: '1.0847', c: '-0.08%', up: false },
              { n: '10Y UST', v: '4.52%', c: '+2bps', up: false },
              { n: 'FED RATE', v: '5.25%', c: 'HOLD', up: null },
            ].map(({ n, v, c, up }) => (
              <div key={n} className="flex items-center justify-between">
                <span className="mono text-[10px] text-dim">{n}</span>
                <div className="text-right">
                  <div className="mono text-[11px] text-terminal">{v}</div>
                  <div className="mono text-[9px]" style={{
                    color: up === null ? 'var(--amber)' : up ? 'var(--green)' : 'var(--red)'
                  }}>{c}</div>
                </div>
              </div>
            ))}
          </div>
        </aside>
      </div>
    </div>
  );
}
