import { useState, useEffect } from 'react';
import { Plus, RefreshCw, TrendingUp, TrendingDown, Sparkles, LineChart, Table, MessageSquare, BarChart3, Layers } from 'lucide-react';
import AITerminal from '../components/AITerminal';
import ChartPanel from '../components/ChartPanel';
import MetricCard from '../components/MetricCard';
import { getStockQuotes } from '../lib/api';

const DEFAULT_SYMBOLS = ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'META', 'AMZN', 'TSLA', 'INFY.NS'];

const COMPANY_NAMES = {
  AAPL: 'Apple Inc.',
  MSFT: 'Microsoft Corp',
  NVDA: 'NVIDIA Corp',
  GOOGL: 'Alphabet Inc.',
  META: 'Meta Platforms',
  AMZN: 'Amazon.com',
  TSLA: 'Tesla Inc.',
  'INFY.NS': 'Infosys Ltd',
};

const SCREENER_DATA = [
  { symbol: 'NVDA', pe: '54.2x', eps: '$2.82', mktcap: '$3.12T', rev: '+122%', sector: 'Semiconductors' },
  { symbol: 'AAPL', pe: '33.5x', eps: '$6.57', mktcap: '$3.48T', rev: '+4.9%', sector: 'Consumer Tech' },
  { symbol: 'MSFT', pe: '35.4x', eps: '$11.80', mktcap: '$3.31T', rev: '+15.2%', sector: 'Enterprise Cloud' },
  { symbol: 'META', pe: '28.1x', eps: '$21.20', mktcap: '$1.49T', rev: '+22.1%', sector: 'Digital Media' },
  { symbol: 'AMZN', pe: '42.8x', eps: '$4.18', mktcap: '$1.94T', rev: '+11.0%', sector: 'E-Commerce / Cloud' },
];

const TABS = [
  { id: 'CHAT', label: 'AI Terminal', icon: MessageSquare },
  { id: 'WATCHLIST', label: 'Watchlist', icon: Table },
  { id: 'CHARTS', label: 'Live Charts', icon: LineChart },
  { id: 'SCREENER', label: 'Screener', icon: BarChart3 },
];

export default function Terminal() {
  const [activeTab, setActiveTab] = useState('CHAT');
  const [symbols, setSymbols] = useState(() => {
    const saved = localStorage.getItem('finsight-watchlist');
    return saved ? JSON.parse(saved) : DEFAULT_SYMBOLS;
  });
  const [watchlistData, setWatchlistData] = useState([]);
  const [loading, setLoading] = useState(false);
  const [newTickerInput, setNewTickerInput] = useState('');

  const fetchLiveWatchlist = async (symList) => {
    setLoading(true);
    try {
      const quotes = await getStockQuotes(symList);
      if (Array.isArray(quotes)) {
        const enriched = quotes.map((q) => ({
          symbol: q.symbol,
          name: COMPANY_NAMES[q.symbol] || q.symbol,
          price: q.price !== null && q.price !== undefined ? `$${q.price}` : 'Loading...',
          chg: q.change ? `${q.change > 0 ? '+' : ''}${q.change}` : '—',
          pct: q.change_pct ? `${q.change_pct > 0 ? '+' : ''}${q.change_pct}%` : '—',
          up: q.up,
        }));
        setWatchlistData(enriched);
      }
    } catch (err) {
      console.warn('Failed to fetch live watchlist:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchLiveWatchlist(symbols);
  }, [symbols]);

  const handleAddTicker = (e) => {
    e.preventDefault();
    const clean = newTickerInput.trim().toUpperCase();
    if (clean && !symbols.includes(clean)) {
      const updated = [...symbols, clean];
      setSymbols(updated);
      localStorage.setItem('finsight-watchlist', JSON.stringify(updated));
      setNewTickerInput('');
    }
  };

  const handleRemoveTicker = (sym) => {
    const updated = symbols.filter(s => s !== sym);
    setSymbols(updated);
    localStorage.setItem('finsight-watchlist', JSON.stringify(updated));
  };

  return (
    <div className="min-h-[calc(100vh-3.5rem)] flex flex-col bg-zinc-50 dark:bg-zinc-950 text-zinc-900 dark:text-zinc-50">
      {/* Top Status Bar */}
      <div className="border-b border-zinc-200 dark:border-zinc-800 px-6 py-2.5 flex items-center justify-between bg-white dark:bg-zinc-900/60 text-xs font-mono">
        <div className="flex items-center gap-3">
          <span className="font-bold text-indigo-600 dark:text-indigo-400">FINSIGHT PRO TERMINAL</span>
          <span className="text-zinc-400">|</span>
          <span className="text-zinc-500">LIVE FEED: <span className="text-emerald-500 font-semibold">ACTIVE</span></span>
        </div>
        <div className="flex items-center gap-4 text-zinc-500 dark:text-zinc-400">
          <span className="hidden sm:inline">NYSE: OPEN</span>
          <span className="hidden sm:inline">•</span>
          <span className="hidden sm:inline">NASDAQ: OPEN</span>
          <span className="hidden sm:inline">•</span>
          <span className="flex items-center gap-1.5 text-emerald-600 dark:text-emerald-400 font-semibold">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
            REAL-TIME
          </span>
        </div>
      </div>

      <div className="flex-1 flex overflow-hidden">
        {/* Left Sidebar — Live Watchlist */}
        <aside className="hidden lg:flex flex-col w-64 border-r border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900/40 shrink-0">
          <div className="p-3 border-b border-zinc-200 dark:border-zinc-800 flex items-center justify-between">
            <span className="text-xs font-semibold text-zinc-900 dark:text-zinc-100 flex items-center gap-1.5">
              <span>Watchlist</span>
              <span className="text-[10px] font-mono text-zinc-400 font-normal">({symbols.length})</span>
            </span>
            <button
              onClick={() => fetchLiveWatchlist(symbols)}
              title="Refresh live quotes"
              className="p-1 text-zinc-400 hover:text-zinc-600 dark:hover:text-zinc-200 rounded"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin text-indigo-500' : ''}`} />
            </button>
          </div>

          {/* Add Ticker input */}
          <form onSubmit={handleAddTicker} className="p-2 border-b border-zinc-200/80 dark:border-zinc-800/80">
            <div className="relative">
              <input
                type="text"
                value={newTickerInput}
                onChange={(e) => setNewTickerInput(e.target.value)}
                placeholder="+ Add symbol (e.g. TSLA)"
                className="w-full bg-zinc-100 dark:bg-zinc-800/60 text-xs font-mono text-zinc-900 dark:text-zinc-100 placeholder:text-zinc-400 px-3 py-1.5 rounded-lg border border-transparent focus:border-indigo-500 focus:outline-none"
              />
            </div>
          </form>

          {/* Watchlist Items */}
          <div className="flex-1 overflow-y-auto divide-y divide-zinc-100 dark:divide-zinc-800/60">
            {watchlistData.map((stock) => (
              <div
                key={stock.symbol}
                className="p-3 hover:bg-zinc-50 dark:hover:bg-zinc-800/40 cursor-pointer transition-colors group"
              >
                <div className="flex items-center justify-between mb-0.5">
                  <span className="font-mono text-xs font-bold text-zinc-900 dark:text-zinc-100">
                    {stock.symbol}
                  </span>
                  <span className="font-mono text-xs font-semibold text-zinc-900 dark:text-zinc-100">
                    {stock.price}
                  </span>
                </div>
                <div className="flex items-center justify-between text-[11px] font-mono">
                  <span className="text-zinc-400 truncate max-w-[110px]">{stock.name}</span>
                  <span className={`font-semibold ${stock.up ? 'text-emerald-600 dark:text-emerald-400' : 'text-rose-600 dark:text-rose-400'}`}>
                    {stock.pct}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </aside>

        {/* Main Workspace Area */}
        <main className="flex-1 flex flex-col overflow-hidden">
          {/* Tab Navigation */}
          <div className="flex border-b border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900/60 px-4">
            {TABS.map((tab) => {
              const Icon = tab.icon;
              const active = activeTab === tab.id;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`flex items-center gap-2 text-xs font-semibold px-4 py-3 border-b-2 transition-all ${
                    active
                      ? 'border-indigo-600 text-indigo-600 dark:text-indigo-400'
                      : 'border-transparent text-zinc-500 hover:text-zinc-900 dark:hover:text-zinc-200'
                  }`}
                >
                  <Icon className="w-3.5 h-3.5" />
                  <span>{tab.label}</span>
                </button>
              );
            })}
          </div>

          {/* Active Tab View */}
          <div className="flex-1 overflow-auto p-4 sm:p-6">
            {activeTab === 'CHAT' && <AITerminal />}

            {activeTab === 'WATCHLIST' && (
              <div className="fin-card p-6">
                <div className="flex items-center justify-between pb-4 mb-4 border-b border-zinc-100 dark:border-zinc-800">
                  <div>
                    <h3 className="font-semibold text-base text-zinc-900 dark:text-zinc-100">
                      Live Watchlist Portfolio
                    </h3>
                    <p className="text-xs text-zinc-500">Real-time quotes fetched live via Yahoo Finance</p>
                  </div>
                  <button
                    onClick={() => fetchLiveWatchlist(symbols)}
                    className="flex items-center gap-1.5 text-xs font-semibold px-3 py-1.5 rounded-lg bg-zinc-100 dark:bg-zinc-800 text-zinc-700 dark:text-zinc-300 hover:bg-zinc-200 dark:hover:bg-zinc-700 transition-colors"
                  >
                    <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
                    <span>Refresh Quotes</span>
                  </button>
                </div>

                <div className="overflow-x-auto">
                  <table className="w-full text-left text-xs font-mono">
                    <thead>
                      <tr className="border-b border-zinc-100 dark:border-zinc-800 text-zinc-400 uppercase text-[10px]">
                        <th className="py-2.5 px-3">Symbol</th>
                        <th className="py-2.5 px-3">Company</th>
                        <th className="py-2.5 px-3">Live Price</th>
                        <th className="py-2.5 px-3">Change ($)</th>
                        <th className="py-2.5 px-3">Change (%)</th>
                        <th className="py-2.5 px-3 text-right">Actions</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-zinc-100 dark:divide-zinc-800/60">
                      {watchlistData.map((s) => (
                        <tr key={s.symbol} className="hover:bg-zinc-50 dark:hover:bg-zinc-900/40 transition-colors">
                          <td className="py-3 px-3 font-bold text-zinc-900 dark:text-zinc-100">{s.symbol}</td>
                          <td className="py-3 px-3 text-zinc-500">{s.name}</td>
                          <td className="py-3 px-3 font-bold text-zinc-900 dark:text-zinc-100">{s.price}</td>
                          <td className={`py-3 px-3 font-semibold ${s.up ? 'text-emerald-600 dark:text-emerald-400' : 'text-rose-600 dark:text-rose-400'}`}>
                            {s.chg}
                          </td>
                          <td className={`py-3 px-3 font-semibold ${s.up ? 'text-emerald-600 dark:text-emerald-400' : 'text-rose-600 dark:text-rose-400'}`}>
                            {s.pct}
                          </td>
                          <td className="py-3 px-3 text-right">
                            <button
                              onClick={() => handleRemoveTicker(s.symbol)}
                              className="text-[11px] text-zinc-400 hover:text-rose-500 transition-colors"
                            >
                              Remove
                            </button>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}

            {activeTab === 'CHARTS' && (
              <div className="space-y-6">
                <ChartPanel />
                <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
                  {[
                    { label: 'MARKET CAP', value: '$3.48T', change: '+24B', changePct: '+0.7%' },
                    { label: '52W HIGH', value: '$237.23', sub: 'AAPL' },
                    { label: '52W LOW', value: '$164.08', sub: 'AAPL' },
                    { label: 'P/E RATIO', value: '33.5x', sub: 'TRAILING' },
                  ].map((m, i) => (
                    <MetricCard key={i} {...m} />
                  ))}
                </div>
              </div>
            )}

            {activeTab === 'SCREENER' && (
              <div className="fin-card p-6">
                <div className="flex items-center justify-between pb-4 mb-4 border-b border-zinc-100 dark:border-zinc-800">
                  <h3 className="font-semibold text-base text-zinc-900 dark:text-zinc-100">
                    Equity Valuation Screener
                  </h3>
                  <span className="text-xs font-mono text-zinc-400">Large Cap Tech</span>
                </div>

                <div className="overflow-x-auto">
                  <table className="w-full text-left text-xs font-mono">
                    <thead>
                      <tr className="border-b border-zinc-100 dark:border-zinc-800 text-zinc-400 uppercase text-[10px]">
                        <th className="py-2.5 px-3">Ticker</th>
                        <th className="py-2.5 px-3">P/E</th>
                        <th className="py-2.5 px-3">EPS</th>
                        <th className="py-2.5 px-3">Market Cap</th>
                        <th className="py-2.5 px-3">Revenue YoY</th>
                        <th className="py-2.5 px-3">Sector</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-zinc-100 dark:divide-zinc-800/60">
                      {SCREENER_DATA.map((row) => (
                        <tr key={row.symbol} className="hover:bg-zinc-50 dark:hover:bg-zinc-900/40 transition-colors">
                          <td className="py-3 px-3 font-bold text-indigo-600 dark:text-indigo-400">{row.symbol}</td>
                          <td className="py-3 px-3 text-zinc-700 dark:text-zinc-300">{row.pe}</td>
                          <td className="py-3 px-3 text-zinc-700 dark:text-zinc-300">{row.eps}</td>
                          <td className="py-3 px-3 text-zinc-700 dark:text-zinc-300">{row.mktcap}</td>
                          <td className="py-3 px-3 font-semibold text-emerald-600 dark:text-emerald-400">{row.rev}</td>
                          <td className="py-3 px-3 text-zinc-500">{row.sector}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </div>
        </main>
      </div>
    </div>
  );
}
