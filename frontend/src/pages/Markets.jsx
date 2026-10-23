import { TrendingUp, TrendingDown, Sparkles } from 'lucide-react';
import ChartPanel from '../components/ChartPanel';
import MetricCard from '../components/MetricCard';

const SECTOR_PERFORMANCE = [
  { sector: 'Technology', chg: '+2.14%', ytd: '+28.4%', up: true },
  { sector: 'Communication Services', chg: '+1.78%', ytd: '+22.1%', up: true },
  { sector: 'Financials', chg: '+1.23%', ytd: '+18.7%', up: true },
  { sector: 'Healthcare', chg: '+0.87%', ytd: '+12.1%', up: true },
  { sector: 'Industrials', chg: '+0.56%', ytd: '+11.3%', up: true },
  { sector: 'Materials', chg: '+0.33%', ytd: '+7.5%', up: true },
  { sector: 'Consumer Discretionary', chg: '-0.12%', ytd: '+9.8%', up: false },
  { sector: 'Energy', chg: '-0.45%', ytd: '+5.2%', up: false },
  { sector: 'Utilities', chg: '-0.78%', ytd: '-2.1%', up: false },
  { sector: 'Real Estate', chg: '-1.02%', ytd: '-4.7%', up: false },
];

const TOP_MOVERS = {
  gainers: [
    { symbol: 'NVDA', name: 'NVIDIA Corp', price: '$126.40', chg: '+4.2%' },
    { symbol: 'SMCI', name: 'Super Micro', price: '$48.20', chg: '+3.9%' },
    { symbol: 'PLTR', name: 'Palantir Tech', price: '$42.15', chg: '+3.1%' },
    { symbol: 'ARM',  name: 'Arm Holdings', price: '$138.80', chg: '+2.8%' },
  ],
  losers: [
    { symbol: 'INTC', name: 'Intel Corp', price: '$22.10', chg: '-3.8%' },
    { symbol: 'WBD',  name: 'Warner Bros', price: '$7.45', chg: '-2.9%' },
    { symbol: 'PARA', name: 'Paramount', price: '$10.80', chg: '-2.4%' },
    { symbol: 'VFC',  name: 'V.F. Corp', price: '$15.20', chg: '-1.9%' },
  ],
};

const MACRO_METRICS = [
  { label: 'S&P 500', value: '5,864.67', change: '+24.12', changePct: '+0.41%', sub: 'INDEX' },
  { label: 'NASDAQ 100', value: '20,388.20', change: '+142.50', changePct: '+0.70%', sub: 'INDEX' },
  { label: '10Y TREASURY', value: '4.24%', change: '-0.03', changePct: '-0.70%', sub: 'YIELD' },
  { label: 'FED FUNDS RATE', value: '4.75–5.00%', sub: 'FOMC TARGET', accent: true },
];

export default function Markets({ onOpenCopilot }) {
  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 py-8">
      {/* Page Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-8">
        <div>
          <div className="flex items-center gap-2 mb-1.5">
            <span className="text-xs font-mono font-medium text-indigo-600 dark:text-indigo-400 uppercase tracking-wider">
              Market Dashboard
            </span>
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
            <span className="text-[11px] font-mono text-emerald-600 dark:text-emerald-400">Live Global Feeds</span>
          </div>
          <h1 className="text-2xl sm:text-3xl font-bold tracking-tight text-zinc-900 dark:text-zinc-50">
            Market Intelligence
          </h1>
        </div>

        <button
          onClick={onOpenCopilot}
          className="self-start sm:self-auto flex items-center gap-2 px-4 py-2 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold shadow-sm transition-all hover:scale-[1.02] active:scale-[0.98]"
        >
          <Sparkles className="w-3.5 h-3.5" />
          <span>Ask Copilot About Markets</span>
        </button>
      </div>

      {/* Macro Metric Cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
        {MACRO_METRICS.map((m, i) => (
          <MetricCard key={i} {...m} />
        ))}
      </div>

      {/* Main Interactive Chart */}
      <div className="mb-6">
        <ChartPanel />
      </div>

      {/* Sectors and Top Movers */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Sector Performance */}
        <div className="fin-card p-5 sm:p-6 lg:col-span-2">
          <div className="flex items-center justify-between pb-3 mb-4 border-b border-zinc-100 dark:border-zinc-800/80">
            <span className="text-sm font-semibold text-zinc-900 dark:text-zinc-100">
              Sector Performance (1D)
            </span>
            <span className="text-xs font-mono text-zinc-400">YTD Return</span>
          </div>

          <div className="space-y-3">
            {SECTOR_PERFORMANCE.map((s) => (
              <div key={s.sector} className="flex items-center text-xs font-mono">
                <span className="w-44 text-zinc-700 dark:text-zinc-300 truncate">{s.sector}</span>

                {/* Performance bar */}
                <div className="flex-1 mx-3 hidden sm:block">
                  <div className="h-1.5 w-full bg-zinc-100 dark:bg-zinc-800 rounded-full overflow-hidden">
                    <div
                      className={`h-full rounded-full ${
                        s.up ? 'bg-emerald-500' : 'bg-rose-500'
                      }`}
                      style={{
                        width: `${Math.min(Math.abs(parseFloat(s.chg)) * 25, 100)}%`,
                      }}
                    />
                  </div>
                </div>

                <span
                  className={`w-16 text-right font-semibold ${
                    s.up ? 'text-emerald-600 dark:text-emerald-400' : 'text-rose-600 dark:text-rose-400'
                  }`}
                >
                  {s.chg}
                </span>

                <span className="w-14 text-right text-zinc-400">{s.ytd}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Top Movers */}
        <div className="space-y-4">
          {/* Gainers */}
          <div className="fin-card p-5">
            <div className="flex items-center gap-1.5 text-xs font-semibold text-emerald-600 dark:text-emerald-400 mb-3">
              <TrendingUp className="w-4 h-4" />
              <span>Top Gainers</span>
            </div>
            <div className="space-y-2.5">
              {TOP_MOVERS.gainers.map((m) => (
                <div key={m.symbol} className="flex items-center justify-between text-xs font-mono py-1 border-b border-zinc-100 dark:border-zinc-800/60 last:border-0">
                  <div>
                    <span className="font-bold text-zinc-900 dark:text-zinc-100 mr-2">{m.symbol}</span>
                    <span className="text-[11px] text-zinc-400">{m.name}</span>
                  </div>
                  <div className="text-right">
                    <span className="text-zinc-700 dark:text-zinc-300 mr-2">{m.price}</span>
                    <span className="font-semibold text-emerald-600 dark:text-emerald-400">{m.chg}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Losers */}
          <div className="fin-card p-5">
            <div className="flex items-center gap-1.5 text-xs font-semibold text-rose-600 dark:text-rose-400 mb-3">
              <TrendingDown className="w-4 h-4" />
              <span>Top Losers</span>
            </div>
            <div className="space-y-2.5">
              {TOP_MOVERS.losers.map((m) => (
                <div key={m.symbol} className="flex items-center justify-between text-xs font-mono py-1 border-b border-zinc-100 dark:border-zinc-800/60 last:border-0">
                  <div>
                    <span className="font-bold text-zinc-900 dark:text-zinc-100 mr-2">{m.symbol}</span>
                    <span className="text-[11px] text-zinc-400">{m.name}</span>
                  </div>
                  <div className="text-right">
                    <span className="text-zinc-700 dark:text-zinc-300 mr-2">{m.price}</span>
                    <span className="font-semibold text-rose-600 dark:text-rose-400">{m.chg}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
