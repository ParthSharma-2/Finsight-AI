const TICKER_DATA = [
  { symbol: 'AAPL', price: '228.45', change: '+2.14', pct: '+0.95%', up: true },
  { symbol: 'NVDA', price: '126.40', change: '+5.10', pct: '+4.20%', up: true },
  { symbol: 'MSFT', price: '445.12', change: '+3.20', pct: '+0.72%', up: true },
  { symbol: 'GOOGL', price: '168.80', change: '-0.85', pct: '-0.50%', up: false },
  { symbol: 'AMZN', price: '186.50', change: '+1.45', pct: '+0.78%', up: true },
  { symbol: 'META', price: '592.20', change: '+8.40', pct: '+1.44%', up: true },
  { symbol: 'TSLA', price: '218.90', change: '-4.10', pct: '-1.84%', up: false },
  { symbol: 'SPY', price: '585.60', change: '+2.80', pct: '+0.48%', up: true },
  { symbol: 'QQQ', price: '502.10', change: '+3.40', pct: '+0.68%', up: true },
];

export default function TickerTape() {
  const items = [...TICKER_DATA, ...TICKER_DATA];

  return (
    <div className="w-full border-b border-zinc-200/80 dark:border-zinc-800/80 bg-zinc-50/50 dark:bg-zinc-900/40 backdrop-blur-sm overflow-hidden py-1.5 transition-colors duration-300">
      <div className="flex items-center overflow-x-auto no-scrollbar whitespace-nowrap">
        <div className="inline-flex items-center gap-6 animate-ticker">
          {items.map((item, i) => (
            <span key={i} className="inline-flex items-center gap-2 text-xs font-mono">
              <span className="font-bold text-zinc-900 dark:text-zinc-100">{item.symbol}</span>
              <span className="text-zinc-500 dark:text-zinc-400">${item.price}</span>
              <span
                className={`font-semibold ${
                  item.up ? 'text-emerald-600 dark:text-emerald-400' : 'text-rose-600 dark:text-rose-400'
                }`}
              >
                {item.up ? '▲' : '▼'} {item.pct}
              </span>
              <span className="text-zinc-300 dark:text-zinc-800 ml-2">•</span>
            </span>
          ))}
        </div>
      </div>
    </div>
  );
}
