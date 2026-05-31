// Mock ticker data — replace with live API data when backend market endpoints are ready
const TICKER_DATA = [
  { symbol: 'AAPL', price: '189.47', change: '+1.23', pct: '+0.65%', up: true },
  { symbol: 'MSFT', price: '415.32', change: '+3.11', pct: '+0.75%', up: true },
  { symbol: 'NVDA', price: '875.20', change: '-12.40', pct: '-1.40%', up: false },
  { symbol: 'GOOGL', price: '175.58', change: '+2.89', pct: '+1.67%', up: true },
  { symbol: 'META', price: '524.13', change: '+7.02', pct: '+1.36%', up: true },
  { symbol: 'AMZN', price: '198.44', change: '-0.87', pct: '-0.44%', up: false },
  { symbol: 'TSLA', price: '181.29', change: '+4.65', pct: '+2.64%', up: true },
  { symbol: 'BTC-USD', price: '67,240', change: '+1,230', pct: '+1.86%', up: true },
  { symbol: 'SPY', price: '527.91', change: '+1.44', pct: '+0.27%', up: true },
  { symbol: 'QQQ', price: '455.22', change: '+2.18', pct: '+0.48%', up: true },
  { symbol: 'GLD', price: '224.67', change: '-0.32', pct: '-0.14%', up: false },
  { symbol: 'JPM', price: '213.05', change: '+1.85', pct: '+0.88%', up: true },
];

export default function TickerTape() {
  const items = [...TICKER_DATA, ...TICKER_DATA]; // duplicate for seamless loop

  return (
    <div className="w-full border-b border-terminal bg-surface overflow-hidden">
      <div className="ticker-wrap py-1.5 relative">
        {/* Fade edges */}
        <div className="absolute left-0 top-0 bottom-0 w-16 z-10 pointer-events-none"
             style={{ background: 'linear-gradient(to right, var(--surface), transparent)' }} />
        <div className="absolute right-0 top-0 bottom-0 w-16 z-10 pointer-events-none"
             style={{ background: 'linear-gradient(to left, var(--surface), transparent)' }} />

        <div className="ticker-inner">
          {items.map((item, i) => (
            <span key={i} className="inline-flex items-center gap-2 px-5">
              <span className="mono text-[11px] font-600 text-white">{item.symbol}</span>
              <span className="mono text-[11px] text-terminal">{item.price}</span>
              <span
                className="mono text-[10px] font-500"
                style={{ color: item.up ? 'var(--green)' : 'var(--red)' }}
              >
                {item.change} ({item.pct})
              </span>
              <span className="text-[#1e2d3d] mx-1">│</span>
            </span>
          ))}
        </div>
      </div>
    </div>
  );
}
