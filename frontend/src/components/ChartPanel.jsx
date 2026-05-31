import { useState } from 'react';
import {
  AreaChart, Area, LineChart, Line, XAxis, YAxis,
  Tooltip, ResponsiveContainer, CartesianGrid,
} from 'recharts';

// Synthetic price data generator
function generatePriceData(basePrice, days = 60, volatility = 0.02) {
  const data = [];
  let price = basePrice;
  const now = new Date();
  for (let i = days; i >= 0; i--) {
    const date = new Date(now);
    date.setDate(date.getDate() - i);
    price = price * (1 + (Math.random() - 0.48) * volatility);
    data.push({
      date: date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
      price: parseFloat(price.toFixed(2)),
      volume: Math.floor(Math.random() * 50000000 + 10000000),
    });
  }
  return data;
}

const MOCK_CHARTS = {
  AAPL: generatePriceData(175, 60, 0.018),
  MSFT: generatePriceData(400, 60, 0.016),
  NVDA: generatePriceData(820, 60, 0.032),
  SPY:  generatePriceData(510, 60, 0.01),
};

const PERIODS = ['1W', '1M', '3M', '6M', '1Y'];

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="panel px-3 py-2">
      <div className="mono text-[10px] text-dim mb-1">{label}</div>
      <div className="mono text-[13px] font-600" style={{ color: 'var(--accent)' }}>
        ${payload[0].value?.toLocaleString()}
      </div>
    </div>
  );
};

export default function ChartPanel() {
  const [selected, setSelected] = useState('AAPL');
  const [period, setPeriod] = useState('1M');

  const allData = MOCK_CHARTS[selected] || MOCK_CHARTS.AAPL;
  const periodMap = { '1W': 7, '1M': 30, '3M': 60, '6M': 60, '1Y': 60 };
  const data = allData.slice(-Math.min(periodMap[period], allData.length));

  const firstPrice = data[0]?.price || 0;
  const lastPrice = data[data.length - 1]?.price || 0;
  const pctChange = (((lastPrice - firstPrice) / firstPrice) * 100).toFixed(2);
  const isPositive = lastPrice >= firstPrice;
  const lineColor = isPositive ? '#00ff88' : '#ff4466';

  return (
    <div className="panel">
      {/* Header */}
      <div className="panel-header justify-between">
        <div className="flex items-center gap-2">
          <span style={{ color: 'var(--accent)' }}>◈</span>
          PRICE CHART
        </div>
        <div className="flex items-center gap-1">
          {Object.keys(MOCK_CHARTS).map((sym) => (
            <button
              key={sym}
              onClick={() => setSelected(sym)}
              className="mono text-[10px] px-2 py-0.5 transition-all"
              style={{
                color: selected === sym ? 'var(--accent)' : 'var(--muted)',
                borderBottom: selected === sym ? '1px solid var(--accent)' : '1px solid transparent',
              }}
            >
              {sym}
            </button>
          ))}
        </div>
      </div>

      {/* Price header */}
      <div className="px-4 py-3 flex items-end justify-between border-b border-terminal">
        <div>
          <div className="mono text-2xl font-600" style={{ color: 'var(--text)' }}>
            ${lastPrice.toFixed(2)}
          </div>
          <div className="mono text-[11px] mt-0.5" style={{ color: isPositive ? 'var(--green)' : 'var(--red)' }}>
            {isPositive ? '▲' : '▼'} {Math.abs(lastPrice - firstPrice).toFixed(2)} ({isPositive ? '+' : ''}{pctChange}%)
          </div>
        </div>
        {/* Period selector */}
        <div className="flex items-center gap-1">
          {PERIODS.map((p) => (
            <button
              key={p}
              onClick={() => setPeriod(p)}
              className="mono text-[10px] px-2.5 py-1 transition-all border"
              style={{
                background: period === p ? lineColor + '15' : 'transparent',
                borderColor: period === p ? lineColor + '50' : 'var(--border)',
                color: period === p ? lineColor : 'var(--dim)',
              }}
            >
              {p}
            </button>
          ))}
        </div>
      </div>

      {/* Chart */}
      <div className="p-4 h-52">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data} margin={{ top: 5, right: 5, bottom: 0, left: 0 }}>
            <defs>
              <linearGradient id="priceGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor={lineColor} stopOpacity={0.15} />
                <stop offset="95%" stopColor={lineColor} stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e2d3d" vertical={false} />
            <XAxis
              dataKey="date"
              tick={{ fontFamily: 'JetBrains Mono', fontSize: 9, fill: '#4a6278' }}
              tickLine={false}
              axisLine={false}
              interval={Math.floor(data.length / 5)}
            />
            <YAxis
              tick={{ fontFamily: 'JetBrains Mono', fontSize: 9, fill: '#4a6278' }}
              tickLine={false}
              axisLine={false}
              tickFormatter={(v) => `$${v.toFixed(0)}`}
              width={55}
              domain={['auto', 'auto']}
            />
            <Tooltip content={<CustomTooltip />} />
            <Area
              type="monotone"
              dataKey="price"
              stroke={lineColor}
              strokeWidth={1.5}
              fill="url(#priceGrad)"
              dot={false}
              activeDot={{ r: 3, fill: lineColor, strokeWidth: 0 }}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      <div className="px-4 pb-2 mono text-[9px] text-muted">
        * Simulated data — connect Yahoo Finance API for live prices
      </div>
    </div>
  );
}
