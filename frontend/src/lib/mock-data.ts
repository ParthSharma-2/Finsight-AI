// Mock financial data for FinSight AI
export type StockTick = {
  symbol: string;
  name: string;
  price: number;
  change: number;
  changePct: number;
  marketCap: string;
  sector: string;
};

export const watchlist: StockTick[] = [
  { symbol: "NVDA", name: "NVIDIA Corp.", price: 1284.42, change: 28.11, changePct: 2.24, marketCap: "$3.15T", sector: "Semiconductors" },
  { symbol: "AAPL", name: "Apple Inc.", price: 232.18, change: -1.42, changePct: -0.61, marketCap: "$3.52T", sector: "Consumer Tech" },
  { symbol: "MSFT", name: "Microsoft Corp.", price: 451.07, change: 3.84, changePct: 0.86, marketCap: "$3.36T", sector: "Software" },
  { symbol: "GOOGL", name: "Alphabet Inc.", price: 187.55, change: 1.22, changePct: 0.65, marketCap: "$2.31T", sector: "Internet" },
  { symbol: "TSLA", name: "Tesla Inc.", price: 312.88, change: -5.74, changePct: -1.80, marketCap: "$998B", sector: "Auto" },
  { symbol: "AMZN", name: "Amazon.com Inc.", price: 218.94, change: 2.17, changePct: 1.00, marketCap: "$2.27T", sector: "E-Commerce" },
  { symbol: "META", name: "Meta Platforms", price: 612.30, change: 8.55, changePct: 1.42, marketCap: "$1.55T", sector: "Social" },
  { symbol: "BRK.B", name: "Berkshire Hathaway", price: 482.11, change: 0.42, changePct: 0.09, marketCap: "$1.04T", sector: "Conglomerate" },
];

export const trending: StockTick[] = [
  { symbol: "PLTR", name: "Palantir Tech", price: 78.22, change: 4.31, changePct: 5.83, marketCap: "$178B", sector: "Software" },
  { symbol: "SMCI", name: "Super Micro", price: 56.41, change: 3.92, changePct: 7.47, marketCap: "$33B", sector: "Hardware" },
  { symbol: "COIN", name: "Coinbase Global", price: 312.66, change: 12.04, changePct: 4.00, marketCap: "$79B", sector: "Crypto" },
  { symbol: "ARM", name: "Arm Holdings", price: 152.10, change: -3.22, changePct: -2.07, marketCap: "$158B", sector: "Semis" },
];

// Generate deterministic price series
export function priceSeries(
  base: number,
  points = 60,
  volatility = 0.012,
  seed = 7,
) {
  let v = base;
  let s = seed;
  const rng = () => {
    s = (s * 9301 + 49297) % 233280;
    return s / 233280;
  };
  return Array.from({ length: points }, (_, i) => {
    v = v * (1 + (rng() - 0.48) * volatility);
    return { i, t: `T-${points - i}`, value: +v.toFixed(2) };
  });
}

export const marketIndexes = [
  { name: "S&P 500", value: 5982.41, change: 0.74 },
  { name: "Nasdaq 100", value: 21442.18, change: 1.12 },
  { name: "Dow Jones", value: 43021.55, change: 0.31 },
  { name: "VIX", value: 14.22, change: -3.41 },
  { name: "US 10Y", value: 4.21, change: -0.04 },
  { name: "BTC", value: 98412.55, change: 2.18 },
];

export const news = [
  {
    id: 1,
    source: "Bloomberg",
    time: "4m",
    title: "Nvidia briefly tops $3.5T as Blackwell ramps faster than guided",
    tag: "AI / Semis",
    sentiment: "bullish",
  },
  {
    id: 2,
    source: "Reuters",
    time: "12m",
    title: "Fed minutes signal a measured path; markets price 67% cut probability for March",
    tag: "Macro",
    sentiment: "neutral",
  },
  {
    id: 3,
    source: "FT",
    time: "29m",
    title: "Apple negotiating multi-year cloud-compute deal with Anthropic, sources say",
    tag: "M&A",
    sentiment: "bullish",
  },
  {
    id: 4,
    source: "WSJ",
    time: "1h",
    title: "Tesla recalls 240k vehicles over software defect in self-driving stack",
    tag: "Risk",
    sentiment: "bearish",
  },
  {
    id: 5,
    source: "CNBC",
    time: "2h",
    title: "Palantir surges as US Army expands TITAN contract by $618M",
    tag: "Earnings",
    sentiment: "bullish",
  },
];

export const insights = [
  {
    id: "i1",
    agent: "Research Agent",
    title: "NVDA: data-center revenue +112% YoY likely to beat consensus by 4–6%",
    confidence: 0.86,
    time: "just now",
  },
  {
    id: "i2",
    agent: "Risk Agent",
    title: "Portfolio beta drifted to 1.34 — consider trimming high-vol semis exposure",
    confidence: 0.74,
    time: "8m",
  },
  {
    id: "i3",
    agent: "News Agent",
    title: "Sentiment on AAPL turned net positive (+0.31) post-WWDC keynote",
    confidence: 0.69,
    time: "21m",
  },
];

export const agents = [
  { id: "research", name: "Research Agent", status: "active", task: "Drafting NVDA Q3 model", icon: "Brain", tasksToday: 38, model: "gpt-5-pro" },
  { id: "risk", name: "Risk Agent", status: "active", task: "Re-running VaR scenarios", icon: "ShieldAlert", tasksToday: 14, model: "claude-opus-4.1" },
  { id: "news", name: "News Agent", status: "idle", task: "Watching 412 sources", icon: "Newspaper", tasksToday: 612, model: "gemini-2.5-pro" },
  { id: "portfolio", name: "Portfolio Agent", status: "active", task: "Rebalancing scenario A/B", icon: "PieChart", tasksToday: 22, model: "gpt-5-pro" },
  { id: "filings", name: "Filings Agent", status: "active", task: "Indexed 14 10-Qs", icon: "FileText", tasksToday: 87, model: "claude-sonnet-4.5" },
  { id: "earnings", name: "Earnings Agent", status: "idle", task: "Awaiting AAPL transcript", icon: "Mic", tasksToday: 9, model: "gpt-5-pro" },
];

export const documents = [
  { id: "d1", name: "NVDA-10-Q-Q3-2025.pdf", pages: 84, status: "indexed", chunks: 612, uploaded: "2h ago" },
  { id: "d2", name: "AAPL-10-K-2025.pdf", pages: 162, status: "indexed", chunks: 1184, uploaded: "1d ago" },
  { id: "d3", name: "MSFT-Earnings-Call-Q2.pdf", pages: 24, status: "indexed", chunks: 188, uploaded: "1d ago" },
  { id: "d4", name: "TSLA-8-K-Recall.pdf", pages: 6, status: "indexing", chunks: 0, uploaded: "just now" },
  { id: "d5", name: "Fed-FOMC-Minutes-Nov.pdf", pages: 18, status: "indexed", chunks: 142, uploaded: "3d ago" },
];

export const portfolio = [
  { symbol: "NVDA", shares: 420, cost: 612.4, weight: 0.31 },
  { symbol: "MSFT", shares: 180, cost: 312.5, weight: 0.18 },
  { symbol: "AAPL", shares: 240, cost: 184.2, weight: 0.14 },
  { symbol: "GOOGL", shares: 310, cost: 142.7, weight: 0.12 },
  { symbol: "META", shares: 95, cost: 482.1, weight: 0.10 },
  { symbol: "BRK.B", shares: 60, cost: 410.0, weight: 0.08 },
  { symbol: "CASH", shares: 1, cost: 0, weight: 0.07 },
];

export const suggestedPrompts = [
  "Summarize NVDA's latest 10-Q and flag the three biggest risk factors",
  "Compare AAPL vs MSFT capex trends over the last 8 quarters",
  "What did the Fed actually say about Q1 cuts? Cite the FOMC minutes.",
  "Build a bear case for TSLA based on the latest recall 8-K",
  "Show me semis with FCF margin > 35% and net debt < 0",
];
