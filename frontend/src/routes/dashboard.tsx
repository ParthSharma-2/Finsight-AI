import { createFileRoute, Link } from "@tanstack/react-router";
import { AppShell } from "@/components/app-shell";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { useMemo } from "react";
import { motion } from "framer-motion";
import {
  ArrowDown,
  ArrowUp,
  Brain,
  ShieldAlert,
  Newspaper,
  Sparkles,
  TrendingUp,
  TrendingDown,
  Plus,
  ChevronRight,
} from "lucide-react";
import {
  insights,
  marketIndexes,
  news,
  portfolio,
  priceSeries,
  trending,
  watchlist,
} from "@/lib/mock-data";

export const Route = createFileRoute("/dashboard")({
  component: Dashboard,
});

function StatCard({
  label,
  value,
  delta,
  series,
}: {
  label: string;
  value: string;
  delta: number;
  series: { i: number; value: number }[];
}) {
  const up = delta >= 0;
  return (
    <div className="rounded-xl border border-border bg-card/60 p-4 hover:border-emerald/30 transition-colors">
      <div className="flex items-start justify-between">
        <div>
          <div className="text-[11px] uppercase tracking-wider text-muted-foreground">{label}</div>
          <div className="mt-1.5 text-xl font-semibold tracking-tight font-mono">{value}</div>
        </div>
        <div className={`flex items-center gap-0.5 text-xs font-medium ${up ? "text-emerald" : "text-loss"}`}>
          {up ? <ArrowUp className="h-3 w-3" /> : <ArrowDown className="h-3 w-3" />}
          {Math.abs(delta).toFixed(2)}%
        </div>
      </div>
      <div className="h-12 mt-2 -mx-1">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={series}>
            <defs>
              <linearGradient id={`spark-${label}`} x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor={up ? "oklch(0.78 0.16 160)" : "oklch(0.66 0.22 22)"} stopOpacity={0.5} />
                <stop offset="100%" stopColor={up ? "oklch(0.78 0.16 160)" : "oklch(0.66 0.22 22)"} stopOpacity={0} />
              </linearGradient>
            </defs>
            <Area dataKey="value" stroke={up ? "oklch(0.85 0.18 160)" : "oklch(0.7 0.21 22)"} strokeWidth={1.5} fill={`url(#spark-${label})`} />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

function MarketOverview() {
  const cards = useMemo(
    () =>
      marketIndexes.slice(0, 4).map((m, i) => ({
        label: m.name,
        value: m.value.toLocaleString(undefined, { maximumFractionDigits: 2 }),
        delta: m.change,
        series: priceSeries(m.value, 40, 0.01, 5 + i),
      })),
    [],
  );
  return (
    <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
      {cards.map((c) => (
        <StatCard key={c.label} {...c} />
      ))}
    </div>
  );
}

function PortfolioPanel() {
  const colors = [
    "oklch(0.78 0.16 160)",
    "oklch(0.7 0.14 200)",
    "oklch(0.75 0.16 80)",
    "oklch(0.7 0.2 30)",
    "oklch(0.65 0.18 300)",
    "oklch(0.6 0.15 240)",
    "oklch(0.45 0.01 240)",
  ];
  const pie = portfolio.map((p, i) => ({ name: p.symbol, value: p.weight, fill: colors[i] }));
  const equity = useMemo(() => priceSeries(1, 60, 0.014, 17).map((d) => ({ ...d, value: 1_000_000 * d.value })), []);
  return (
    <div className="rounded-xl border border-border bg-card/60 p-5">
      <div className="flex items-center justify-between">
        <div>
          <div className="text-xs text-muted-foreground">Portfolio · Alpha-1</div>
          <div className="mt-1 flex items-baseline gap-3">
            <div className="text-2xl font-semibold font-mono tracking-tight">$8,412,330</div>
            <div className="text-sm text-emerald font-medium">+$112,415 · +1.35%</div>
          </div>
        </div>
        <Link to="/research" className="text-xs text-muted-foreground hover:text-foreground inline-flex items-center gap-1">
          View detail <ChevronRight className="h-3.5 w-3.5" />
        </Link>
      </div>
      <div className="mt-4 grid lg:grid-cols-[1.4fr_1fr] gap-5">
        <div className="h-44">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={equity} margin={{ top: 5, right: 0, left: 0, bottom: 0 }}>
              <defs>
                <linearGradient id="portFill" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="oklch(0.78 0.16 160)" stopOpacity={0.4} />
                  <stop offset="100%" stopColor="oklch(0.78 0.16 160)" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid stroke="oklch(1 0 0 / 0.05)" vertical={false} />
              <XAxis dataKey="t" hide />
              <YAxis hide domain={["dataMin", "dataMax"]} />
              <Tooltip
                contentStyle={{ background: "oklch(0.2 0.007 240)", border: "1px solid oklch(1 0 0 / 0.1)", borderRadius: 8, fontSize: 12 }}
                formatter={(v: number) => "$" + v.toLocaleString(undefined, { maximumFractionDigits: 0 })}
              />
              <Area dataKey="value" stroke="oklch(0.85 0.18 160)" strokeWidth={1.5} fill="url(#portFill)" />
            </AreaChart>
          </ResponsiveContainer>
        </div>
        <div className="flex items-center gap-3">
          <div className="h-44 w-44">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie data={pie} dataKey="value" innerRadius={45} outerRadius={70} paddingAngle={2} stroke="oklch(0.16 0.005 240)">
                  {pie.map((p, i) => <Cell key={i} fill={p.fill} />)}
                </Pie>
                <Tooltip
                  contentStyle={{ background: "oklch(0.2 0.007 240)", border: "1px solid oklch(1 0 0 / 0.1)", borderRadius: 8, fontSize: 12 }}
                  formatter={(v: number) => (v * 100).toFixed(0) + "%"}
                />
              </PieChart>
            </ResponsiveContainer>
          </div>
          <div className="flex-1 space-y-1.5 text-xs">
            {pie.map((p) => (
              <div key={p.name} className="flex items-center gap-2">
                <span className="h-2 w-2 rounded-sm" style={{ background: p.fill }} />
                <span className="text-muted-foreground">{p.name}</span>
                <span className="ml-auto font-mono">{(p.value * 100).toFixed(0)}%</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

function Watchlist() {
  return (
    <div className="rounded-xl border border-border bg-card/60">
      <div className="flex items-center justify-between p-4 border-b border-border">
        <div>
          <div className="text-sm font-semibold">Watchlist</div>
          <div className="text-xs text-muted-foreground">8 instruments · live</div>
        </div>
        <button className="text-xs text-muted-foreground hover:text-foreground inline-flex items-center gap-1">
          <Plus className="h-3.5 w-3.5" /> Add ticker
        </button>
      </div>
      <div className="divide-y divide-border">
        {watchlist.map((s) => {
          const up = s.change >= 0;
          const series = priceSeries(s.price, 30, 0.013, s.symbol.charCodeAt(0));
          return (
            <Link
              to="/research"
              key={s.symbol}
              className="grid grid-cols-[1fr_80px_90px] items-center gap-3 px-4 py-3 hover:bg-accent/40 transition-colors"
            >
              <div className="min-w-0">
                <div className="flex items-center gap-2">
                  <span className="font-mono text-sm font-semibold">{s.symbol}</span>
                  <span className="text-[10px] uppercase tracking-wider text-muted-foreground">{s.sector}</span>
                </div>
                <div className="text-xs text-muted-foreground truncate">{s.name}</div>
              </div>
              <div className="h-9">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={series}>
                    <Area dataKey="value" stroke={up ? "oklch(0.85 0.18 160)" : "oklch(0.7 0.21 22)"} strokeWidth={1.4} fill="transparent" />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
              <div className="text-right">
                <div className="font-mono text-sm">${s.price.toFixed(2)}</div>
                <div className={`text-[11px] font-medium ${up ? "text-emerald" : "text-loss"}`}>
                  {up ? "+" : ""}{s.changePct.toFixed(2)}%
                </div>
              </div>
            </Link>
          );
        })}
      </div>
    </div>
  );
}

function InsightsPanel() {
  const iconMap: Record<string, typeof Brain> = {
    "Research Agent": Brain,
    "Risk Agent": ShieldAlert,
    "News Agent": Newspaper,
  };
  return (
    <div className="rounded-xl border border-border bg-card/60">
      <div className="flex items-center justify-between p-4 border-b border-border">
        <div className="flex items-center gap-2">
          <Sparkles className="h-3.5 w-3.5 text-emerald" />
          <div className="text-sm font-semibold">AI-generated insights</div>
        </div>
        <span className="text-[10px] uppercase tracking-wider text-emerald">streaming</span>
      </div>
      <div className="p-2 space-y-1">
        {insights.map((i, idx) => {
          const Icon = iconMap[i.agent] || Sparkles;
          return (
            <motion.div
              key={i.id}
              initial={{ opacity: 0, y: 6 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: idx * 0.08 }}
              className="rounded-lg p-3 hover:bg-accent/40 cursor-pointer"
            >
              <div className="flex items-center gap-2 text-[11px] text-muted-foreground">
                <Icon className="h-3.5 w-3.5 text-emerald" />
                <span>{i.agent}</span>
                <span className="ml-auto">{i.time}</span>
              </div>
              <p className="mt-1.5 text-sm leading-snug text-foreground/90">{i.title}</p>
              <div className="mt-2 flex items-center gap-2">
                <div className="h-1 flex-1 rounded-full bg-muted overflow-hidden">
                  <div className="h-full bg-emerald" style={{ width: `${i.confidence * 100}%` }} />
                </div>
                <span className="text-[10px] font-mono text-muted-foreground">
                  {(i.confidence * 100).toFixed(0)}% conf
                </span>
              </div>
            </motion.div>
          );
        })}
      </div>
    </div>
  );
}

function NewsFeed() {
  return (
    <div className="rounded-xl border border-border bg-card/60">
      <div className="p-4 border-b border-border flex items-center justify-between">
        <div className="text-sm font-semibold">Live market news</div>
        <span className="text-[10px] uppercase tracking-wider text-muted-foreground">412 sources</span>
      </div>
      <div className="divide-y divide-border">
        {news.map((n) => (
          <div key={n.id} className="p-4 hover:bg-accent/30 cursor-pointer">
            <div className="flex items-center gap-2 text-[11px] text-muted-foreground">
              <span className="font-medium text-foreground/80">{n.source}</span>
              <span>·</span>
              <span>{n.time}</span>
              <span className="ml-auto inline-flex items-center gap-1">
                <span
                  className={`h-1.5 w-1.5 rounded-full ${
                    n.sentiment === "bullish"
                      ? "bg-emerald"
                      : n.sentiment === "bearish"
                        ? "bg-loss"
                        : "bg-muted-foreground"
                  }`}
                />
                {n.tag}
              </span>
            </div>
            <p className="mt-1.5 text-sm leading-snug text-foreground/95">{n.title}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

function TrendingPanel() {
  return (
    <div className="rounded-xl border border-border bg-card/60 p-4">
      <div className="flex items-center justify-between mb-3">
        <div className="text-sm font-semibold">Trending now</div>
        <span className="text-[10px] uppercase tracking-wider text-muted-foreground">last 5m</span>
      </div>
      <div className="space-y-2">
        {trending.map((t) => {
          const up = t.change >= 0;
          return (
            <div key={t.symbol} className="flex items-center gap-3">
              <div className={`h-7 w-7 rounded grid place-items-center text-[10px] font-semibold ${up ? "bg-emerald/10 text-emerald" : "bg-loss/10 text-loss"}`}>
                {up ? <TrendingUp className="h-3.5 w-3.5" /> : <TrendingDown className="h-3.5 w-3.5" />}
              </div>
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2">
                  <span className="font-mono text-sm">{t.symbol}</span>
                  <span className="text-[10px] text-muted-foreground truncate">{t.name}</span>
                </div>
              </div>
              <div className="text-right">
                <div className="font-mono text-xs">${t.price.toFixed(2)}</div>
                <div className={`text-[10px] ${up ? "text-emerald" : "text-loss"}`}>
                  {up ? "+" : ""}{t.changePct.toFixed(2)}%
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function AgentActivity() {
  const data = [
    { t: "9", calls: 12 },
    { t: "10", calls: 28 },
    { t: "11", calls: 22 },
    { t: "12", calls: 41 },
    { t: "13", calls: 35 },
    { t: "14", calls: 52 },
    { t: "15", calls: 38 },
    { t: "16", calls: 44 },
  ];
  return (
    <div className="rounded-xl border border-border bg-card/60 p-4">
      <div className="flex items-center justify-between">
        <div>
          <div className="text-sm font-semibold">Agent activity</div>
          <div className="text-xs text-muted-foreground">782 tool calls today</div>
        </div>
        <Link to="/agents" className="text-xs text-muted-foreground hover:text-foreground inline-flex items-center gap-1">
          Open <ChevronRight className="h-3.5 w-3.5" />
        </Link>
      </div>
      <div className="h-28 mt-3 -mx-2">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data}>
            <XAxis dataKey="t" axisLine={false} tickLine={false} tick={{ fill: "oklch(0.66 0.012 240)", fontSize: 10 }} />
            <Tooltip cursor={{ fill: "oklch(1 0 0 / 0.04)" }} contentStyle={{ background: "oklch(0.2 0.007 240)", border: "1px solid oklch(1 0 0 / 0.1)", borderRadius: 8, fontSize: 12 }} />
            <Bar dataKey="calls" fill="oklch(0.78 0.16 160)" radius={[3, 3, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

function Dashboard() {
  return (
    <AppShell title="Workspace" subtitle="Wednesday, May 20 · markets open">
      <div className="p-4 lg:p-6 space-y-5">
        <MarketOverview />
        <div className="grid grid-cols-1 xl:grid-cols-[2fr_1fr] gap-5">
          <div className="space-y-5">
            <PortfolioPanel />
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
              <Watchlist />
              <NewsFeed />
            </div>
          </div>
          <div className="space-y-5">
            <InsightsPanel />
            <TrendingPanel />
            <AgentActivity />
          </div>
        </div>
      </div>
    </AppShell>
  );
}
