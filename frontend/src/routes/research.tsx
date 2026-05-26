import { createFileRoute } from "@tanstack/react-router";
import { AppShell } from "@/components/app-shell";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { useMemo, useState } from "react";
import {
  Sparkles,
  TrendingUp,
  ShieldAlert,
  Brain,
  Star,
  Bell,
} from "lucide-react";
import { priceSeries } from "@/lib/mock-data";

export const Route = createFileRoute("/research")({
  component: ResearchPage,
});

const ranges = ["1D", "5D", "1M", "6M", "YTD", "1Y", "5Y", "MAX"] as const;

const revenueByQuarter = [
  { q: "Q1'24", rev: 22.1, eps: 0.61 },
  { q: "Q2'24", rev: 26.0, eps: 0.68 },
  { q: "Q3'24", rev: 30.0, eps: 0.81 },
  { q: "Q4'24", rev: 32.5, eps: 0.89 },
  { q: "Q1'25", rev: 35.1, eps: 0.96 },
  { q: "Q2'25", rev: 36.8, eps: 1.05 },
  { q: "Q3'25", rev: 38.4, eps: 1.12 },
  { q: "Q4'25E", rev: 42.1, eps: 1.22 },
];

const segmentBreakdown = [
  { name: "Data Center", v: 30.8, color: "oklch(0.78 0.16 160)" },
  { name: "Gaming", v: 4.1, color: "oklch(0.7 0.14 200)" },
  { name: "Professional", v: 1.9, color: "oklch(0.75 0.16 80)" },
  { name: "Automotive", v: 1.6, color: "oklch(0.65 0.18 300)" },
];

const consensus = { buy: 42, hold: 6, sell: 1, pt: 1480, current: 1284 };

const risks = [
  { sev: "high", label: "Customer concentration", detail: "Top 4 hyperscalers = 46% of DC revenue" },
  { sev: "med", label: "China H20 sanctions", detail: "Capped at <3% of revenue; further restrictions possible" },
  { sev: "med", label: "Blackwell yield ramp", detail: "TSMC CoWoS-L capacity remains tight through H1'26" },
  { sev: "low", label: "Competitive networking", detail: "Cisco/Arista entering enterprise RFPs" },
];

function ResearchPage() {
  const [range, setRange] = useState<(typeof ranges)[number]>("6M");
  const points = { "1D": 78, "5D": 60, "1M": 90, "6M": 130, YTD: 110, "1Y": 220, "5Y": 260, MAX: 300 }[range];
  const data = useMemo(() => priceSeries(1284, points, 0.02, 11), [range, points]);

  return (
    <AppShell title="NVDA · NVIDIA Corp." subtitle="Semiconductors · Mega Cap · Covering analyst: A. Kim">
      <div className="p-4 lg:p-6 space-y-5">
        {/* Header bar */}
        <div className="rounded-xl border border-border bg-card/60 p-5 flex flex-wrap items-end gap-6">
          <div>
            <div className="text-xs text-muted-foreground">Last price · 16:00 EST</div>
            <div className="mt-1 flex items-baseline gap-3">
              <div className="text-3xl font-semibold font-mono">$1,284.42</div>
              <div className="text-sm text-emerald">+$28.11 · +2.24%</div>
            </div>
          </div>
          <div className="flex gap-6 text-xs">
            {[
              ["Mkt Cap", "$3.15T"], ["P/E", "62.1"], ["Fwd P/E", "34.8"],
              ["Rev TTM", "$132B"], ["FCF Margin", "48%"], ["Beta", "1.74"],
            ].map(([k, v]) => (
              <div key={k}>
                <div className="text-muted-foreground">{k}</div>
                <div className="font-mono mt-0.5">{v}</div>
              </div>
            ))}
          </div>
          <div className="ml-auto flex items-center gap-2">
            <button className="h-9 px-3 inline-flex items-center gap-1.5 rounded-md border border-border text-xs hover:border-emerald/40">
              <Bell className="h-3.5 w-3.5" /> Alert
            </button>
            <button className="h-9 px-3 inline-flex items-center gap-1.5 rounded-md border border-border text-xs hover:border-emerald/40">
              <Star className="h-3.5 w-3.5" /> Watchlist
            </button>
            <button className="h-9 px-3 inline-flex items-center gap-1.5 rounded-md bg-emerald text-black text-xs font-semibold">
              <Sparkles className="h-3.5 w-3.5" /> Ask FinSight
            </button>
          </div>
        </div>

        {/* AI summary */}
        <div className="rounded-xl border border-emerald/30 bg-gradient-to-br from-emerald/[0.06] to-transparent p-5">
          <div className="flex items-center gap-2 text-xs uppercase tracking-wider text-emerald mb-2">
            <Sparkles className="h-3.5 w-3.5" /> AI Research Summary · updated 4m ago
          </div>
          <p className="text-[15px] leading-relaxed text-foreground/95 max-w-4xl">
            NVIDIA continues to compound at category-defining rates. Q3 data-center revenue of
            <span className="text-emerald font-medium"> $30.8B (+112% YoY)</span> reflects faster-than-modeled
            Blackwell ramp and a structurally higher networking attach rate (71%). We see asymmetric upside
            to FY26 EPS (~$5.42 vs Street $5.18). Primary risks remain
            <span className="text-loss font-medium"> customer concentration</span> (top 4 hyperscalers = 46%)
            and Blackwell yield ramp at CoWoS-L.
          </p>
          <div className="mt-3 flex flex-wrap gap-1.5">
            {["Beat & raise", "Margin expansion", "Concentration risk", "Blackwell ramp"].map((t) => (
              <span key={t} className="text-[11px] rounded-full border border-border bg-card/60 px-2.5 py-0.5 text-muted-foreground">
                {t}
              </span>
            ))}
          </div>
        </div>

        {/* Chart */}
        <div className="rounded-xl border border-border bg-card/60 p-5">
          <div className="flex items-center justify-between mb-3">
            <div className="text-sm font-semibold">Price · {range}</div>
            <div className="flex gap-1">
              {ranges.map((r) => (
                <button
                  key={r}
                  onClick={() => setRange(r)}
                  className={`px-2.5 h-7 rounded-md text-[11px] font-medium transition-colors ${
                    r === range ? "bg-emerald/10 text-emerald" : "text-muted-foreground hover:text-foreground hover:bg-accent"
                  }`}
                >
                  {r}
                </button>
              ))}
            </div>
          </div>
          <div className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={data}>
                <defs>
                  <linearGradient id="r1" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="oklch(0.78 0.16 160)" stopOpacity={0.35} />
                    <stop offset="100%" stopColor="oklch(0.78 0.16 160)" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid stroke="oklch(1 0 0 / 0.05)" vertical={false} />
                <XAxis dataKey="t" hide />
                <YAxis tick={{ fill: "oklch(0.66 0.012 240)", fontSize: 10 }} axisLine={false} tickLine={false} width={40} />
                <Tooltip contentStyle={{ background: "oklch(0.2 0.007 240)", border: "1px solid oklch(1 0 0 / 0.1)", borderRadius: 8, fontSize: 12 }} />
                <Area dataKey="value" stroke="oklch(0.85 0.18 160)" strokeWidth={1.6} fill="url(#r1)" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Revenue / EPS / Consensus row */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
          <div className="rounded-xl border border-border bg-card/60 p-5">
            <div className="text-sm font-semibold mb-1">Revenue ($B)</div>
            <div className="text-xs text-muted-foreground mb-3">Quarterly · estimate dashed</div>
            <div className="h-48">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={revenueByQuarter}>
                  <CartesianGrid stroke="oklch(1 0 0 / 0.05)" vertical={false} />
                  <XAxis dataKey="q" tick={{ fill: "oklch(0.66 0.012 240)", fontSize: 10 }} axisLine={false} tickLine={false} />
                  <YAxis tick={{ fill: "oklch(0.66 0.012 240)", fontSize: 10 }} axisLine={false} tickLine={false} width={28} />
                  <Tooltip cursor={{ fill: "oklch(1 0 0 / 0.04)" }} contentStyle={{ background: "oklch(0.2 0.007 240)", border: "1px solid oklch(1 0 0 / 0.1)", borderRadius: 8, fontSize: 12 }} />
                  <Bar dataKey="rev" radius={[3, 3, 0, 0]}>
                    {revenueByQuarter.map((d, i) => (
                      <Cell key={i} fill={d.q.includes("E") ? "oklch(0.78 0.16 160 / 0.4)" : "oklch(0.78 0.16 160)"} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
          <div className="rounded-xl border border-border bg-card/60 p-5">
            <div className="text-sm font-semibold mb-1">EPS trend</div>
            <div className="text-xs text-muted-foreground mb-3">GAAP diluted · trailing</div>
            <div className="h-48">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={revenueByQuarter}>
                  <CartesianGrid stroke="oklch(1 0 0 / 0.05)" vertical={false} />
                  <XAxis dataKey="q" tick={{ fill: "oklch(0.66 0.012 240)", fontSize: 10 }} axisLine={false} tickLine={false} />
                  <YAxis tick={{ fill: "oklch(0.66 0.012 240)", fontSize: 10 }} axisLine={false} tickLine={false} width={28} />
                  <Tooltip contentStyle={{ background: "oklch(0.2 0.007 240)", border: "1px solid oklch(1 0 0 / 0.1)", borderRadius: 8, fontSize: 12 }} />
                  <Line dataKey="eps" stroke="oklch(0.85 0.18 160)" strokeWidth={2} dot={{ fill: "oklch(0.85 0.18 160)", r: 3 }} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
          <div className="rounded-xl border border-border bg-card/60 p-5">
            <div className="text-sm font-semibold mb-1">Analyst consensus</div>
            <div className="text-xs text-muted-foreground mb-3">49 sell-side analysts</div>
            <div className="flex items-center gap-3 mb-3">
              <div className="flex-1 h-2 rounded-full overflow-hidden flex">
                <div className="bg-emerald" style={{ width: `${(consensus.buy / 49) * 100}%` }} />
                <div className="bg-muted-foreground/40" style={{ width: `${(consensus.hold / 49) * 100}%` }} />
                <div className="bg-loss" style={{ width: `${(consensus.sell / 49) * 100}%` }} />
              </div>
            </div>
            <div className="grid grid-cols-3 gap-2 text-xs">
              <div><div className="text-emerald font-mono text-lg">{consensus.buy}</div><div className="text-muted-foreground">Buy</div></div>
              <div><div className="font-mono text-lg">{consensus.hold}</div><div className="text-muted-foreground">Hold</div></div>
              <div><div className="text-loss font-mono text-lg">{consensus.sell}</div><div className="text-muted-foreground">Sell</div></div>
            </div>
            <div className="mt-4 pt-4 border-t border-border flex items-end justify-between">
              <div>
                <div className="text-xs text-muted-foreground">Avg price target</div>
                <div className="font-mono text-2xl">${consensus.pt}</div>
              </div>
              <div className="text-emerald text-sm font-medium inline-flex items-center gap-1">
                <TrendingUp className="h-3.5 w-3.5" /> +15.2% upside
              </div>
            </div>
          </div>
        </div>

        {/* Statements + risk + sentiment + earnings */}
        <div className="grid grid-cols-1 lg:grid-cols-[1.4fr_1fr] gap-5">
          <div className="rounded-xl border border-border bg-card/60 overflow-hidden">
            <div className="p-4 border-b border-border flex items-center justify-between">
              <div className="text-sm font-semibold">Financial statements · Income</div>
              <div className="text-xs text-muted-foreground">Last 4 quarters</div>
            </div>
            <table className="w-full text-xs">
              <thead className="bg-background/40 text-muted-foreground">
                <tr>
                  <th className="text-left font-medium px-4 py-2">Metric ($B)</th>
                  <th className="text-right font-medium px-4 py-2">Q4'24</th>
                  <th className="text-right font-medium px-4 py-2">Q1'25</th>
                  <th className="text-right font-medium px-4 py-2">Q2'25</th>
                  <th className="text-right font-medium px-4 py-2">Q3'25</th>
                  <th className="text-right font-medium px-4 py-2">YoY%</th>
                </tr>
              </thead>
              <tbody className="font-mono">
                {[
                  ["Revenue", "32.5", "35.1", "36.8", "38.4", "+94%", true],
                  ["Gross profit", "24.1", "26.3", "27.4", "28.8", "+96%", true],
                  ["Operating inc.", "20.0", "21.9", "22.8", "24.0", "+101%", true],
                  ["Net income", "16.6", "18.2", "18.9", "19.9", "+105%", true],
                  ["FCF", "15.5", "16.4", "17.2", "18.4", "+88%", true],
                  ["Capex", "(0.95)", "(1.10)", "(1.20)", "(1.34)", "+44%", false],
                ].map((r, i) => (
                  <tr key={i} className="border-t border-border hover:bg-accent/30">
                    <td className="px-4 py-2 font-sans text-foreground/80">{r[0]}</td>
                    <td className="px-4 py-2 text-right">{r[1]}</td>
                    <td className="px-4 py-2 text-right">{r[2]}</td>
                    <td className="px-4 py-2 text-right">{r[3]}</td>
                    <td className="px-4 py-2 text-right">{r[4]}</td>
                    <td className={`px-4 py-2 text-right ${r[6] ? "text-emerald" : "text-loss"}`}>{r[5]}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="space-y-5">
            <div className="rounded-xl border border-border bg-card/60 p-5">
              <div className="flex items-center gap-2 text-sm font-semibold mb-3">
                <ShieldAlert className="h-4 w-4 text-loss" /> Risk analysis
              </div>
              <div className="space-y-2">
                {risks.map((r) => (
                  <div key={r.label} className="flex gap-3 p-2 rounded hover:bg-accent/30">
                    <span className={`mt-1 h-2 w-2 rounded-full shrink-0 ${
                      r.sev === "high" ? "bg-loss" : r.sev === "med" ? "bg-amber-400" : "bg-muted-foreground"
                    }`} />
                    <div className="text-xs">
                      <div className="font-medium">{r.label}</div>
                      <div className="text-muted-foreground mt-0.5">{r.detail}</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div className="rounded-xl border border-border bg-card/60 p-5">
              <div className="flex items-center gap-2 text-sm font-semibold mb-3">
                <Brain className="h-4 w-4 text-emerald" /> Sentiment · 24h
              </div>
              <div className="flex items-end gap-2 h-24">
                {[0.4, 0.6, 0.55, 0.7, 0.65, 0.8, 0.72, 0.85, 0.9, 0.88, 0.94, 0.91].map((v, i) => (
                  <div key={i} className="flex-1 rounded-t" style={{ height: `${v * 100}%`, background: "oklch(0.78 0.16 160)", opacity: 0.5 + v * 0.5 }} />
                ))}
              </div>
              <div className="mt-3 text-xs text-muted-foreground flex justify-between">
                <span>Net sentiment <span className="text-emerald font-mono ml-1">+0.74</span></span>
                <span>4,218 mentions</span>
              </div>
            </div>
          </div>
        </div>

        {/* Earnings insights */}
        <div className="rounded-xl border border-border bg-card/60 p-5">
          <div className="text-sm font-semibold mb-3">Earnings insights · Q3'25 call</div>
          <div className="grid lg:grid-cols-3 gap-4 text-sm">
            {[
              { tag: "Bull", color: "text-emerald", border: "border-emerald/30", q: "“Blackwell production is now exceeding our internal expectations across every tier.”", who: "Jensen Huang, CEO" },
              { tag: "Watch", color: "text-amber-400", border: "border-amber-400/30", q: "“We continue to monitor Hopper-to-Blackwell transition pacing in mid-market.”", who: "Colette Kress, CFO" },
              { tag: "Bear", color: "text-loss", border: "border-loss/30", q: "“Customer concentration in data-center remains elevated this quarter.”", who: "10-Q Risk Factors" },
            ].map((c) => (
              <div key={c.tag} className={`rounded-lg border ${c.border} bg-background/40 p-4`}>
                <div className={`text-[10px] uppercase tracking-widest ${c.color} mb-2`}>{c.tag}</div>
                <p className="text-foreground/90 leading-relaxed">{c.q}</p>
                <div className="text-xs text-muted-foreground mt-3">— {c.who}</div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </AppShell>
  );
}
