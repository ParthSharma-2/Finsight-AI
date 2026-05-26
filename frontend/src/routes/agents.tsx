import { createFileRoute } from "@tanstack/react-router";
import { AppShell } from "@/components/app-shell";
import {
  Brain,
  ShieldAlert,
  Newspaper,
  PieChart,
  FileText,
  Mic,
  Play,
  Pause,
  Plus,
  CheckCircle2,
  Loader2,
} from "lucide-react";
import { agents } from "@/lib/mock-data";

export const Route = createFileRoute("/agents")({
  component: AgentsPage,
});

const iconMap: Record<string, typeof Brain> = {
  Brain, ShieldAlert, Newspaper, PieChart, FileText, Mic,
};

const timeline = [
  { t: "16:02", agent: "Research Agent", action: "Composed NVDA Q3 memo · 1,284 tokens", status: "done" },
  { t: "16:01", agent: "Filings Agent", action: "Retrieved 8 chunks across 3 documents", status: "done" },
  { t: "15:58", agent: "Risk Agent", action: "Ran VaR · 99% / 1d = $284k", status: "done" },
  { t: "15:55", agent: "News Agent", action: "Flagged Bloomberg headline on Apple/Anthropic", status: "done" },
  { t: "15:52", agent: "Portfolio Agent", action: "Computing rebalance scenario B…", status: "running" },
  { t: "15:48", agent: "Earnings Agent", action: "Waiting on AAPL transcript publication", status: "idle" },
];

function AgentsPage() {
  return (
    <AppShell title="AI Agents" subtitle="6 autonomous agents · orchestrated via FinSight Conductor">
      <div className="p-4 lg:p-6 space-y-5">
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {agents.map((a) => {
            const Icon = iconMap[a.icon] ?? Brain;
            const active = a.status === "active";
            return (
              <div key={a.id} className="rounded-xl border border-border bg-card/60 p-5 hover:border-emerald/30 transition-colors">
                <div className="flex items-center justify-between">
                  <div className="h-10 w-10 rounded-lg grid place-items-center bg-emerald/10 border border-emerald/20 text-emerald">
                    <Icon className="h-5 w-5" />
                  </div>
                  <span className={`inline-flex items-center gap-1.5 text-[10px] uppercase tracking-wider ${active ? "text-emerald" : "text-muted-foreground"}`}>
                    <span className={`h-1.5 w-1.5 rounded-full ${active ? "bg-emerald animate-pulse" : "bg-muted-foreground"}`} />
                    {a.status}
                  </span>
                </div>
                <h3 className="mt-4 text-sm font-semibold">{a.name}</h3>
                <p className="text-xs text-muted-foreground mt-1">{a.task}</p>
                <div className="mt-4 flex items-center justify-between text-[11px] text-muted-foreground">
                  <span className="font-mono">{a.model}</span>
                  <span>{a.tasksToday} tasks today</span>
                </div>
                <div className="mt-3 flex gap-2">
                  <button className="h-8 flex-1 rounded-md border border-border text-xs hover:border-emerald/40 inline-flex items-center justify-center gap-1.5">
                    {active ? <><Pause className="h-3.5 w-3.5" /> Pause</> : <><Play className="h-3.5 w-3.5" /> Run</>}
                  </button>
                  <button className="h-8 px-3 rounded-md border border-border text-xs hover:border-emerald/40">Configure</button>
                </div>
              </div>
            );
          })}
          <button className="rounded-xl border-2 border-dashed border-border bg-card/20 p-5 text-center hover:border-emerald/40 transition-colors">
            <Plus className="h-5 w-5 mx-auto text-emerald" />
            <div className="mt-2 text-sm font-medium">Create custom agent</div>
            <div className="text-xs text-muted-foreground mt-1">Define tools, model, prompts</div>
          </button>
        </div>

        {/* Orchestration */}
        <div className="grid grid-cols-1 lg:grid-cols-[1.2fr_1fr] gap-5">
          <div className="rounded-xl border border-border bg-card/60 p-5">
            <div className="text-sm font-semibold mb-1">Orchestration workflow · NVDA earnings memo</div>
            <div className="text-xs text-muted-foreground mb-4">Conductor v3 · DAG execution</div>
            <div className="rounded-lg border border-border bg-background/40 p-4 overflow-x-auto">
              <div className="grid grid-cols-5 gap-3 min-w-[640px]">
                {[
                  { n: "Plan", a: "Conductor", s: "done" },
                  { n: "Retrieve", a: "Filings", s: "done" },
                  { n: "Analyze", a: "Research", s: "done" },
                  { n: "Risk", a: "Risk", s: "running" },
                  { n: "Synthesize", a: "Conductor", s: "idle" },
                ].map((n, i, arr) => (
                  <div key={n.n} className="relative">
                    <div className={`rounded-lg border p-3 ${
                      n.s === "done" ? "border-emerald/30 bg-emerald/5"
                      : n.s === "running" ? "border-amber-400/40 bg-amber-400/5"
                      : "border-border bg-card/40"
                    }`}>
                      <div className="flex items-center gap-1.5 text-[10px] uppercase tracking-wider">
                        {n.s === "done" ? <CheckCircle2 className="h-3 w-3 text-emerald" />
                          : n.s === "running" ? <Loader2 className="h-3 w-3 text-amber-400 animate-spin" />
                          : <div className="h-3 w-3 rounded-full border border-muted-foreground" />}
                        <span className={n.s === "done" ? "text-emerald" : n.s === "running" ? "text-amber-400" : "text-muted-foreground"}>{n.s}</span>
                      </div>
                      <div className="mt-1.5 text-sm font-semibold">{n.n}</div>
                      <div className="text-[11px] text-muted-foreground">{n.a}</div>
                    </div>
                    {i < arr.length - 1 && (
                      <div className="hidden lg:block absolute top-1/2 -right-2 h-px w-3 bg-border" />
                    )}
                  </div>
                ))}
              </div>
            </div>
          </div>

          <div className="rounded-xl border border-border bg-card/60 p-5">
            <div className="text-sm font-semibold mb-3">Task execution timeline</div>
            <div className="space-y-3">
              {timeline.map((t, i) => (
                <div key={i} className="flex gap-3">
                  <div className="text-[11px] text-muted-foreground font-mono w-12 shrink-0 pt-0.5">{t.t}</div>
                  <div className="relative flex-1 pl-4 border-l border-border">
                    <span className={`absolute -left-1 top-1.5 h-2 w-2 rounded-full ${
                      t.status === "done" ? "bg-emerald"
                      : t.status === "running" ? "bg-amber-400 animate-pulse"
                      : "bg-muted-foreground"
                    }`} />
                    <div className="text-xs font-medium">{t.agent}</div>
                    <div className="text-xs text-muted-foreground">{t.action}</div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </AppShell>
  );
}
