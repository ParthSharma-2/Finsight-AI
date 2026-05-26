import { createFileRoute, Link } from "@tanstack/react-router";
import { motion } from "framer-motion";
import {
  Area,
  AreaChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  ArrowRight,
  Sparkles,
  Brain,
  ShieldCheck,
  Zap,
  FileSearch,
  LineChart as LineIcon,
  Bot,
  Check,
  Quote,
  Send,
} from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { Logo } from "@/components/app-shell";
import { priceSeries, marketIndexes, suggestedPrompts } from "@/lib/mock-data";

export const Route = createFileRoute("/")({
  component: Landing,
});

function Nav() {
  return (
    <header className="sticky top-0 z-40 border-b border-border/70 bg-background/70 backdrop-blur-xl">
      <div className="mx-auto max-w-7xl px-6 h-16 flex items-center gap-8">
        <Logo />
        <nav className="hidden md:flex items-center gap-7 text-sm text-muted-foreground">
          <a href="#platform" className="hover:text-foreground transition-colors">Platform</a>
          <a href="#agents" className="hover:text-foreground transition-colors">Agents</a>
          <a href="#pricing" className="hover:text-foreground transition-colors">Pricing</a>
          <a href="#customers" className="hover:text-foreground transition-colors">Customers</a>
          <a href="#" className="hover:text-foreground transition-colors">Docs</a>
        </nav>
        <div className="ml-auto flex items-center gap-2">
          <Link to="/dashboard" className="hidden sm:inline text-sm text-muted-foreground hover:text-foreground px-3 py-2">
            Sign in
          </Link>
          <Link
            to="/dashboard"
            className="inline-flex items-center gap-1.5 rounded-md bg-emerald px-3.5 h-9 text-sm font-medium text-black hover:brightness-110 transition"
          >
            Launch terminal <ArrowRight className="h-3.5 w-3.5" />
          </Link>
        </div>
      </div>
    </header>
  );
}

function Ticker() {
  return (
    <div className="border-y border-border bg-card/40 overflow-hidden ticker-mask">
      <div className="flex gap-10 py-2.5 animate-[scroll_40s_linear_infinite] whitespace-nowrap">
        {[...marketIndexes, ...marketIndexes, ...marketIndexes].map((m, i) => (
          <div key={i} className="flex items-center gap-2 text-xs font-mono">
            <span className="text-muted-foreground">{m.name}</span>
            <span className="text-foreground">{m.value.toLocaleString()}</span>
            <span className={m.change >= 0 ? "text-emerald" : "text-loss"}>
              {m.change >= 0 ? "+" : ""}{m.change.toFixed(2)}%
            </span>
          </div>
        ))}
      </div>
      <style>{`@keyframes scroll{from{transform:translateX(0)}to{transform:translateX(-33.33%)}}`}</style>
    </div>
  );
}

function HeroChart() {
  const data = useMemo(() => priceSeries(420, 80, 0.018, 13), []);
  return (
    <ResponsiveContainer width="100%" height="100%">
      <AreaChart data={data} margin={{ top: 10, right: 0, left: 0, bottom: 0 }}>
        <defs>
          <linearGradient id="heroFill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="oklch(0.78 0.16 160)" stopOpacity={0.5} />
            <stop offset="100%" stopColor="oklch(0.78 0.16 160)" stopOpacity={0} />
          </linearGradient>
        </defs>
        <XAxis dataKey="t" hide />
        <YAxis hide domain={["dataMin - 5", "dataMax + 5"]} />
        <Tooltip
          contentStyle={{ background: "oklch(0.2 0.007 240)", border: "1px solid oklch(1 0 0 / 0.1)", borderRadius: 8, fontSize: 12 }}
          labelStyle={{ color: "oklch(0.66 0.012 240)" }}
        />
        <Area
          type="monotone"
          dataKey="value"
          stroke="oklch(0.85 0.18 160)"
          strokeWidth={1.5}
          fill="url(#heroFill)"
        />
      </AreaChart>
    </ResponsiveContainer>
  );
}

function AIDemoCard() {
  const [q, setQ] = useState("");
  const [streamed, setStreamed] = useState("");
  const [running, setRunning] = useState(false);
  const demoAnswer =
    "NVDA's Q3 revenue beat consensus by 5.4%, driven by Blackwell data-center growth (+112% YoY). Gross margin expanded 220bps to 75.1% on premium-tier mix. Forward guidance implies ~$42B Q4, ~6% above Street. Key risk: Hopper-to-Blackwell transition timing in mid-market hyperscalers.\n\nSources: NVDA-10-Q-Q3-2025.pdf (p.12, p.31), Earnings-Call-Transcript-Q3.pdf";
  function run() {
    if (running) return;
    setStreamed("");
    setRunning(true);
    let i = 0;
    const id = setInterval(() => {
      i += 4;
      setStreamed(demoAnswer.slice(0, i));
      if (i >= demoAnswer.length) {
        clearInterval(id);
        setRunning(false);
      }
    }, 18);
  }
  return (
    <div className="glass rounded-2xl p-4 shadow-2xl">
      <div className="flex items-center gap-2 text-xs text-muted-foreground mb-3">
        <Sparkles className="h-3.5 w-3.5 text-emerald" />
        FinSight Research Agent · gpt-5-pro · RAG over 412 filings
        <span className="ml-auto inline-flex items-center gap-1 text-emerald">
          <span className="h-1.5 w-1.5 rounded-full bg-emerald animate-pulse" /> live
        </span>
      </div>
      <div className="flex items-center gap-2 mb-3">
        <input
          value={q}
          onChange={(e) => setQ(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && run()}
          placeholder="Ask: 'Summarize NVDA Q3 earnings and flag top risks'"
          className="flex-1 h-10 rounded-md bg-background/60 border border-border px-3 text-sm focus:outline-none focus:ring-2 focus:ring-emerald/50"
        />
        <button
          onClick={run}
          className="h-10 px-3.5 rounded-md bg-emerald text-black text-sm font-medium hover:brightness-110 inline-flex items-center gap-1.5"
        >
          <Send className="h-3.5 w-3.5" /> Run
        </button>
      </div>
      <div className="rounded-xl bg-background/60 border border-border p-4 min-h-[180px] text-sm leading-relaxed text-foreground/90 whitespace-pre-wrap">
        {streamed ? (
          <>
            {streamed}
            {running && <span className="inline-block w-1.5 h-4 align-middle bg-emerald animate-pulse ml-0.5" />}
          </>
        ) : (
          <span className="text-muted-foreground">
            Try one of the suggestions below to see streaming RAG over SEC filings, earnings transcripts and macro reports.
          </span>
        )}
      </div>
      <div className="mt-3 flex flex-wrap gap-1.5">
        {suggestedPrompts.slice(0, 3).map((p) => (
          <button
            key={p}
            onClick={() => { setQ(p); setTimeout(run, 50); }}
            className="text-[11px] px-2.5 py-1 rounded-full border border-border bg-card/60 text-muted-foreground hover:text-foreground hover:border-emerald/40 transition"
          >
            {p}
          </button>
        ))}
      </div>
    </div>
  );
}

function Hero() {
  return (
    <section className="relative overflow-hidden">
      <div className="absolute inset-0 grid-bg opacity-40" />
      <div className="absolute -top-40 left-1/2 -translate-x-1/2 h-[480px] w-[900px] rounded-full bg-emerald/20 blur-[140px] pointer-events-none" />
      <div className="relative mx-auto max-w-7xl px-6 pt-20 pb-20 lg:pt-28 lg:pb-28 grid lg:grid-cols-[1.05fr_1fr] gap-14 items-center">
        <div>
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="inline-flex items-center gap-2 rounded-full border border-border bg-card/60 px-3 py-1 text-xs text-muted-foreground"
          >
            <span className="h-1.5 w-1.5 rounded-full bg-emerald animate-pulse" />
            New: Multi-agent earnings analyst · live
          </motion.div>
          <motion.h1
            initial={{ opacity: 0, y: 14 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.05 }}
            className="mt-5 text-[44px] sm:text-[56px] lg:text-[64px] leading-[1.02] font-semibold tracking-tight text-balance"
          >
            The AI equity research
            <br />
            <span className="bg-gradient-to-r from-foreground via-emerald to-foreground/80 bg-clip-text text-transparent">
              terminal for institutions.
            </span>
          </motion.h1>
          <motion.p
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.15 }}
            className="mt-6 max-w-xl text-[15px] leading-relaxed text-muted-foreground"
          >
            FinSight AI fuses RAG over SEC filings, autonomous research agents,
            real-time market intelligence and portfolio analytics — in a single
            command-driven workspace built for analysts, PMs and quants.
          </motion.p>
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.25 }}
            className="mt-8 flex flex-wrap items-center gap-3"
          >
            <Link
              to="/dashboard"
              className="inline-flex items-center gap-1.5 h-11 px-5 rounded-md bg-emerald text-black text-sm font-semibold hover:brightness-110 glow-emerald"
            >
              Open the terminal <ArrowRight className="h-4 w-4" />
            </Link>
            <Link
              to="/chat"
              className="inline-flex items-center gap-1.5 h-11 px-5 rounded-md border border-border bg-card/60 text-sm font-medium hover:border-emerald/40"
            >
              Try the AI chat
            </Link>
          </motion.div>
          <div className="mt-10 flex items-center gap-6 text-xs text-muted-foreground">
            <div className="flex items-center gap-2"><Check className="h-3.5 w-3.5 text-emerald" /> SOC 2 Type II</div>
            <div className="flex items-center gap-2"><Check className="h-3.5 w-3.5 text-emerald" /> 99.99% SLA</div>
            <div className="flex items-center gap-2"><Check className="h-3.5 w-3.5 text-emerald" /> SSO / SAML</div>
          </div>
        </div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="relative"
        >
          <div className="absolute -inset-6 bg-emerald/10 blur-3xl rounded-3xl" />
          <div className="relative space-y-4">
            <div className="glass rounded-2xl p-4">
              <div className="flex items-center justify-between text-xs text-muted-foreground mb-2">
                <div className="flex items-center gap-2">
                  <span className="font-mono text-foreground">NVDA</span>
                  <span>NVIDIA Corp.</span>
                </div>
                <div className="flex items-center gap-2 font-mono">
                  <span className="text-foreground">$1,284.42</span>
                  <span className="text-emerald">+2.24%</span>
                </div>
              </div>
              <div className="h-44">
                <HeroChart />
              </div>
            </div>
            <AIDemoCard />
          </div>
        </motion.div>
      </div>
    </section>
  );
}

const features = [
  {
    icon: Brain,
    title: "Multi-agent research",
    desc: "Orchestrate Research, Risk, News, Earnings & Portfolio agents that reason across every filing you own.",
  },
  {
    icon: FileSearch,
    title: "RAG over SEC filings",
    desc: "Drop 10-Ks, 10-Qs, 8-Ks. We chunk, embed and retrieve with citations down to the page and paragraph.",
  },
  {
    icon: LineIcon,
    title: "Institutional analytics",
    desc: "Revenue/EPS decomposition, factor exposures, sentiment, VaR & scenario analysis out-of-the-box.",
  },
  {
    icon: Zap,
    title: "Real-time market layer",
    desc: "Sub-second streaming quotes, news and macro signals piped into every agent and chart.",
  },
  {
    icon: ShieldCheck,
    title: "Audit-grade citations",
    desc: "Every claim hyperlinks back to the exact source span. Built for compliance and reproducibility.",
  },
  {
    icon: Bot,
    title: "Bring your own LLM",
    desc: "GPT-5, Claude Opus 4.1, Gemini 2.5 Pro and private endpoints — routed per task automatically.",
  },
];

function Features() {
  return (
    <section id="platform" className="relative py-24 border-t border-border">
      <div className="mx-auto max-w-7xl px-6">
        <div className="max-w-2xl">
          <div className="text-xs uppercase tracking-[0.2em] text-emerald">The platform</div>
          <h2 className="mt-3 text-3xl sm:text-4xl font-semibold tracking-tight text-balance">
            Everything a research team needs, unified.
          </h2>
          <p className="mt-4 text-muted-foreground">
            Stop stitching Bloomberg, Excel and ChatGPT. FinSight is one workspace
            for filings, models, agents and decisions.
          </p>
        </div>
        <div className="mt-12 grid sm:grid-cols-2 lg:grid-cols-3 gap-px bg-border rounded-2xl overflow-hidden border border-border">
          {features.map((f) => {
            const Icon = f.icon;
            return (
              <div key={f.title} className="bg-background p-7 hover:bg-card/60 transition-colors group">
                <div className="h-9 w-9 rounded-lg grid place-items-center bg-emerald/10 text-emerald border border-emerald/20 group-hover:scale-105 transition-transform">
                  <Icon className="h-4.5 w-4.5" />
                </div>
                <h3 className="mt-4 text-[15px] font-semibold">{f.title}</h3>
                <p className="mt-1.5 text-sm text-muted-foreground leading-relaxed">{f.desc}</p>
              </div>
            );
          })}
        </div>
      </div>
    </section>
  );
}

function AgentsShowcase() {
  const agentList = [
    { name: "Research Agent", desc: "Builds the model. Reads filings, writes the memo.", color: "from-emerald-400/70 to-emerald-700/40" },
    { name: "Risk Agent", desc: "Monitors VaR, beta drift, concentration & tail risk.", color: "from-amber-400/70 to-amber-700/30" },
    { name: "News Agent", desc: "Watches 8,400 sources, flags moving narratives.", color: "from-sky-400/70 to-sky-700/30" },
    { name: "Portfolio Agent", desc: "Runs rebalancing scenarios & factor decomposition.", color: "from-fuchsia-400/70 to-fuchsia-700/30" },
  ];
  return (
    <section id="agents" className="relative py-24 border-t border-border">
      <div className="mx-auto max-w-7xl px-6 grid lg:grid-cols-[1fr_1.1fr] gap-14 items-center">
        <div>
          <div className="text-xs uppercase tracking-[0.2em] text-emerald">Autonomous agents</div>
          <h2 className="mt-3 text-3xl sm:text-4xl font-semibold tracking-tight text-balance">
            A research desk that never sleeps.
          </h2>
          <p className="mt-4 text-muted-foreground max-w-md">
            Specialist agents collaborate on every ticker — drafting models,
            scoring sentiment, stress-testing portfolios and surfacing alpha
            while you sleep.
          </p>
          <Link
            to="/agents"
            className="mt-6 inline-flex items-center gap-1.5 text-sm text-emerald hover:text-emerald/80"
          >
            Meet the agent fleet <ArrowRight className="h-3.5 w-3.5" />
          </Link>
        </div>
        <div className="grid sm:grid-cols-2 gap-4">
          {agentList.map((a) => (
            <div key={a.name} className="glass rounded-xl p-5 relative overflow-hidden">
              <div className={`absolute -top-10 -right-10 h-32 w-32 rounded-full bg-gradient-to-br ${a.color} blur-2xl opacity-60`} />
              <div className="relative flex items-center gap-2 text-xs text-muted-foreground">
                <span className="h-1.5 w-1.5 rounded-full bg-emerald animate-pulse" /> running
              </div>
              <h3 className="relative mt-2 text-base font-semibold">{a.name}</h3>
              <p className="relative mt-1 text-sm text-muted-foreground">{a.desc}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

function Testimonials() {
  const items = [
    { q: "FinSight replaced four tools and one analyst seat in our first month. The agent fleet is genuinely additive.", n: "Marcus Lin", r: "Portfolio Manager, $4.2B L/S fund" },
    { q: "Citations down to the paragraph means we can actually ship AI-generated research to IC without a panic attack.", n: "Sara Okafor", r: "Head of Research, Family Office" },
    { q: "The terminal feels like Bloomberg if Bloomberg had been built in 2026 by people who actually use AI.", n: "Daniel Hwang", r: "Director, Global Macro" },
  ];
  return (
    <section id="customers" className="relative py-24 border-t border-border">
      <div className="mx-auto max-w-7xl px-6">
        <div className="text-xs uppercase tracking-[0.2em] text-emerald">Trusted by serious money</div>
        <h2 className="mt-3 text-3xl sm:text-4xl font-semibold tracking-tight max-w-2xl text-balance">
          Used inside hedge funds, family offices and bulge-bracket research desks.
        </h2>
        <div className="mt-10 grid md:grid-cols-3 gap-5">
          {items.map((t) => (
            <figure key={t.n} className="rounded-2xl border border-border bg-card/50 p-6">
              <Quote className="h-5 w-5 text-emerald" />
              <blockquote className="mt-4 text-[15px] leading-relaxed text-foreground/90">
                {t.q}
              </blockquote>
              <figcaption className="mt-5 text-sm">
                <div className="font-medium">{t.n}</div>
                <div className="text-muted-foreground">{t.r}</div>
              </figcaption>
            </figure>
          ))}
        </div>
        <div className="mt-12 grid grid-cols-2 sm:grid-cols-6 gap-6 opacity-60">
          {["GOLDMAN", "CITADEL", "JPMORGAN", "BRIDGEWATER", "BLACKROCK", "POINT72"].map((l) => (
            <div key={l} className="text-center text-xs tracking-[0.3em] text-muted-foreground font-medium">
              {l}
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

function Pricing() {
  const tiers = [
    {
      name: "Analyst",
      price: "$249",
      period: "/seat/mo",
      desc: "For independent analysts & small funds.",
      features: ["3 AI agents", "200 filings indexed", "GPT-5 + Claude Sonnet", "Citations & exports", "Email support"],
      cta: "Start free trial",
      featured: false,
    },
    {
      name: "Institutional",
      price: "$899",
      period: "/seat/mo",
      desc: "Full multi-agent platform for desks.",
      features: ["All 6 agents + orchestration", "Unlimited filings", "GPT-5 Pro · Claude Opus · Gemini", "Real-time market data", "SSO, SOC 2, audit logs", "Dedicated CSM"],
      cta: "Talk to sales",
      featured: true,
    },
    {
      name: "Enterprise",
      price: "Custom",
      period: "",
      desc: "Private cloud or on-prem deployment.",
      features: ["Bring-your-own LLM endpoints", "Private VPC / on-prem", "Custom agents & workflows", "Compliance & PII controls", "24/7 SLA"],
      cta: "Contact us",
      featured: false,
    },
  ];
  return (
    <section id="pricing" className="relative py-24 border-t border-border">
      <div className="mx-auto max-w-7xl px-6">
        <div className="max-w-2xl">
          <div className="text-xs uppercase tracking-[0.2em] text-emerald">Pricing</div>
          <h2 className="mt-3 text-3xl sm:text-4xl font-semibold tracking-tight text-balance">
            Priced like infrastructure, not like a SaaS.
          </h2>
        </div>
        <div className="mt-12 grid md:grid-cols-3 gap-5">
          {tiers.map((t) => (
            <div
              key={t.name}
              className={`relative rounded-2xl border p-7 ${
                t.featured
                  ? "border-emerald/40 bg-gradient-to-b from-emerald/[0.06] to-transparent glow-emerald"
                  : "border-border bg-card/40"
              }`}
            >
              {t.featured && (
                <div className="absolute -top-3 left-7 text-[10px] tracking-widest font-semibold bg-emerald text-black px-2 py-0.5 rounded">
                  MOST POPULAR
                </div>
              )}
              <div className="text-sm font-medium">{t.name}</div>
              <div className="mt-3 flex items-baseline gap-1">
                <span className="text-4xl font-semibold tracking-tight">{t.price}</span>
                <span className="text-muted-foreground text-sm">{t.period}</span>
              </div>
              <p className="mt-2 text-sm text-muted-foreground">{t.desc}</p>
              <ul className="mt-6 space-y-2.5 text-sm">
                {t.features.map((f) => (
                  <li key={f} className="flex items-start gap-2">
                    <Check className="h-4 w-4 mt-0.5 text-emerald shrink-0" />
                    <span>{f}</span>
                  </li>
                ))}
              </ul>
              <Link
                to="/dashboard"
                className={`mt-7 w-full inline-flex items-center justify-center h-10 rounded-md text-sm font-medium ${
                  t.featured
                    ? "bg-emerald text-black hover:brightness-110"
                    : "border border-border hover:border-emerald/40"
                }`}
              >
                {t.cta}
              </Link>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

function FinalCTA() {
  return (
    <section className="relative py-24 border-t border-border">
      <div className="mx-auto max-w-5xl px-6">
        <div className="relative overflow-hidden rounded-3xl border border-emerald/30 bg-gradient-to-br from-emerald/[0.08] via-card to-background p-10 lg:p-14 text-center">
          <div className="absolute -top-20 left-1/2 -translate-x-1/2 h-64 w-[600px] bg-emerald/30 blur-3xl rounded-full" />
          <div className="relative">
            <h3 className="text-3xl sm:text-4xl font-semibold tracking-tight text-balance">
              Ship better research. In hours, not weeks.
            </h3>
            <p className="mt-4 text-muted-foreground max-w-xl mx-auto">
              Spin up your workspace, drop in your filings, and let the agents
              start working before your next IC meeting.
            </p>
            <div className="mt-7 flex flex-wrap items-center justify-center gap-3">
              <Link
                to="/dashboard"
                className="inline-flex items-center gap-1.5 h-11 px-5 rounded-md bg-emerald text-black text-sm font-semibold glow-emerald"
              >
                Launch FinSight terminal <ArrowRight className="h-4 w-4" />
              </Link>
              <a
                href="#pricing"
                className="inline-flex items-center gap-1.5 h-11 px-5 rounded-md border border-border bg-card/60 text-sm font-medium hover:border-emerald/40"
              >
                See pricing
              </a>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

function Footer() {
  return (
    <footer className="border-t border-border py-12">
      <div className="mx-auto max-w-7xl px-6 grid md:grid-cols-4 gap-10 text-sm">
        <div className="md:col-span-1">
          <Logo />
          <p className="mt-4 text-muted-foreground text-xs leading-relaxed">
            Institutional-grade AI for equity research. Built in NYC.
          </p>
        </div>
        {[
          { h: "Product", l: ["Dashboard", "AI Chat", "Research", "Documents", "Agents"] },
          { h: "Company", l: ["About", "Customers", "Careers", "Press", "Contact"] },
          { h: "Legal", l: ["Terms", "Privacy", "Security", "SOC 2", "DPA"] },
        ].map((c) => (
          <div key={c.h}>
            <div className="text-xs uppercase tracking-[0.18em] text-muted-foreground">{c.h}</div>
            <ul className="mt-3 space-y-2">
              {c.l.map((x) => (
                <li key={x}><a href="#" className="text-muted-foreground hover:text-foreground transition-colors">{x}</a></li>
              ))}
            </ul>
          </div>
        ))}
      </div>
      <div className="mx-auto max-w-7xl px-6 mt-10 pt-6 border-t border-border flex flex-wrap items-center justify-between gap-3 text-xs text-muted-foreground">
        <div>© 2026 FinSight AI, Inc. Market data for illustration only.</div>
        <div className="flex items-center gap-2">
          <span className="h-1.5 w-1.5 rounded-full bg-emerald animate-pulse" /> All systems operational
        </div>
      </div>
    </footer>
  );
}

function Landing() {
  // SEO: single H1, semantic structure
  useEffect(() => {
    document.documentElement.classList.add("dark");
  }, []);
  return (
    <div className="min-h-screen bg-background text-foreground">
      <Nav />
      <Ticker />
      <Hero />
      <Features />
      <AgentsShowcase />
      <Testimonials />
      <Pricing />
      <FinalCTA />
      <Footer />
    </div>
  );
}
