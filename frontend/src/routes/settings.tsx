import { createFileRoute } from "@tanstack/react-router";
import { AppShell } from "@/components/app-shell";
import { useState } from "react";
import { Key, Eye, EyeOff, Copy, Check, Bell, Brain, Palette } from "lucide-react";

export const Route = createFileRoute("/settings")({
  component: SettingsPage,
});

const tabs = ["Profile", "API Keys", "Models", "Notifications", "Appearance"] as const;

function SettingsPage() {
  const [tab, setTab] = useState<(typeof tabs)[number]>("Profile");
  return (
    <AppShell title="Settings" subtitle="Workspace · Alpha Capital">
      <div className="p-4 lg:p-6 grid grid-cols-1 lg:grid-cols-[220px_1fr] gap-6 max-w-6xl">
        <aside>
          <nav className="space-y-0.5">
            {tabs.map((t) => (
              <button
                key={t}
                onClick={() => setTab(t)}
                className={`w-full text-left px-3 py-2 rounded-md text-sm transition-colors ${
                  tab === t ? "bg-accent text-foreground" : "text-muted-foreground hover:bg-accent/60 hover:text-foreground"
                }`}
              >
                {t}
              </button>
            ))}
          </nav>
        </aside>
        <section className="space-y-5">
          {tab === "Profile" && <ProfileTab />}
          {tab === "API Keys" && <KeysTab />}
          {tab === "Models" && <ModelsTab />}
          {tab === "Notifications" && <NotifTab />}
          {tab === "Appearance" && <AppearanceTab />}
        </section>
      </div>
    </AppShell>
  );
}

function Card({ title, desc, children }: { title: string; desc?: string; children: React.ReactNode }) {
  return (
    <div className="rounded-xl border border-border bg-card/60 p-5">
      <div className="text-sm font-semibold">{title}</div>
      {desc && <div className="text-xs text-muted-foreground mt-1">{desc}</div>}
      <div className="mt-4">{children}</div>
    </div>
  );
}

function Input(props: React.InputHTMLAttributes<HTMLInputElement>) {
  return (
    <input
      {...props}
      className="w-full h-9 rounded-md bg-background border border-border px-3 text-sm focus:outline-none focus:border-emerald/40"
    />
  );
}

function ProfileTab() {
  return (
    <>
      <Card title="Profile" desc="How you appear in FinSight">
        <div className="flex items-center gap-4">
          <div className="h-14 w-14 rounded-xl grid place-items-center text-base font-semibold text-black"
               style={{ backgroundImage: "linear-gradient(135deg, oklch(0.85 0.18 160), oklch(0.5 0.14 170))" }}>
            AK
          </div>
          <button className="h-8 px-3 rounded-md border border-border text-xs hover:border-emerald/40">Change avatar</button>
        </div>
        <div className="mt-5 grid sm:grid-cols-2 gap-4">
          <div>
            <label className="text-xs text-muted-foreground">Full name</label>
            <Input defaultValue="Alex Kim" />
          </div>
          <div>
            <label className="text-xs text-muted-foreground">Email</label>
            <Input defaultValue="alex@alphacap.com" />
          </div>
          <div>
            <label className="text-xs text-muted-foreground">Role</label>
            <Input defaultValue="Portfolio Manager" />
          </div>
          <div>
            <label className="text-xs text-muted-foreground">Timezone</label>
            <Input defaultValue="America/New_York" />
          </div>
        </div>
      </Card>
      <Card title="Workspace" desc="Alpha Capital · 14 seats">
        <div className="text-xs text-muted-foreground">SSO via Okta · SOC 2 Type II</div>
      </Card>
    </>
  );
}

function KeyRow({ label, value }: { label: string; value: string }) {
  const [shown, setShown] = useState(false);
  const [copied, setCopied] = useState(false);
  return (
    <div className="flex items-center gap-2 rounded-md border border-border bg-background/60 p-2">
      <Key className="h-3.5 w-3.5 text-emerald" />
      <div className="text-xs font-medium w-40">{label}</div>
      <code className="flex-1 font-mono text-xs text-muted-foreground truncate">
        {shown ? value : "•".repeat(value.length)}
      </code>
      <button className="h-7 w-7 grid place-items-center rounded hover:bg-accent" onClick={() => setShown((s) => !s)}>
        {shown ? <EyeOff className="h-3.5 w-3.5" /> : <Eye className="h-3.5 w-3.5" />}
      </button>
      <button
        className="h-7 w-7 grid place-items-center rounded hover:bg-accent"
        onClick={() => { navigator.clipboard?.writeText(value); setCopied(true); setTimeout(() => setCopied(false), 1200); }}
      >
        {copied ? <Check className="h-3.5 w-3.5 text-emerald" /> : <Copy className="h-3.5 w-3.5" />}
      </button>
    </div>
  );
}

function KeysTab() {
  return (
    <Card title="API Keys" desc="Bring your own LLM endpoints — keys are encrypted at rest">
      <div className="space-y-2">
        <KeyRow label="OpenAI" value="sk-proj-9a4f...c218" />
        <KeyRow label="Anthropic" value="sk-ant-api03-...7d2" />
        <KeyRow label="Google Gemini" value="AIzaSyB...kQ12" />
        <KeyRow label="Polygon.io" value="pk_live_...4tQa" />
      </div>
      <button className="mt-4 h-9 px-3 rounded-md bg-emerald text-black text-xs font-semibold">+ Add provider</button>
    </Card>
  );
}

function ModelsTab() {
  const opts = [
    { id: "research", label: "Research agent", val: "gpt-5-pro" },
    { id: "risk", label: "Risk agent", val: "claude-opus-4.1" },
    { id: "news", label: "News agent", val: "gemini-2.5-pro" },
    { id: "earnings", label: "Earnings agent", val: "gpt-5-pro" },
  ];
  return (
    <Card title="LLM routing" desc="Choose the model for each agent">
      <div className="space-y-3">
        {opts.map((o) => (
          <div key={o.id} className="flex items-center gap-3">
            <Brain className="h-4 w-4 text-emerald" />
            <div className="text-sm flex-1">{o.label}</div>
            <select defaultValue={o.val} className="h-9 rounded-md bg-background border border-border px-3 text-sm">
              <option>gpt-5-pro</option>
              <option>claude-opus-4.1</option>
              <option>claude-sonnet-4.5</option>
              <option>gemini-2.5-pro</option>
              <option>llama-3.3-70b</option>
            </select>
          </div>
        ))}
      </div>
    </Card>
  );
}

function NotifTab() {
  const items = [
    ["Price alerts", "Push & email when watchlist crosses thresholds", true],
    ["Earnings releases", "When a covered ticker reports", true],
    ["Agent completions", "When a long-running agent finishes", false],
    ["Risk thresholds", "VaR or concentration breaches", true],
  ] as const;
  return (
    <Card title="Notifications" desc="Choose what to be pinged about">
      <div className="space-y-2">
        {items.map(([l, d, on]) => (
          <Toggle key={l} label={l} desc={d} defaultOn={on} />
        ))}
      </div>
    </Card>
  );
}

function Toggle({ label, desc, defaultOn }: { label: string; desc: string; defaultOn: boolean }) {
  const [on, setOn] = useState(defaultOn);
  return (
    <div className="flex items-center gap-3 p-2.5 rounded-md hover:bg-accent/30">
      <Bell className="h-4 w-4 text-muted-foreground" />
      <div className="flex-1">
        <div className="text-sm">{label}</div>
        <div className="text-xs text-muted-foreground">{desc}</div>
      </div>
      <button
        onClick={() => setOn(!on)}
        className={`h-5 w-9 rounded-full p-0.5 transition-colors ${on ? "bg-emerald" : "bg-muted"}`}
      >
        <div className={`h-4 w-4 bg-background rounded-full transition-transform ${on ? "translate-x-4" : ""}`} />
      </button>
    </div>
  );
}

function AppearanceTab() {
  return (
    <Card title="Appearance" desc="FinSight is dark by design">
      <div className="flex items-center gap-3">
        <Palette className="h-4 w-4 text-emerald" />
        <div className="text-sm flex-1">Theme</div>
        <div className="flex gap-2">
          {["Institutional Dark", "Carbon", "Midnight"].map((t, i) => (
            <button key={t} className={`h-9 px-3 rounded-md border text-xs ${i === 0 ? "border-emerald/40 bg-emerald/10 text-emerald" : "border-border"}`}>
              {t}
            </button>
          ))}
        </div>
      </div>
    </Card>
  );
}
