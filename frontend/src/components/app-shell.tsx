import { Link, useRouterState } from "@tanstack/react-router";
import {
  LayoutDashboard,
  MessageSquare,
  LineChart,
  FileText,
  Bot,
  Settings,
  Sparkles,
  Search,
  Bell,
  Command,
} from "lucide-react";
import { type ReactNode, useState, useEffect } from "react";
import { cn } from "@/lib/utils";

const nav = [
  { to: "/dashboard", label: "Dashboard", icon: LayoutDashboard },
  { to: "/chat", label: "AI Chat", icon: MessageSquare },
  { to: "/research", label: "Research", icon: LineChart },
  { to: "/documents", label: "Documents", icon: FileText },
  { to: "/agents", label: "Agents", icon: Bot },
  { to: "/settings", label: "Settings", icon: Settings },
];

function Logo({ size = "md" }: { size?: "sm" | "md" }) {
  return (
    <Link to="/" className="flex items-center gap-2.5 group">
      <div
        className={cn(
          "relative flex items-center justify-center rounded-lg bg-gradient-to-br from-emerald-400/90 to-emerald-600/70 shadow-[0_0_20px_-4px_var(--emerald)]",
          size === "md" ? "h-8 w-8" : "h-7 w-7",
        )}
        style={{
          backgroundImage:
            "linear-gradient(135deg, oklch(0.85 0.18 160), oklch(0.55 0.16 170))",
        }}
      >
        <Sparkles className="h-4 w-4 text-black" strokeWidth={2.5} />
      </div>
      <div className="flex flex-col leading-none">
        <span className="text-[15px] font-semibold tracking-tight text-foreground">
          FinSight<span className="text-emerald">.AI</span>
        </span>
        <span className="text-[10px] uppercase tracking-[0.18em] text-muted-foreground">
          Institutional
        </span>
      </div>
    </Link>
  );
}

export function Sidebar() {
  const { location } = useRouterState();
  return (
    <aside className="hidden lg:flex w-60 shrink-0 flex-col border-r border-border bg-sidebar/60 backdrop-blur-xl">
      <div className="h-16 px-4 flex items-center border-b border-border">
        <Logo />
      </div>
      <nav className="flex-1 p-3 space-y-0.5">
        <div className="px-2 pb-2 text-[10px] uppercase tracking-[0.18em] text-muted-foreground/70">
          Workspace
        </div>
        {nav.map((item) => {
          const active = location.pathname.startsWith(item.to);
          const Icon = item.icon;
          return (
            <Link
              key={item.to}
              to={item.to}
              className={cn(
                "group flex items-center gap-3 rounded-md px-2.5 py-2 text-sm transition-colors",
                active
                  ? "bg-sidebar-accent text-foreground"
                  : "text-muted-foreground hover:bg-sidebar-accent/60 hover:text-foreground",
              )}
            >
              <Icon
                className={cn(
                  "h-4 w-4 transition-colors",
                  active ? "text-emerald" : "text-muted-foreground group-hover:text-foreground",
                )}
              />
              <span className="font-medium">{item.label}</span>
              {item.label === "AI Chat" && (
                <span className="ml-auto rounded-sm bg-emerald/10 px-1.5 py-0.5 text-[9px] font-medium text-emerald">
                  LIVE
                </span>
              )}
            </Link>
          );
        })}
      </nav>
      <div className="p-3 border-t border-border">
        <div className="rounded-lg border border-border bg-card/60 p-3">
          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            <span className="h-1.5 w-1.5 rounded-full bg-emerald animate-pulse" />
            6 agents online
          </div>
          <div className="mt-2 text-[11px] text-muted-foreground">
            14.2k tokens / min · 99.98% uptime
          </div>
        </div>
      </div>
    </aside>
  );
}

export function TopBar({ title, subtitle }: { title?: string; subtitle?: string }) {
  const [now, setNow] = useState("");
  useEffect(() => {
    const t = () =>
      setNow(
        new Date().toLocaleTimeString("en-US", {
          hour: "2-digit",
          minute: "2-digit",
          second: "2-digit",
          hour12: false,
        }) + " EST",
      );
    t();
    const id = setInterval(t, 1000);
    return () => clearInterval(id);
  }, []);
  return (
    <header className="h-16 border-b border-border bg-background/70 backdrop-blur-xl sticky top-0 z-30">
      <div className="h-full px-4 lg:px-6 flex items-center gap-4">
        <div className="flex-1 min-w-0">
          {title && (
            <h1 className="text-[15px] font-semibold tracking-tight truncate">
              {title}
            </h1>
          )}
          {subtitle && (
            <p className="text-xs text-muted-foreground truncate">{subtitle}</p>
          )}
        </div>
        <button className="hidden md:flex items-center gap-2 h-9 px-3 rounded-md border border-border bg-card/50 text-xs text-muted-foreground hover:text-foreground transition-colors min-w-[260px]">
          <Search className="h-3.5 w-3.5" />
          <span>Ask FinSight or search tickers…</span>
          <span className="ml-auto inline-flex items-center gap-1 text-[10px] text-muted-foreground/80">
            <Command className="h-3 w-3" /> K
          </span>
        </button>
        <div className="hidden md:block text-[11px] font-mono text-muted-foreground">
          {now}
        </div>
        <button className="relative h-9 w-9 grid place-items-center rounded-md border border-border bg-card/50 text-muted-foreground hover:text-foreground">
          <Bell className="h-4 w-4" />
          <span className="absolute top-1.5 right-1.5 h-1.5 w-1.5 rounded-full bg-emerald" />
        </button>
        <div className="h-9 w-9 rounded-md bg-gradient-to-br from-emerald-400/80 to-emerald-700/60 grid place-items-center text-[11px] font-semibold text-black"
             style={{ backgroundImage: "linear-gradient(135deg, oklch(0.85 0.18 160), oklch(0.5 0.14 170))" }}>
          AK
        </div>
      </div>
    </header>
  );
}

export function AppShell({
  children,
  title,
  subtitle,
}: {
  children: ReactNode;
  title?: string;
  subtitle?: string;
}) {
  return (
    <div className="flex min-h-screen bg-background text-foreground">
      <Sidebar />
      <div className="flex-1 min-w-0 flex flex-col">
        <TopBar title={title} subtitle={subtitle} />
        <main className="flex-1 min-w-0">{children}</main>
      </div>
    </div>
  );
}

export { Logo };
