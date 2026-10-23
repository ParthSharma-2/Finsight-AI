import { Link } from 'react-router-dom';
import { ArrowRight, Sparkles, TrendingUp, FileText, Database, ShieldCheck, Zap, BarChart3, ChevronRight } from 'lucide-react';

export default function Home({ onOpenCopilot }) {
  return (
    <div className="relative min-h-[calc(100vh-3.5rem)] overflow-hidden">
      {/* Apple-style ambient background glow */}
      <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[800px] h-[450px] bg-indigo-500/15 dark:bg-indigo-600/20 blur-[130px] rounded-full pointer-events-none -z-10" />
      <div className="absolute top-40 right-10 w-[350px] h-[350px] bg-emerald-500/10 dark:bg-emerald-500/15 blur-[100px] rounded-full pointer-events-none -z-10" />

      {/* Hero Section */}
      <div className="max-w-6xl mx-auto px-4 sm:px-6 pt-16 pb-20 text-center">
        {/* Release Pill */}
        <div className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-zinc-100 dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 text-xs font-medium text-zinc-700 dark:text-zinc-300 mb-8 shadow-sm hover:border-indigo-500/40 transition-colors">
          <Sparkles className="w-3.5 h-3.5 text-indigo-500" />
          <span>FinSight 2.0 • Powered by Groq & LangGraph</span>
          <span className="w-1 h-1 rounded-full bg-zinc-400" />
          <span className="text-indigo-600 dark:text-indigo-400">Institutional Edition</span>
        </div>

        {/* Hero Headline */}
        <h1 className="text-4xl sm:text-6xl lg:text-7xl font-bold tracking-tight text-zinc-950 dark:text-white max-w-4xl mx-auto leading-[1.1]">
          Institutional Market Intelligence,{' '}
          <span className="bg-gradient-to-r from-indigo-500 via-indigo-600 to-purple-600 bg-clip-text text-transparent">
            Powered by AI.
          </span>
        </h1>

        {/* Subtitle */}
        <p className="mt-6 text-lg sm:text-xl text-zinc-600 dark:text-zinc-400 max-w-2xl mx-auto font-normal leading-relaxed">
          Deep equity research in seconds. Analyze live ticker fundamentals, parse complex 10-K filings with grounded RAG, and query structured financial data with natural language.
        </p>

        {/* Action Buttons */}
        <div className="mt-9 flex flex-col sm:flex-row items-center justify-center gap-3.5">
          <Link
            to="/markets"
            className="w-full sm:w-auto flex items-center justify-center gap-2 px-6 py-3.5 rounded-xl bg-zinc-900 dark:bg-white text-white dark:text-zinc-950 text-sm font-semibold shadow-lg shadow-zinc-900/10 hover:bg-zinc-800 dark:hover:bg-zinc-100 transition-all hover:scale-[1.02] active:scale-[0.98]"
          >
            <span>Launch Markets Workspace</span>
            <ArrowRight className="w-4 h-4" />
          </Link>

          <button
            onClick={onOpenCopilot}
            className="w-full sm:w-auto flex items-center justify-center gap-2 px-6 py-3.5 rounded-xl bg-white dark:bg-zinc-900 text-zinc-900 dark:text-zinc-100 text-sm font-semibold border border-zinc-200 dark:border-zinc-800 hover:bg-zinc-50 dark:hover:bg-zinc-800/80 shadow-sm transition-all hover:scale-[1.02] active:scale-[0.98]"
          >
            <Sparkles className="w-4 h-4 text-indigo-500" />
            <span>Open AI Copilot</span>
          </button>
        </div>

        {/* Interactive Floating Preview Card */}
        <div className="mt-16 max-w-3xl mx-auto relative group">
          <div className="absolute -inset-1 bg-gradient-to-r from-indigo-500 to-purple-600 rounded-3xl blur-xl opacity-25 group-hover:opacity-40 transition duration-500 pointer-events-none" />

          <div className="relative bg-white dark:bg-zinc-900/90 rounded-2xl border border-zinc-200 dark:border-zinc-800/80 p-6 sm:p-8 shadow-2xl backdrop-blur-xl text-left">
            {/* Window chrome header */}
            <div className="flex items-center justify-between pb-4 border-b border-zinc-100 dark:border-zinc-800/80 mb-5">
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-full bg-rose-500/80" />
                <div className="w-3 h-3 rounded-full bg-amber-500/80" />
                <div className="w-3 h-3 rounded-full bg-emerald-500/80" />
                <span className="text-xs font-mono text-zinc-400 dark:text-zinc-500 ml-2">FinSight Copilot — Analysis Session</span>
              </div>
              <div className="flex items-center gap-1.5 text-[11px] font-mono text-emerald-600 dark:text-emerald-400 bg-emerald-500/10 px-2 py-0.5 rounded-full border border-emerald-500/20">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
                Connected
              </div>
            </div>

            {/* Prompt */}
            <div className="flex items-start gap-3 mb-4">
              <div className="w-7 h-7 rounded-lg bg-zinc-100 dark:bg-zinc-800 flex items-center justify-center text-xs font-semibold text-zinc-700 dark:text-zinc-300 shrink-0">
                You
              </div>
              <div className="text-sm font-medium text-zinc-800 dark:text-zinc-200 pt-1">
                "Analyze Apple (AAPL) latest quarterly performance and cloud revenue segment."
              </div>
            </div>

            {/* Response Card */}
            <div className="bg-zinc-50 dark:bg-zinc-950/60 rounded-xl p-4 sm:p-5 border border-zinc-200/80 dark:border-zinc-800/80">
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2">
                  <span className="font-bold text-sm text-zinc-900 dark:text-zinc-100">AAPL</span>
                  <span className="text-xs font-mono text-zinc-500">Apple Inc.</span>
                  <span className="text-xs font-mono font-semibold text-emerald-600 dark:text-emerald-400 bg-emerald-500/10 px-2 py-0.5 rounded border border-emerald-500/20">
                    +2.14% ▲
                  </span>
                </div>
                <span className="text-xs font-mono text-zinc-400">$228.45 USD</span>
              </div>

              <p className="text-xs text-zinc-600 dark:text-zinc-400 leading-relaxed mb-3">
                Services and Subscription revenue reached an all-time record, growing +14.1% YoY. Gross margins expanded to 46.2%, driven by hardware mix stabilization.
              </p>

              {/* Metric pills */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs font-mono">
                <div className="bg-white dark:bg-zinc-900 p-2 rounded-lg border border-zinc-200/60 dark:border-zinc-800">
                  <div className="text-[10px] text-zinc-400">P/E RATIO</div>
                  <div className="font-semibold text-zinc-900 dark:text-zinc-100 mt-0.5">33.2x</div>
                </div>
                <div className="bg-white dark:bg-zinc-900 p-2 rounded-lg border border-zinc-200/60 dark:border-zinc-800">
                  <div className="text-[10px] text-zinc-400">MARKET CAP</div>
                  <div className="font-semibold text-zinc-900 dark:text-zinc-100 mt-0.5">$3.48T</div>
                </div>
                <div className="bg-white dark:bg-zinc-900 p-2 rounded-lg border border-zinc-200/60 dark:border-zinc-800">
                  <div className="text-[10px] text-zinc-400">OPERATING MARGIN</div>
                  <div className="font-semibold text-emerald-600 dark:text-emerald-400 mt-0.5">30.1%</div>
                </div>
                <div className="bg-white dark:bg-zinc-900 p-2 rounded-lg border border-zinc-200/60 dark:border-zinc-800">
                  <div className="text-[10px] text-zinc-400">SOURCE</div>
                  <div className="font-semibold text-indigo-600 dark:text-indigo-400 mt-0.5">Q3 10-Q [p. 18]</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* 3 Pillars Section */}
      <div className="border-t border-zinc-200 dark:border-zinc-800/80 bg-white/50 dark:bg-zinc-900/30 backdrop-blur-sm py-20">
        <div className="max-w-6xl mx-auto px-4 sm:px-6">
          <div className="text-center mb-12">
            <h2 className="text-2xl sm:text-3xl font-bold tracking-tight text-zinc-900 dark:text-white">
              Institutional-grade capabilities. Zero setup.
            </h2>
            <p className="mt-2 text-sm text-zinc-500 dark:text-zinc-400">
              Three unified pillars delivering audited financial answers in milliseconds.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {/* Pillar 1 */}
            <div className="fin-card p-6 sm:p-7 relative overflow-hidden group">
              <div className="w-10 h-10 rounded-xl bg-indigo-500/10 text-indigo-600 dark:text-indigo-400 flex items-center justify-center mb-5 border border-indigo-500/20">
                <TrendingUp className="w-5 h-5" />
              </div>
              <h3 className="font-semibold text-base text-zinc-900 dark:text-zinc-100 mb-2">
                Live Market Intelligence
              </h3>
              <p className="text-sm text-zinc-500 dark:text-zinc-400 leading-relaxed mb-4">
                Real-time stock price quotes, multi-period historical returns, and side-by-side stock comparison across US and Indian exchanges.
              </p>
              <Link to="/markets" className="inline-flex items-center gap-1 text-xs font-semibold text-indigo-600 dark:text-indigo-400 hover:underline">
                <span>View Markets Dashboard</span>
                <ChevronRight className="w-3.5 h-3.5" />
              </Link>
            </div>

            {/* Pillar 2 */}
            <div className="fin-card p-6 sm:p-7 relative overflow-hidden group">
              <div className="w-10 h-10 rounded-xl bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 flex items-center justify-center mb-5 border border-emerald-500/20">
                <FileText className="w-5 h-5" />
              </div>
              <h3 className="font-semibold text-base text-zinc-900 dark:text-zinc-100 mb-2">
                Grounded RAG Ingestion
              </h3>
              <p className="text-sm text-zinc-500 dark:text-zinc-400 leading-relaxed mb-4">
                Upload 10-K, 10-Q filings and annual reports. ChromaDB vector retrieval ensures answers are strictly grounded with exact page citations.
              </p>
              <Link to="/research" className="inline-flex items-center gap-1 text-xs font-semibold text-emerald-600 dark:text-emerald-400 hover:underline">
                <span>Open Research Workbench</span>
                <ChevronRight className="w-3.5 h-3.5" />
              </Link>
            </div>

            {/* Pillar 3 */}
            <div className="fin-card p-6 sm:p-7 relative overflow-hidden group">
              <div className="w-10 h-10 rounded-xl bg-purple-500/10 text-purple-600 dark:text-purple-400 flex items-center justify-center mb-5 border border-purple-500/20">
                <Database className="w-5 h-5" />
              </div>
              <h3 className="font-semibold text-base text-zinc-900 dark:text-zinc-100 mb-2">
                Structured SQL Database
              </h3>
              <p className="text-sm text-zinc-500 dark:text-zinc-400 leading-relaxed mb-4">
                Query audited financial databases directly. Extract multi-year revenue breakdowns, segment contributions, and geographical splits without hallucinations.
              </p>
              <button onClick={onOpenCopilot} className="inline-flex items-center gap-1 text-xs font-semibold text-purple-600 dark:text-purple-400 hover:underline">
                <span>Query Database</span>
                <ChevronRight className="w-3.5 h-3.5" />
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
