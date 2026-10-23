import { Link } from 'react-router-dom';

export default function Footer() {
  return (
    <footer className="border-t border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-950 mt-auto transition-colors duration-300">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 py-10">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-8 mb-8 text-xs">
          {/* Brand */}
          <div className="col-span-2 md:col-span-1">
            <div className="flex items-center gap-2 mb-2 font-bold text-sm text-zinc-900 dark:text-zinc-100">
              <span className="w-5 h-5 rounded-md bg-indigo-600 text-white flex items-center justify-center text-[11px]">F</span>
              <span>FinSight AI</span>
            </div>
            <p className="text-zinc-500 dark:text-zinc-400 leading-relaxed">
              Institutional-grade AI financial intelligence & equity research platform.
            </p>
          </div>

          {/* Platform */}
          <div>
            <div className="font-semibold text-zinc-900 dark:text-zinc-100 uppercase tracking-wider text-[11px] mb-3">
              Platform
            </div>
            <ul className="space-y-2 text-zinc-500 dark:text-zinc-400">
              <li><Link to="/markets" className="hover:text-indigo-600 dark:hover:text-indigo-400 transition-colors">Markets</Link></li>
              <li><Link to="/research" className="hover:text-indigo-600 dark:hover:text-indigo-400 transition-colors">Research</Link></li>
              <li><Link to="/terminal" className="hover:text-indigo-600 dark:hover:text-indigo-400 transition-colors">Terminal</Link></li>
            </ul>
          </div>

          {/* Stack */}
          <div>
            <div className="font-semibold text-zinc-900 dark:text-zinc-100 uppercase tracking-wider text-[11px] mb-3">
              Engine
            </div>
            <ul className="space-y-2 text-zinc-500 dark:text-zinc-400 font-mono text-[11px]">
              <li>React 18 + Tailwind</li>
              <li>FastAPI + Uvicorn</li>
              <li>LangGraph Orchestration</li>
              <li>Groq • GPT-OSS 120B</li>
            </ul>
          </div>

          {/* Infrastructure */}
          <div>
            <div className="font-semibold text-zinc-900 dark:text-zinc-100 uppercase tracking-wider text-[11px] mb-3">
              Hosting
            </div>
            <ul className="space-y-2 text-zinc-500 dark:text-zinc-400 font-mono text-[11px]">
              <li className="flex items-center gap-1.5">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-500" />
                Vercel Edge Frontend
              </li>
              <li className="flex items-center gap-1.5">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-500" />
                AWS EC2 Backend
              </li>
              <li>ChromaDB Vector Store</li>
            </ul>
          </div>
        </div>

        <div className="pt-6 border-t border-zinc-100 dark:border-zinc-900 flex flex-col sm:flex-row items-center justify-between text-[11px] text-zinc-400 dark:text-zinc-500">
          <div>© {new Date().getFullYear()} FinSight AI. Institutional Market Intelligence.</div>
          <div className="mt-2 sm:mt-0">Not financial or investment advice. For institutional analysis and demonstration.</div>
        </div>
      </div>
    </footer>
  );
}
