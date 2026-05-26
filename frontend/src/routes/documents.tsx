import { createFileRoute } from "@tanstack/react-router";
import { AppShell } from "@/components/app-shell";
import { useState } from "react";
import {
  Upload,
  FileText,
  Search,
  Sparkles,
  CheckCircle2,
  Loader2,
  Filter,
  Download,
} from "lucide-react";
import { documents } from "@/lib/mock-data";

export const Route = createFileRoute("/documents")({
  component: DocsPage,
});

function DocsPage() {
  const [drag, setDrag] = useState(false);
  const [selected, setSelected] = useState(documents[0]);

  return (
    <AppShell title="RAG Document Workspace" subtitle="2,134 chunks indexed · 412 documents · vector store v3">
      <div className="grid grid-cols-1 xl:grid-cols-[320px_1fr] h-[calc(100vh-4rem)]">
        {/* Sidebar: doc list */}
        <aside className="border-r border-border flex flex-col min-h-0">
          <div className="p-3 border-b border-border space-y-2">
            <div
              onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
              onDragLeave={() => setDrag(false)}
              onDrop={(e) => { e.preventDefault(); setDrag(false); }}
              className={`rounded-lg border-2 border-dashed p-4 text-center transition-colors cursor-pointer ${
                drag ? "border-emerald bg-emerald/5" : "border-border bg-card/40 hover:border-emerald/40"
              }`}
            >
              <Upload className="h-5 w-5 mx-auto text-emerald" />
              <div className="mt-2 text-xs font-medium">Drop SEC filings here</div>
              <div className="text-[10px] text-muted-foreground mt-0.5">PDF, DOCX, TXT · up to 200MB</div>
            </div>
            <div className="relative">
              <Search className="absolute left-2.5 top-2.5 h-3.5 w-3.5 text-muted-foreground" />
              <input
                placeholder="Search documents…"
                className="w-full h-8 rounded-md bg-background border border-border pl-8 pr-2 text-xs focus:outline-none focus:border-emerald/40"
              />
            </div>
          </div>
          <div className="flex-1 overflow-y-auto scrollbar-thin">
            {documents.map((d) => (
              <button
                key={d.id}
                onClick={() => setSelected(d)}
                className={`w-full text-left px-3 py-2.5 border-b border-border hover:bg-accent/30 ${
                  selected.id === d.id ? "bg-accent/40 border-l-2 border-l-emerald" : ""
                }`}
              >
                <div className="flex items-center gap-2">
                  <FileText className="h-3.5 w-3.5 text-emerald shrink-0" />
                  <div className="text-xs font-medium truncate flex-1">{d.name}</div>
                </div>
                <div className="mt-1 flex items-center gap-2 text-[10px] text-muted-foreground">
                  <span>{d.pages}p</span>
                  <span>·</span>
                  <span>{d.uploaded}</span>
                  <span className="ml-auto inline-flex items-center gap-1">
                    {d.status === "indexed" ? (
                      <><CheckCircle2 className="h-2.5 w-2.5 text-emerald" /><span className="text-emerald">indexed</span></>
                    ) : (
                      <><Loader2 className="h-2.5 w-2.5 animate-spin text-amber-400" /><span className="text-amber-400">indexing</span></>
                    )}
                  </span>
                </div>
                {d.status === "indexing" && (
                  <div className="mt-1.5 h-1 rounded-full bg-muted overflow-hidden">
                    <div className="h-full bg-amber-400 animate-pulse" style={{ width: "62%" }} />
                  </div>
                )}
              </button>
            ))}
          </div>
        </aside>

        {/* Main viewer */}
        <div className="flex flex-col min-h-0">
          <div className="px-5 py-4 border-b border-border flex items-center gap-3">
            <FileText className="h-4 w-4 text-emerald" />
            <div className="flex-1 min-w-0">
              <div className="text-sm font-semibold truncate">{selected.name}</div>
              <div className="text-xs text-muted-foreground">
                {selected.pages} pages · {selected.chunks} chunks · embedding: text-embedding-3-large
              </div>
            </div>
            <button className="h-8 px-2.5 rounded-md border border-border text-xs inline-flex items-center gap-1.5 hover:border-emerald/40">
              <Download className="h-3.5 w-3.5" /> Export
            </button>
            <button className="h-8 px-2.5 rounded-md bg-emerald text-black text-xs font-semibold inline-flex items-center gap-1.5">
              <Sparkles className="h-3.5 w-3.5" /> Ask this document
            </button>
          </div>

          {/* RAG search */}
          <div className="px-5 py-3 border-b border-border">
            <div className="relative">
              <Search className="absolute left-3 top-2.5 h-4 w-4 text-emerald" />
              <input
                placeholder="Semantic search across this document…"
                className="w-full h-9 rounded-md bg-card/60 border border-border pl-9 pr-24 text-sm focus:outline-none focus:border-emerald/40"
                defaultValue="data-center revenue Blackwell ramp"
              />
              <div className="absolute right-2 top-1.5 inline-flex items-center gap-1 text-[10px] text-muted-foreground">
                <Filter className="h-3 w-3" /> 8 results
              </div>
            </div>
          </div>

          <div className="flex-1 overflow-y-auto scrollbar-thin">
            <div className="grid lg:grid-cols-[1fr_360px] gap-0 h-full">
              {/* Document viewer (mock pages) */}
              <div className="p-6 space-y-4 bg-background/30">
                {[12, 13, 31].map((pageNum) => (
                  <div key={pageNum} className="rounded-lg border border-border bg-card/40 p-6">
                    <div className="text-[11px] text-muted-foreground mb-3">Page {pageNum}</div>
                    <div className="space-y-2 text-sm leading-relaxed text-foreground/85">
                      <p>
                        Item 2. Management's Discussion and Analysis of Financial Condition and Results of Operations.
                      </p>
                      <p>
                        <mark className="bg-emerald/25 text-foreground rounded px-1">
                          Data center revenue of $30.8 billion grew 112% year-over-year, driven primarily by Blackwell platform shipments
                        </mark>
                        {" "}exceeding internal forecasts across every tier of customer. Networking attach
                        increased to approximately 71% of data center revenue, compared with 64% in the
                        prior quarter. Gross margin expanded 220 basis points to 75.1% on a richer product mix.
                      </p>
                      <p>
                        Customer concentration remains elevated, with our top four customers in aggregate
                        representing approximately 46% of total data center revenue during the period. We
                        continue to invest in capacity expansion at CoWoS-L with our foundry partners.
                      </p>
                    </div>
                  </div>
                ))}
              </div>

              {/* Right panel: AI contextual retrieval */}
              <aside className="border-l border-border bg-sidebar/40 p-4 space-y-4 overflow-y-auto">
                <div>
                  <div className="text-[11px] uppercase tracking-wider text-muted-foreground mb-2">
                    AI summary
                  </div>
                  <div className="rounded-lg border border-emerald/20 bg-emerald/5 p-3 text-xs leading-relaxed">
                    NVDA Q3 10-Q confirms data-center +112% YoY at $30.8B with networking attach at 71%.
                    Gross margin +220 bps to 75.1%. Customer concentration (46% from top-4) is the main
                    structural risk. Capex guidance unchanged.
                  </div>
                </div>

                <div>
                  <div className="text-[11px] uppercase tracking-wider text-muted-foreground mb-2">
                    Top retrieved chunks
                  </div>
                  <div className="space-y-2">
                    {[
                      { p: 12, s: 0.94, t: "Data center revenue of $30.8B grew 112% YoY, driven primarily by Blackwell platform…" },
                      { p: 31, s: 0.91, t: "Customer concentration: top four customers represent ~46% of data center revenue…" },
                      { p: 18, s: 0.88, t: "Gross margin expanded 220 bps to 75.1% on richer Blackwell product mix…" },
                      { p: 44, s: 0.82, t: "Capital expenditures of $1.34B primarily for data-center capacity expansion…" },
                    ].map((c) => (
                      <div key={c.p} className="rounded-lg border border-border bg-card/60 p-2.5 hover:border-emerald/40 cursor-pointer">
                        <div className="flex items-center gap-2 text-[10px] text-muted-foreground">
                          <span className="font-mono text-emerald">p.{c.p}</span>
                          <span className="ml-auto font-mono">score {c.s}</span>
                        </div>
                        <p className="text-xs mt-1 text-foreground/85 leading-snug">{c.t}</p>
                      </div>
                    ))}
                  </div>
                </div>

                <div>
                  <div className="text-[11px] uppercase tracking-wider text-muted-foreground mb-2">
                    Index stats
                  </div>
                  <div className="rounded-lg border border-border bg-card/60 p-3 space-y-1.5 text-xs">
                    <div className="flex justify-between"><span className="text-muted-foreground">Chunks</span><span className="font-mono">{selected.chunks}</span></div>
                    <div className="flex justify-between"><span className="text-muted-foreground">Dim</span><span className="font-mono">3072</span></div>
                    <div className="flex justify-between"><span className="text-muted-foreground">Vector store</span><span className="font-mono">pgvector</span></div>
                    <div className="flex justify-between"><span className="text-muted-foreground">Index type</span><span className="font-mono">HNSW</span></div>
                  </div>
                </div>
              </aside>
            </div>
          </div>
        </div>
      </div>
    </AppShell>
  );
}
