import { useState, useRef } from 'react';
import AITerminal from '../components/AITerminal';
import { uploadDocument } from '../lib/api';

const SAMPLE_REPORTS = [
  { ticker: 'AAPL', title: 'Apple Inc. 10-K FY2024', type: '10-K', date: '2024-11-01', pages: 88 },
  { ticker: 'MSFT', title: 'Microsoft Annual Report 2024', type: '10-K', date: '2024-07-30', pages: 102 },
  { ticker: 'NVDA', title: 'NVIDIA Q4 FY2025 Earnings', type: '8-K', date: '2025-02-26', pages: 12 },
  { ticker: 'GOOGL', title: 'Alphabet 10-K FY2024', type: '10-K', date: '2025-01-30', pages: 96 },
  { ticker: 'META', title: 'Meta Q1 2025 10-Q', type: '10-Q', date: '2025-04-25', pages: 54 },
];

const RESEARCH_PROMPTS = [
  'What are AAPL main revenue segments?',
  'Summarize NVDA risk factors',
  'Compare MSFT and GOOGL cloud revenue',
  'What is Apple\'s free cash flow trend?',
  'Extract NVDA guidance for next quarter',
  'What does management say about AI capex?',
];

export default function Research() {
  const [uploadStatus, setUploadStatus] = useState(null); // null | 'uploading' | 'done' | 'error'
  const [uploadedFiles, setUploadedFiles] = useState([]);
  const fileInputRef = useRef();

  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setUploadStatus('uploading');
    try {
      // Attempt real upload; falls back gracefully if backend isn't ready
      await uploadDocument(file);
      setUploadedFiles(prev => [...prev, { name: file.name, size: file.size, status: 'indexed' }]);
      setUploadStatus('done');
    } catch {
      // Backend may not have /research/upload yet — show it as pending
      setUploadedFiles(prev => [...prev, { name: file.name, size: file.size, status: 'pending' }]);
      setUploadStatus('error');
    }
    setTimeout(() => setUploadStatus(null), 3000);
  };

  return (
    <div className="min-h-screen pt-12 bg-terminal-bg">
      <div className="max-w-7xl mx-auto px-4 md:px-6 py-8">
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center gap-2 mb-1">
            <span className="w-3 h-px bg-accent" />
            <span className="mono text-[10px] tracking-widest text-accent">RESEARCH WORKBENCH</span>
          </div>
          <h1 className="font-display font-700 text-2xl text-white mb-1">AI Financial Research</h1>
          <p className="font-body text-sm text-dim">
            RAG-powered document analysis — upload filings and query them with natural language
          </p>
        </div>

        <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
          {/* Left: Document library + upload */}
          <div className="space-y-4">
            {/* Upload panel */}
            <div className="panel">
              <div className="panel-header">
                <span style={{ color: 'var(--purple)' }}>⬆</span>
                UPLOAD DOCUMENT
              </div>
              <div className="p-4">
                <div
                  onClick={() => fileInputRef.current?.click()}
                  className="border border-dashed border-terminal hover:border-muted transition-colors rounded-none cursor-pointer p-6 text-center group"
                >
                  <div className="mono text-3xl text-muted mb-2 group-hover:text-dim transition-colors">⊕</div>
                  <div className="mono text-xs text-dim mb-1">Click to upload PDF</div>
                  <div className="mono text-[10px] text-muted">10-K, 10-Q, 8-K, Earnings Transcripts</div>
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept=".pdf"
                    className="hidden"
                    onChange={handleFileUpload}
                  />
                </div>

                {/* Upload status */}
                {uploadStatus === 'uploading' && (
                  <div className="mt-3 flex items-center gap-2 mono text-[11px] text-amber">
                    <div className="w-3 h-3 border-t border-amber rounded-full animate-spin" />
                    Uploading & indexing...
                  </div>
                )}
                {uploadStatus === 'done' && (
                  <div className="mt-3 mono text-[11px] text-green">✓ Document indexed successfully</div>
                )}
                {uploadStatus === 'error' && (
                  <div className="mt-3 mono text-[11px] text-amber">
                    ⚠ Backend RAG endpoint not ready — saved as pending
                  </div>
                )}

                {/* Uploaded files */}
                {uploadedFiles.length > 0 && (
                  <div className="mt-3 space-y-1.5">
                    {uploadedFiles.map((f, i) => (
                      <div key={i} className="flex items-center justify-between p-2 bg-surface border border-terminal">
                        <span className="mono text-[10px] text-dim truncate flex-1">{f.name}</span>
                        <span className="mono text-[9px] ml-2"
                              style={{ color: f.status === 'indexed' ? 'var(--green)' : 'var(--amber)' }}>
                          {f.status.toUpperCase()}
                        </span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>

            {/* Sample document library */}
            <div className="panel">
              <div className="panel-header">
                <span style={{ color: 'var(--accent)' }}>◈</span>
                INDEXED FILINGS
                <span className="ml-auto text-[9px] text-muted">DEMO DATA</span>
              </div>
              <div className="divide-y divide-terminal">
                {SAMPLE_REPORTS.map((report) => (
                  <div key={report.title} className="px-4 py-3 hover:bg-surface transition-colors cursor-pointer group">
                    <div className="flex items-center justify-between mb-1">
                      <div className="flex items-center gap-2">
                        <span className="mono text-[10px] font-600 text-accent">{report.ticker}</span>
                        <span className="mono text-[9px] px-1.5 py-0.5 border border-terminal text-muted">
                          {report.type}
                        </span>
                      </div>
                      <span className="mono text-[9px] text-muted">{report.date}</span>
                    </div>
                    <div className="mono text-[11px] text-dim group-hover:text-terminal transition-colors">
                      {report.title}
                    </div>
                    <div className="mono text-[9px] text-muted mt-0.5">{report.pages} pages</div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Center + Right: AI Query terminal */}
          <div className="xl:col-span-2 space-y-4">
            {/* Quick research prompts */}
            <div className="panel">
              <div className="panel-header">
                <span style={{ color: 'var(--green)' }}>◐</span>
                SUGGESTED QUERIES
              </div>
              <div className="p-3 flex flex-wrap gap-2">
                {RESEARCH_PROMPTS.map((p) => (
                  <button
                    key={p}
                    className="mono text-[10px] px-3 py-1.5 border border-terminal text-dim hover:text-accent hover:border-accent/40 transition-all"
                  >
                    {p}
                  </button>
                ))}
              </div>
            </div>

            {/* AI Terminal */}
            <AITerminal />

            {/* How RAG works */}
            <div className="panel p-4">
              <div className="mono text-[10px] tracking-widest text-muted mb-3">HOW IT WORKS</div>
              <div className="flex items-center gap-3 flex-wrap">
                {[
                  '① PDF Upload',
                  '→ Text Chunking',
                  '→ Embeddings (OpenAI)',
                  '→ Vector Store (Pinecone)',
                  '→ RAG Query',
                  '→ LLM Response',
                ].map((step, i) => (
                  <span key={i} className="mono text-[10px]"
                        style={{ color: step.startsWith('→') ? 'var(--dim)' : 'var(--accent)' }}>
                    {step}
                  </span>
                ))}
              </div>
              <div className="mono text-[10px] text-muted mt-2">
                Backend: FastAPI + LangChain + Pinecone/ChromaDB. Vector search returns relevant document chunks,
                which are passed as context to the LLM for grounded answers.
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
