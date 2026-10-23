import { useState, useRef } from 'react';
import { UploadCloud, FileText, CheckCircle2, AlertCircle, Sparkles, ArrowRight, Layers, FileCheck } from 'lucide-react';
import { uploadDocument } from '../lib/api';

const SAMPLE_REPORTS = [
  {
    ticker: 'AAPL',
    title: 'Apple Inc. 10-K Annual Report',
    type: '10-K',
    date: 'FY2024',
    pages: 88,
    kpis: ['Revenue: $391B', 'Gross Margin: 46.2%', 'Services Record'],
    prompts: ['What are AAPL main revenue segments?', 'Summarize key risk factors for Apple'],
  },
  {
    ticker: 'NVDA',
    title: 'NVIDIA Q4 Earnings & Financial Overview',
    type: '8-K',
    date: 'FY2025',
    pages: 14,
    kpis: ['Data Center: +409%', 'Gross Margin: 76.0%', 'Blackwell Architecture'],
    prompts: ['Extract NVDA guidance for next quarter', 'Summarize AI capex trend from filing'],
  },
  {
    ticker: 'MSFT',
    title: 'Microsoft Corporation Annual Report',
    type: '10-K',
    date: 'FY2024',
    pages: 102,
    kpis: ['Cloud Rev: $118B', 'Operating Income: +24%', 'Azure Growth: 29%'],
    prompts: ['Compare MSFT and GOOGL cloud revenue', 'What does management say about AI capex?'],
  },
];

export default function Research({ onOpenCopilot }) {
  const [uploadStatus, setUploadStatus] = useState(null); // null | 'uploading' | 'done' | 'error'
  const [errorMessage, setErrorMessage] = useState('');
  const [uploadedFiles, setUploadedFiles] = useState([
    { name: 'infosys_annual_report_fy24.pdf', size: 10485760, status: 'indexed', pages: 64, ticker: 'INFY' }
  ]);
  const fileInputRef = useRef();

  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setUploadStatus('uploading');
    setErrorMessage('');
    try {
      const res = await uploadDocument(file);
      if (res && res.status === 'error') {
        throw new Error(res.message || 'Upload rejected');
      }
      setUploadedFiles(prev => [
        { name: file.name, size: file.size, status: 'indexed', pages: 'Indexed', ticker: 'DOC' },
        ...prev,
      ]);
      setUploadStatus('done');
    } catch (err) {
      console.error(err);
      setErrorMessage(err.message || 'Upload failed');
      setUploadStatus('error');
    }
    setTimeout(() => setUploadStatus(null), 4000);
  };

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 py-8">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-8">
        <div>
          <div className="flex items-center gap-2 mb-1.5">
            <span className="text-xs font-mono font-medium text-emerald-600 dark:text-emerald-400 uppercase tracking-wider">
              RAG Workbench
            </span>
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
            <span className="text-[11px] font-mono text-zinc-500">Grounded Document Vector Index</span>
          </div>
          <h1 className="text-2xl sm:text-3xl font-bold tracking-tight text-zinc-900 dark:text-zinc-50">
            Financial Research Workbench
          </h1>
          <p className="mt-1 text-sm text-zinc-500 dark:text-zinc-400">
            Upload financial filings, 10-K reports, and transcripts. Ask grounded questions with exact page citations.
          </p>
        </div>

        <button
          onClick={onOpenCopilot}
          className="self-start sm:self-auto flex items-center gap-2 px-4 py-2 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold shadow-sm transition-all hover:scale-[1.02] active:scale-[0.98]"
        >
          <Sparkles className="w-3.5 h-3.5" />
          <span>Ask Copilot About Filings</span>
        </button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Column: Upload Dropzone & Recent Uploads */}
        <div className="space-y-6">
          {/* Upload Dropzone */}
          <div className="fin-card p-6">
            <div className="flex items-center gap-2 mb-4 text-xs font-semibold text-zinc-900 dark:text-zinc-100">
              <UploadCloud className="w-4 h-4 text-indigo-500" />
              <span>Upload Document</span>
            </div>

            <div
              onClick={() => fileInputRef.current?.click()}
              className="border-2 border-dashed border-zinc-200 dark:border-zinc-800 hover:border-indigo-500/50 dark:hover:border-indigo-500/50 rounded-xl p-8 text-center cursor-pointer transition-all bg-zinc-50/50 dark:bg-zinc-900/30 group"
            >
              <div className="w-12 h-12 rounded-2xl bg-indigo-50 dark:bg-indigo-950/40 text-indigo-600 dark:text-indigo-400 flex items-center justify-center mx-auto mb-3 group-hover:scale-110 transition-transform">
                <UploadCloud className="w-6 h-6" />
              </div>
              <p className="text-sm font-medium text-zinc-800 dark:text-zinc-200">
                Click to upload filing
              </p>
              <p className="text-xs text-zinc-400 mt-1">
                PDF only (10-K, 10-Q, 8-K, Annual Reports)
              </p>
              <input
                ref={fileInputRef}
                type="file"
                accept=".pdf"
                className="hidden"
                onChange={handleFileUpload}
              />
            </div>

            {/* Status Feedback */}
            {uploadStatus === 'uploading' && (
              <div className="mt-3 p-3 rounded-lg bg-indigo-50 dark:bg-indigo-950/30 border border-indigo-200 dark:border-indigo-800/60 flex items-center gap-2.5 text-xs text-indigo-700 dark:text-indigo-300">
                <div className="w-3.5 h-3.5 rounded-full border-2 border-indigo-600 border-t-transparent animate-spin" />
                <span>Chunking & indexing into ChromaDB...</span>
              </div>
            )}

            {uploadStatus === 'done' && (
              <div className="mt-3 p-3 rounded-lg bg-emerald-50 dark:bg-emerald-950/30 border border-emerald-200 dark:border-emerald-800/60 flex items-center gap-2 text-xs text-emerald-700 dark:text-emerald-300 font-medium">
                <CheckCircle2 className="w-4 h-4 text-emerald-600 dark:text-emerald-400" />
                <span>Document successfully indexed! Ready for queries.</span>
              </div>
            )}

            {uploadStatus === 'error' && (
              <div className="mt-3 p-3 rounded-lg bg-rose-50 dark:bg-rose-950/30 border border-rose-200 dark:border-rose-800/60 flex items-center gap-2 text-xs text-rose-700 dark:text-rose-300">
                <AlertCircle className="w-4 h-4 text-rose-600 shrink-0" />
                <span className="truncate">{errorMessage || 'Upload failed. Ensure PDF has readable text.'}</span>
              </div>
            )}

            {/* Uploaded Files List */}
            {uploadedFiles.length > 0 && (
              <div className="mt-5 pt-4 border-t border-zinc-100 dark:border-zinc-800">
                <div className="text-xs font-semibold text-zinc-500 mb-2">Indexed Documents</div>
                <div className="space-y-2">
                  {uploadedFiles.map((f, i) => (
                    <div
                      key={i}
                      className="p-3 rounded-xl bg-zinc-50 dark:bg-zinc-900/60 border border-zinc-200/80 dark:border-zinc-800/80 flex items-center justify-between text-xs"
                    >
                      <div className="flex items-center gap-2.5 truncate mr-2">
                        <FileCheck className="w-4 h-4 text-emerald-500 shrink-0" />
                        <span className="font-medium text-zinc-800 dark:text-zinc-200 truncate">{f.name}</span>
                      </div>
                      <span className="font-mono text-[10px] font-semibold text-emerald-600 dark:text-emerald-400 bg-emerald-500/10 px-2 py-0.5 rounded border border-emerald-500/20 shrink-0">
                        INDEXED
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* RAG Engine Info Card */}
          <div className="fin-card p-5">
            <div className="flex items-center gap-2 text-xs font-semibold text-zinc-900 dark:text-zinc-100 mb-2">
              <Layers className="w-4 h-4 text-purple-500" />
              <span>How Vector Retrieval Works</span>
            </div>
            <p className="text-xs text-zinc-500 leading-relaxed mb-3">
              Uploaded PDFs are parsed using <code className="text-zinc-700 dark:text-zinc-300 font-mono">PyPDFLoader</code>, chunked into 500-token windows, and embedded locally into ChromaDB.
            </p>
            <div className="text-[11px] font-mono text-zinc-400 bg-zinc-100 dark:bg-zinc-900/80 p-2.5 rounded-lg border border-zinc-200/60 dark:border-zinc-800/60">
              Similarity Search → Top 4 Chunks → Citation Grounding
            </div>
          </div>
        </div>

        {/* Right 2 Columns: Smart Filing Summary Cards & 1-Click Action Chips */}
        <div className="lg:col-span-2 space-y-4">
          <div className="text-sm font-semibold text-zinc-900 dark:text-zinc-100 mb-1">
            Featured Financial Filings
          </div>

          {SAMPLE_REPORTS.map((report) => (
            <div key={report.ticker} className="fin-card p-5 sm:p-6 hover:shadow-md transition-all">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 pb-3 mb-3 border-b border-zinc-100 dark:border-zinc-800/80">
                <div className="flex items-center gap-2.5">
                  <span className="font-mono font-bold text-sm text-indigo-600 dark:text-indigo-400 bg-indigo-500/10 px-2 py-0.5 rounded border border-indigo-500/20">
                    {report.ticker}
                  </span>
                  <span className="text-sm font-semibold text-zinc-900 dark:text-zinc-100">
                    {report.title}
                  </span>
                </div>
                <div className="flex items-center gap-2 text-xs font-mono text-zinc-400">
                  <span>{report.type}</span>
                  <span>•</span>
                  <span>{report.date}</span>
                  <span>•</span>
                  <span>{report.pages} pages</span>
                </div>
              </div>

              {/* KPI Badges */}
              <div className="flex flex-wrap gap-2 mb-4">
                {report.kpis.map((kpi, j) => (
                  <span
                    key={j}
                    className="text-xs font-mono px-2.5 py-1 rounded-lg bg-zinc-100 dark:bg-zinc-900 text-zinc-700 dark:text-zinc-300 border border-zinc-200/60 dark:border-zinc-800/60"
                  >
                    {kpi}
                  </span>
                ))}
              </div>

              {/* 1-Click Action Prompts */}
              <div>
                <div className="text-[11px] font-medium text-zinc-400 uppercase tracking-wider mb-2">
                  One-Click Prompts
                </div>
                <div className="flex flex-wrap gap-2">
                  {report.prompts.map((p, k) => (
                    <button
                      key={k}
                      onClick={onOpenCopilot}
                      className="inline-flex items-center gap-1.5 text-xs font-medium px-3 py-1.5 rounded-lg bg-white dark:bg-zinc-900 text-zinc-700 dark:text-zinc-300 border border-zinc-200 dark:border-zinc-800 hover:border-indigo-500 hover:text-indigo-600 dark:hover:text-indigo-400 transition-colors shadow-sm"
                    >
                      <Sparkles className="w-3 h-3 text-indigo-500" />
                      <span>{p}</span>
                    </button>
                  ))}
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
