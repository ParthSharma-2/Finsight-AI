# FinSight AI 🧠📈

> **An institutional-grade multi-agent financial intelligence and equity research platform powered by LangGraph, Groq LLM, and RAG — deployed in production on AWS EC2 & Vercel.**

[![Live Web App](https://img.shields.io/badge/Live_App-finsightai.space-00C7B7?style=flat-square&logo=vercel)](https://finsightai.space)
[![Live API](https://img.shields.io/badge/Live_API-api.finsightai.space-FF9900?style=flat-square&logo=amazonaws)](https://api.finsightai.space)
[![RAGAS Benchmark](https://img.shields.io/badge/RAGAS_Benchmark-98.6%25_Score-brightgreen?style=flat-square)](evals/RAGAS_BENCHMARK_REPORT.md)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React_18-Vite_Tailwind-61DAFB?style=flat-square&logo=react)](https://vitejs.dev/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Multi--Agent-FF6B35?style=flat-square)](https://github.com/langchain-ai/langgraph)
[![Groq](https://img.shields.io/badge/Groq-GPT--OSS_120B-F55036?style=flat-square)](https://groq.com/)
[![AWS EC2](https://img.shields.io/badge/AWS-EC2_Ubuntu_24.04-FF9900?style=flat-square&logo=amazonaws)](https://aws.amazon.com/ec2/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-8B5CF6?style=flat-square)](https://www.trychroma.com/)
[![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)](LICENSE)

---

## 🌐 Live Deployments

- **Frontend Application:** [https://finsightai.space](https://finsightai.space) (Vercel Edge Network)
- **Production API Backend:** [https://api.finsightai.space](https://api.finsightai.space) (AWS EC2 • Nginx Reverse Proxy • SSL)
- **API Health Check:** [https://api.finsightai.space/health](https://api.finsightai.space/health)

---

## What is FinSight AI?

FinSight AI is a comprehensive financial intelligence and equity research platform designed to bring institutional-grade analysis to retail investors and analysts. It unifies high-speed market data, multi-perspective LLM reasoning, and document-level retrieval (RAG) over corporate filings.

Rather than relying on a single monolithic prompt, FinSight AI utilizes a **LangGraph multi-agent orchestration architecture**: queries are routed across specialized agents (Market Analyst, Valuation Specialist, Risk Assessor, and Synthesis Agent) equipped with dedicated financial tools:
- **Live Market Data:** Real-time quotes, batch tickers, and historical OHLCV curves via `yfinance` with in-memory TTL caching.
- **RAG Filing Search:** Semantic retrieval over 10-Ks, annual reports, and earnings filings using ChromaDB and BGE embeddings.
- **Quantitative Database:** Direct SQL queries over financial statements and fundamental metrics.
- **LLM Reasoning:** Ultra-low latency synthesis powered by Groq's `openai/gpt-oss-120b` (131K context window).

---

## 🏛️ System Architecture

```
User (Browser)
   │
   ├── [Vercel Edge] ──> React 18 + Tailwind Institutional Workspace
   │                     ├── Dark Zinc (#09090b) / Light Slate Theme Toggle (300ms easing)
   │                     ├── Global AI Copilot Drawer (Cmd/Ctrl + K)
   │                     └── Interactive Financial Terminal & Dynamic Watchlist
   │
   └── [AWS EC2] ──────> Nginx Reverse Proxy (SSL Let's Encrypt)
                            │
                            ▼
                         FastAPI Backend (Uvicorn / systemd service)
                            │
                            ▼
                   LangGraph Multi-Agent Orchestrator
           ┌────────────────────────┼────────────────────────┐
           ▼                        ▼                        ▼
     Market Analyst        Valuation Specialist        Risk Assessor
           │                        │                        │
     yFinance Tools        Financial SQL DB           ChromaDB RAG
   (Quotes / History)     (Fundamental Metrics)   (PDF Filings Search)
           └────────────────────────┬────────────────────────┘
                                    │
                                    ▼
                             Synthesis Agent
                       (Groq openai/gpt-oss-120b)
                                    │
                                    ▼
                         Final Research Report
```

---

## 📊 RAGAS Evaluation Benchmark

The RAG retrieval and synthesis pipeline was rigorously evaluated using the **RAGAS (Retrieval-Augmented Generation Assessment)** framework against ground-truth corporate filings (including Infosys FY24 Annual Report and Ujjivan SFB filings).

👉 **[Read the Full RAGAS Evaluation Report (Scorecard & Methodology)](evals/RAGAS_BENCHMARK_REPORT.md)**

### Benchmark Scorecard

| Metric | Score | Target | Production Assessment |
| :--- | :---: | :---: | :--- |
| **Faithfulness** | **100.00%** | >85.0% | **Zero Hallucinations** (Strictly grounded in cited context) |
| **Answer Relevancy** | **94.55%** | >85.0% | **High Precision** (Directly addresses user inquiries without fluff) |
| **Context Precision** | **100.00%** | >80.0% | **Exact Grounding** (Pinpoint statement & page-level citations) |
| **Context Recall** | **100.00%** | >80.0% | **Complete** (All key metrics & currency amounts captured) |
| **Composite RAGAS Score** | **98.58%** | >80.0% | **Production Grade** |
| **Average Query Latency** | **1.18s** | <3.0s | Sub-second to 1.6s end-to-end response times |

### Run the Benchmark Yourself

```bash
# Run against the live production backend
python evals/ragas_benchmark.py

# Or run against your local development server
FINSIGHT_API_URL="http://localhost:8000/api" python evals/ragas_benchmark.py
```

---

## ⚡ Key Features

| Capability | Technical Implementation | Status |
|---|---|:---:|
| **Multi-Agent Orchestration** | LangGraph state graph routing across 4 specialized agents | ✅ Complete |
| **Live Market Quotes & Charts** | `yfinance` with in-memory TTL caching (US & Indian NSE `.NS` tickers) | ✅ Complete |
| **RAG over Financial Filings** | ChromaDB + `BAAI/bge-small-en-v1.5` embeddings with source page citations | ✅ Complete |
| **Anti-Hallucination Guard** | Fallback validation rejecting out-of-scope queries; empty PDF upload guard | ✅ Complete |
| **Institutional UI/UX** | Dark Zinc (`#09090b`) / Light Slate theme toggle with smooth 300ms easing | ✅ Complete |
| **AI Copilot Drawer** | Persistent slide-out intelligence panel with global `Cmd/Ctrl + K` shortcut | ✅ Complete |
| **Dynamic Terminal Watchlist** | Live multi-ticker quotes with real-time add/remove ticker support | ✅ Complete |
| **AWS EC2 Production Server** | Ubuntu 24.04 LTS, Nginx reverse proxy, systemd auto-restart, Let's Encrypt SSL | ✅ Live |
| **Vercel Edge Frontend** | Production CDN deployment with automatic CI/CD builds | ✅ Live |

---

## 🛠️ Tech Stack

- **Frontend**: React 18, Vite, Tailwind CSS, Lucide Icons, Recharts
- **Backend**: FastAPI, Uvicorn, Python 3.11
- **Agent Orchestration**: LangGraph, LangChain Core
- **LLM Provider**: Groq Cloud API — `openai/gpt-oss-120b` (131K context)
- **Vector Database**: ChromaDB (persistent vector store)
- **Embedding Model**: `BAAI/bge-small-en-v1.5` / HuggingFace Transformers
- **Financial APIs**: `yfinance` (NSE `.NS` & US equity markets)
- **Evaluation**: RAGAS Benchmark Suite (Faithfulness, Relevancy, Precision, Recall)
- **Cloud Infrastructure**: AWS EC2 (t3.micro, ap-south-1), Vercel Edge, Let's Encrypt SSL, Nginx

---

## 🔌 API Reference

### Core Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Server status and health check |
| `POST` | `/chat` | Conversational query to the LangGraph agent |
| `POST` | `/analyze` | Multi-agent equity research synthesis |
| `GET` | `/market/quote?symbol=INFY.NS` | Real-time quote (price, change, volume, P/E) |
| `GET` | `/market/quotes?symbols=AAPL,MSFT,NVDA` | Batch real-time quotes for watchlists |
| `GET` | `/market/chart?symbol=TCS.NS&period=1mo` | OHLCV historical candlestick/line data |
| `GET` | `/market/search?query=apple` | Company and ticker search |
| `POST` | `/research/query` | Dedicated RAG query over uploaded financial filings |
| `POST` | `/research/upload` | Upload PDF annual reports / 10-Ks for chunking & embedding |

> **Ticker Conventions:**  
> - **Indian NSE Stocks:** Append `.NS` (e.g. `INFY.NS`, `TCS.NS`, `RELIANCE.NS`)  
> - **US Equities:** Standard ticker symbol (e.g. `AAPL`, `NVDA`, `MSFT`)

---

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- Node.js 20+
- A [Groq API Key](https://console.groq.com/) (Free tier available)

### Local Development

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ParthSharma-2/FinSight-AI.git
   cd FinSight-AI
   ```

2. **Backend Setup:**
   ```bash
   cd backend
   python -m venv venv
   source venv/bin/activate       # On Windows: venv\Scripts\activate
   pip install -r requirements.txt

   # Configure environment variables
   cp .env.example .env
   # Add your GROQ_API_KEY to .env

   # Run FastAPI server
   uvicorn app:app --reload --port 8000
   ```

3. **Frontend Setup (in a new terminal):**
   ```bash
   cd frontend
   npm install
   npm run dev
   ```

4. Open `http://localhost:5173` in your browser.

---

## 📂 Project Structure

```
FinSight-AI/
├── backend/
│   ├── app.py                   # FastAPI application & route declarations
│   ├── src/
│   │   ├── agents/
│   │   │   ├── graph.py         # LangGraph multi-agent workflow
│   │   │   ├── state.py         # Agent state definitions
│   │   │   └── memory.py        # Conversation memory buffer
│   │   ├── rag/
│   │   │   ├── ingestion.py     # Document chunking, validation & ChromaDB upsert
│   │   │   ├── retriever.py     # Semantic search with BGE embeddings
│   │   │   └── qa_chain.py      # Grounded Q&A prompt chain
│   │   └── tools/
│   │       ├── market_tools.py  # yFinance quote, history & compare tools
│   │       ├── database_tool.py # SQLite fundamental financial query tool
│   │       └── rag_tool.py      # search_uploaded_documents agent tool
│   ├── data/
│   │   ├── raw/                 # PDF filing storage
│   │   └── processed/           # ChromaDB vector store & sqlite DB
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── components/          # Nav, Footer, CopilotDrawer, ChartPanel
│   │   ├── context/             # ThemeContext (Dark/Light mode)
│   │   ├── pages/               # Home, Markets, Research, Terminal
│   │   └── lib/api.js           # API client routing to production/localhost
│   ├── package.json
│   └── tailwind.config.js
├── evals/
│   ├── ragas_benchmark.py       # RAGAS evaluation runner
│   ├── ragas_results.json       # Raw benchmark run outputs
│   └── RAGAS_BENCHMARK_REPORT.md# Executive scorecard & qualitative report
├── test_live_suite.py           # End-to-end integration test suite
└── README.md
```

---

## 🗺️ Roadmap & Milestones

- [x] Phase 1–8: Full production deployment on AWS EC2 (`api.finsightai.space`)
- [x] Vercel Edge frontend deployment (`finsightai.space`)
- [x] RAGAS evaluation of the RAG pipeline (Score: **98.6%** — see [evals/RAGAS_BENCHMARK_REPORT.md](evals/RAGAS_BENCHMARK_REPORT.md))
- [x] Institutional Dark / Light workspace redesign with persistent AI Copilot
- [x] Real-time market batch quotes & interactive chart integration via `yfinance`
- [ ] Portfolio tracking with persistent user sessions
- [ ] Earnings call transcript audio ingestion & speaker diarization
- [ ] Options chain analytics & Greeks calculation

---

## 👨‍💻 Author

**Parth Sharma**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/parth-sharma-work/)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white)](https://github.com/ParthSharma-2)

- **LinkedIn:** [linkedin.com/in/parth-sharma-work](https://www.linkedin.com/in/parth-sharma-work/)
- **GitHub:** [github.com/ParthSharma-2](https://github.com/ParthSharma-2)

*Built as an end-to-end ML System Design & Financial AI project demonstrating multi-agent LLM orchestration, RAG pipelines, financial data engineering, and full-stack cloud deployment.*

---

*FinSight AI is a portfolio and research project and does not constitute financial advice.*
