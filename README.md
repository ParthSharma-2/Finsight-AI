# FinSight AI 🧠📈

> **A multi-agent financial intelligence platform powered by LangGraph, Groq LLM, and RAG — deployed on AWS EC2.**

[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-Vite-61DAFB?style=flat-square&logo=react)](https://vitejs.dev/)
[![LangGraph](https://img.shields.io/badge/LangGraph-multi--agent-FF6B35?style=flat-square)](https://github.com/langchain-ai/langgraph)
[![Groq](https://img.shields.io/badge/Groq-llama3--70b-F55036?style=flat-square)](https://groq.com/)
[![AWS EC2](https://img.shields.io/badge/AWS-EC2%20t3.micro-FF9900?style=flat-square&logo=amazonaws)](https://aws.amazon.com/ec2/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-RAG-8B5CF6?style=flat-square)](https://www.trychroma.com/)
[![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)](LICENSE)

---

## What is FinSight AI?

FinSight AI is a conversational financial research assistant that combines the reasoning power of large language models with real-time market data and document-level retrieval. Ask it about stock prices, compare companies, or query your own financial research documents — all through a single chat interface.

The system uses a **LangGraph multi-agent orchestration layer** to route user queries to the right tool automatically: live stock data via yFinance, vector search over uploaded PDFs via ChromaDB, or direct LLM reasoning via the Groq API.

---

## Architecture

```
Browser (User)
    ↓
React + Vite Frontend        ← api.js handles all HTTP calls
    ↓  POST /chat, GET /market/*, POST /research/*
FastAPI Backend (main.py)    ← Receives, validates, dispatches
    ↓
LangGraph Agent (graph.py)   ← Routes to correct tool based on intent
    ├── Groq LLM             ← llama3-70b-8192 reasoning
    ├── yFinance Tools       ← Live price, history, stock comparison
    ├── ChromaDB RAG         ← PDF document retrieval + vector search
    └── SQL / Data Tools     ← Structured query over datasets
    ↓
JSON Response → Frontend renders chat reply, stock cards, charts
```

Deployed on **AWS EC2 (Ubuntu 24.04 LTS, t3.micro, ap-south-1)** with Nginx as the reverse proxy and Let's Encrypt for SSL.

---

## Features

| Feature | Description | Status |
|---|---|---|
| **Multi-agent orchestration** | LangGraph routes queries to the right tool automatically | ✅ Complete |
| **Real-time stock data** | Live price, daily change %, volume, P/E, EPS, 52-week range | ✅ Complete |
| **Stock history & comparison** | OHLCV history for any period; side-by-side multi-stock comparison | ✅ Complete |
| **RAG over financial documents** | Upload PDFs; ChromaDB embeds and retrieves relevant chunks | ✅ Complete |
| **Groq LLM reasoning** | llama3-70b-8192 for fast, accurate financial Q&A | ✅ Complete |
| **REST API** | Full FastAPI backend with typed endpoints | ✅ Complete |
| **React frontend** | Chat UI with stock cards and chart rendering | ✅ Complete |
| **AWS EC2 deployment** | Ubuntu 24.04, t3.micro, 20GB gp3 EBS | ✅ Live |
| **Nginx reverse proxy** | Routes `/` → React, `/api/*` → FastAPI | 🔄 In progress |
| **HTTPS / SSL** | Let's Encrypt via Certbot | 🔄 Planned |
| **Systemd auto-restart** | Backend survives reboots and crashes | 🔄 Planned |

---

## Tech Stack

- **Backend**: FastAPI + Uvicorn (Python)
- **Frontend**: React + Vite (TypeScript)
- **Agent Orchestration**: LangGraph (multi-agent graph with tool routing)
- **LLM Provider**: Groq API — `llama3-70b-8192` / `mixtral-8x7b`
- **Financial Data**: yFinance (NSE & US markets)
- **Vector Store**: ChromaDB (document embedding + semantic search)
- **Cloud**: AWS EC2 — `t3.micro`, Ubuntu 24.04 LTS, ap-south-1 (Mumbai)
- **Proxy**: Nginx (static serving + API reverse proxy)
- **SSL**: Let's Encrypt via Certbot

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Health check |
| `POST` | `/chat` | Send a message to the LangGraph agent |
| `GET` | `/market/quote?symbol=INFY.NS` | Live stock quote |
| `GET` | `/market/chart?symbol=TCS.NS&period=1mo` | OHLCV chart data |
| `GET` | `/market/search?q=apple` | Ticker symbol search |
| `POST` | `/research/query` | RAG query over uploaded documents |
| `POST` | `/research/upload` | Upload a PDF for embedding |

**Indian stocks**: append `.NS` (e.g. `INFY.NS`, `TCS.NS`, `WIPRO.NS`)
**US stocks**: plain ticker (e.g. `MSFT`, `GOOGL`, `AAPL`)

---

## Getting Started

### Prerequisites

- Python 3.11+
- Node.js 22 LTS
- A [Groq API key](https://console.groq.com/) (free tier available)

### Local Development

```bash
# Clone the repo
git clone https://github.com/ParthSharma-2/FinSight-AI.git
cd FinSight-AI

# Backend
cd backend
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Set environment variables
cp .env.example .env
# Add your GROQ_API_KEY to .env

# Start the backend
uvicorn main:app --reload --port 8000

# Frontend (new terminal)
cd ../frontend
npm install
npm run dev
```

Visit `http://localhost:5173` — the frontend is live and connected to the backend.

### AWS EC2 Deployment

See [`docs/AWS_DEPLOYMENT.md`](docs/AWS_DEPLOYMENT.md) for the full 8-phase deployment guide:

1. EC2 provisioning (AMI, instance type, storage, security groups)
2. SSH access + PEM key setup (Windows icacls fix included)
3. Server setup (apt, Git, Python, Node.js)
4. Backend deployment (venv, pip, uvicorn)
5. Frontend build + Nginx static serving
6. Nginx reverse proxy configuration
7. Elastic IP + domain + HTTPS (Let's Encrypt)
8. Production hardening + systemd auto-restart

---

## Project Structure

```
FinSight-AI/
├── backend/
│   ├── main.py          # FastAPI app — all endpoints
│   ├── graph.py         # LangGraph agent + tool routing
│   ├── market.py        # yFinance tools (@tool decorated)
│   ├── rag.py           # ChromaDB ingestion + retrieval
│   ├── requirements.txt
│   └── .env.example
├── frontend/
│   ├── src/
│   │   ├── api.js       # Centralized API fetch wrapper
│   │   ├── pages/       # Chat, Home, Research pages
│   │   └── components/  # StockCard, ChatBubble, ChartView
│   ├── package.json
│   └── vite.config.ts
├── data/
│   └── chroma/          # ChromaDB vector store (local)
├── docs/
│   └── AWS_DEPLOYMENT.md
└── README.md
```

---

## Financial Tools (market.py)

Three `@tool`-decorated functions, callable by the LangGraph agent:

**`get_stock_price(ticker)`** — Returns live price, daily change %, volume, 52-week high/low, market cap, P/E ratio, EPS, and dividend yield.

**`get_stock_history(ticker, period)`** — Returns start/end price, total return %, period high/low, and trading day count for any period (1d to 5y).

**`compare_stocks(tickers)`** — Accepts a comma-separated list of tickers and returns a side-by-side table of price, 1-month return %, P/E ratio, and market cap.

---

## AWS Cost Summary

Running this on AWS with free-tier eligibility:

| Resource | Monthly Cost |
|---|---|
| EC2 t3.micro (750 hrs/month) | **Free** (12 months) |
| EBS 20GB gp3 | **Free** (30GB limit) |
| Data transfer (first 100GB) | **Free** |
| Elastic IP (when attached) | **Free** |
| Let's Encrypt SSL | **Free** |
| Groq API | **Free tier** |
| Custom domain (optional) | ~$1/month |

---

## Roadmap

- [ ] Phase 3–8: Full production deployment on EC2
- [ ] RAGAS evaluation of the RAG pipeline
- [ ] CloudWatch monitoring + alerting
- [ ] Portfolio tracking with persistent user sessions
- [ ] Earnings call transcript ingestion
- [ ] Options chain data integration
- [ ] Custom domain + HTTPS live URL

---

## Author

**Parth Sharma**

Built as a portfolio project demonstrating end-to-end ML system design: multi-agent LLM orchestration, RAG pipelines, financial data engineering, REST API development, and cloud deployment on AWS.

---

*FinSight AI is a portfolio project and does not constitute financial advice.*
