# FinSight AI

An agentic financial intelligence system combining RAG, multi-agent 
orchestration, and live market data analysis.

## Architecture
PDF Documents → Chunking → Embeddings → ChromaDB
↓
User Query → LangGraph Agent → RAG Tool ──→ Context
→ Market Tool → Live Prices
→ SQL Tool  → Financial DB
→ News Tool → Headlines
↓
phi (Ollama) → Answer

## Tech Stack

| Component | Technology |
|-----------|------------|
| LLM | Ollama phi (local, free) |
| Embeddings | all-MiniLM-L6-v2 (local, free) |
| Vector DB | ChromaDB |
| Agent Framework | LangGraph |
| Market Data | yfinance |
| Backend API | FastAPI |
| Frontend | Streamlit |

## Setup

```bash
git clone https://github.com/ParthSharma-2/finsight-ai
cd finsight-ai
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
ollama pull phi
```

## Run

```bash
# Terminal 1
ollama serve

# Terminal 2 — Streamlit UI
streamlit run app.py

# Terminal 3 — FastAPI (optional)
uvicorn src.api.main:app --reload
```

## Key Features

- RAG pipeline with local embeddings — no API costs
- Multi-agent orchestration via LangGraph
- Live stock prices and market data
- Financial database with SQL interface
- Conversation memory across sessions
- FastAPI backend for production deployment
- RAGAS evaluation harness (evals/)

## Resume Line

Built FinSight AI — an agentic financial intelligence system using 
RAG (ChromaDB + HuggingFace embeddings), LangGraph multi-agent 
orchestration, and live market data integration. Fully local stack 
(Ollama phi LLM) with FastAPI backend and Streamlit UI.
