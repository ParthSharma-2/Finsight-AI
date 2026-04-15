# 🚀 FinSight AI  
### Agentic Financial Intelligence System (RAG + Multi-Agent + Live Data)

> **Built for next-gen finance + AI roles (Quant / Data / AI Engineering)**

---

## 🧠 Overview

**FinSight AI** is an **agentic financial intelligence system** that combines:

- Retrieval-Augmented Generation (RAG)
- Multi-agent orchestration
- Real-time market data
- Financial database querying

It enables users to ask complex financial questions and receive **context-aware, data-driven answers** — all powered by a **fully local AI stack (zero API cost)**.

---

## ⚡ Key Highlights

- 🧩 **Multi-Agent System (LangGraph)** → intelligent tool selection  
- 📊 **Live Market Data Integration** → real-time financial insights  
- 📚 **RAG Pipeline (ChromaDB)** → document-grounded responses  
- 🧠 **Local LLM (Ollama - phi)** → no dependency on paid APIs  
- 💬 **Conversational Memory** → retains user context  
- 🏗 **Production Ready Backend** → FastAPI  
- 🎯 **Evaluation Framework** → RAGAS metrics  

---

# 🧠 FinSight AI Architecture

## 📄 Data Ingestion Pipeline
PDF Documents  
→ Chunking  
→ Embeddings (BAAI/bge-small-en-v1.5)  
→ ChromaDB (Vector Store)

---

## 🔍 Query Processing Pipeline

User Query  
↓  
LangGraph Agent  
├── 📚 RAG Tool → Retrieves Context from ChromaDB  
├── 📈 Market Tool → Fetches Live Prices  
├── 🗄️ SQL Tool → Queries Financial Database  
└── 📰 News Tool → Fetches Latest Headlines  
↓  
🧠 LLM (phi via Ollama)  
↓  
💬 Final Answer

---

## ⚙️ Full Flow (Compact View)

PDF → Chunk → Embed → Store (ChromaDB)  
↓  
User Query  
↓  
Agent (LangGraph)  
↓  
[ RAG | Market | SQL | News ]  
↓  
LLM (Ollama - phi)  
↓  
Answer


---

## Project Structure

finsight-ai/

├── src/

│   ├── rag/

│   │   ├── ingestion.py        # PDF loading, chunking, embedding, ChromaDB

│   │   ├── retriever.py        # Semantic search and context formatting

│   │   └── qa_chain.py         # RAG chain with Ollama phi

│   ├── agents/

│   │   ├── graph.py            # LangGraph agent with StateGraph

│   │   ├── tools.py            # Agent tool definitions

│   │   └── memory.py           # Persistent conversation memory

│   ├── tools/

│   │   ├── market.py           # Live stock prices via yfinance

│   │   ├── news.py             # Company news and analyst recommendations

│   │   ├── sql_tool.py         # SQLite financial database + queries

│   │   └── calculator.py       # Financial ratio and growth calculations

│   └── api/

│       └── main.py             # FastAPI REST API

├── data/

│   ├── raw/                    # Source PDF documents

│   └── processed/

│       ├── chroma_db/          # ChromaDB vector store

│       └── finsight.db         # SQLite financial database

├── evals/                      # RAGAS evaluation harness

├── notebooks/                  # Experimentation notebooks

├── app.py                      # Streamlit UI entry point

├── requirements.txt

└── README.md

---

## Quick Start

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.ai) installed

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/ParthSharma-2/finsight-ai.git
cd finsight-ai

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Pull the LLM
ollama pull phi

# 5. Ingest a financial document
python src/rag/ingestion.py data/raw/your_report.pdf
```

### Environment Setup

Create a `.env` file in the root directory:

```env
# Optional — only needed if using cloud APIs
GOOGLE_API_KEY=your_key_here
TAVILY_API_KEY=your_key_here
```

---

## Usage

### Run the Full System

```bash
# Terminal 1 — Start Ollama
ollama serve

# Terminal 2 — Launch Streamlit UI
streamlit run app.py

# Terminal 3 — Launch FastAPI (optional)
uvicorn src.api.main:app --reload
```

Open `http://localhost:8501` in your browser.

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | System info and available endpoints |
| `GET` | `/health` | Health check |
| `POST` | `/rag/ask` | Ask a question against the document |
| `POST` | `/agent/run` | Run the full multi-agent pipeline |
| `POST` | `/market/stock` | Get live stock price |
| `POST` | `/db/query` | Execute SQL on financial database |
| `GET` | `/memory/recent` | Retrieve conversation history |

**Example API call:**

```bash
curl -X POST "http://localhost:8000/rag/ask" \
     -H "Content-Type: application/json" \
     -d '{"question": "What was Infosys revenue in 2024?", "k": 5}'
```

### RAG Pipeline Only

```python
from src.rag.qa_chain import answer_question

result = answer_question("What was Infosys revenue in FY2024?")
print(result["answer"])
print(f"Sources: {result['num_sources']} chunks cited")
```

### Agent Pipeline

```python
from src.agents.graph import run_agent

result = run_agent("Compare Infosys 2024 revenue with current stock price")
print(result["answer"])
```

---

## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **LLM** | Ollama phi | Local inference, zero cost |
| **Embeddings** | all-MiniLM-L6-v2 | Semantic search, 384-dim vectors |
| **Vector Store** | ChromaDB | Persistent document embeddings |
| **Agent Framework** | LangGraph | Stateful multi-agent orchestration |
| **RAG Framework** | LangChain | Document loading, retrieval chains |
| **Market Data** | yfinance | Live stock prices and fundamentals |
| **Database** | SQLite | Financial metrics and breakdowns |
| **API** | FastAPI | Production REST backend |
| **UI** | Streamlit | Interactive frontend dashboard |
| **Evaluation** | RAGAS | RAG faithfulness and relevance scoring |

---

## Evaluation

The system uses RAGAS to evaluate RAG quality:

```bash
python -m evals.ragas_eval
```

Metrics tracked:
- **Faithfulness** — Are answers grounded in the retrieved context?
- **Answer Relevance** — Does the answer address the question?
- **Context Precision** — Are the retrieved chunks actually useful?
- **Context Recall** — Did retrieval find all relevant information?

---

## Roadmap

- [x] RAG pipeline with local embeddings
- [x] ChromaDB vector store with persistence
- [x] LangGraph multi-agent orchestration
- [x] Live market data integration
- [x] Financial SQL database
- [x] Conversation memory
- [x] FastAPI REST backend
- [x] Streamlit UI
- [ ] RAGAS evaluation harness (in progress)
- [ ] Docker deployment
- [ ] Multi-document support
- [ ] Reranking with cross-encoder

---

## Author

**Parth Sharma**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/parth-sharma-work)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=for-the-badge&logo=github)](https://github.com/ParthSharma-2)

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

**Built with LangChain · LangGraph · Ollama · ChromaDB · FastAPI · Streamlit**

*If this project helped you, please give it a star*

</div>
