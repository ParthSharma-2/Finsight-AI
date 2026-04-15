import os
import sys
sys.path.insert(0, os.path.abspath("."))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from src.rag.qa_chain import answer_question
from src.agents.graph import run_agent
from src.tools.market import get_stock_price
from src.tools.sql_tool import init_database, query_financial_database
from src.agents.memory import add_to_memory, get_recent_context

load_dotenv()
init_database()

app = FastAPI(
    title="FinSight AI API",
    description="Agentic Financial Intelligence System",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class QuestionRequest(BaseModel):
    question: str
    k: int = 5

class AgentRequest(BaseModel):
    question: str

class StockRequest(BaseModel):
    ticker: str

class SQLRequest(BaseModel):
    query: str

@app.get("/")
def root():
    return {
        "name": "FinSight AI",
        "version": "1.0.0",
        "status": "running",
        "endpoints": ["/rag/ask", "/agent/run", "/market/stock", "/db/query", "/memory/recent"]
    }

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/rag/ask")
def ask_document(request: QuestionRequest):
    try:
        result = answer_question(request.question, k=request.k)
        add_to_memory(request.question, result["answer"])
        return {
            "question": result["question"],
            "answer": result["answer"],
            "num_sources": result["num_sources"],
            "sources": [
                {
                    "page": doc.metadata.get("page", "unknown"),
                    "content": doc.page_content[:300]
                }
                for doc in result["source_documents"]
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/agent/run")
def run_agent_endpoint(request: AgentRequest):
    try:
        result = run_agent(request.question)
        add_to_memory(request.question, result["answer"])
        return {
            "question": result["question"],
            "answer": result["answer"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/market/stock")
def get_stock(request: StockRequest):
    try:
        result = get_stock_price.invoke(request.ticker)
        return {"ticker": request.ticker, "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/db/query")
def database_query(request: SQLRequest):
    try:
        result = query_financial_database.invoke(request.query)
        return {"query": request.query, "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/memory/recent")
def recent_memory():
    context = get_recent_context(10)
    return {"memory": context}