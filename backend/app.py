import os
import shutil
import time
import traceback

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import yfinance as yf

from src.agents.graph import run_agent
from src.rag.qa_chain import answer_question
from src.rag.ingestion import ingest_document

# =========================================
# FastAPI App Initialization
# =========================================

app = FastAPI(title="FinSight AI API")

# =========================================
# CORS Configuration
# =========================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8080",
        "https://finsight-ai-fawn.vercel.app",
        "https://www.finsightai.space",
        "https://finsightai.space",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================
# In-Memory Cache for Market Data (TTL)
# =========================================

_market_cache = {}

def get_cached(key, ttl=60):
    if key in _market_cache:
        val, ts = _market_cache[key]
        if time.time() - ts < ttl:
            return val
    return None

def set_cached(key, val):
    _market_cache[key] = (val, time.time())

# =========================================
# Request Schemas
# =========================================

class ChatRequest(BaseModel):
    query: str

class ResearchRequest(BaseModel):
    query: str

# =========================================
# Endpoints
# =========================================

@app.get("/")
def home():
    return {
        "message": "FinSight AI Backend Running"
    }

@app.get("/test")
def test():
    return {
        "status": "success",
        "message": "Frontend connected successfully"
    }

# =========================================
# Live Market Data Endpoints
# =========================================

@app.get("/market/quote")
def get_market_quote(symbol: str):
    sym = symbol.strip().upper()
    cache_key = f"quote_{sym}"
    cached = get_cached(cache_key, ttl=60)
    if cached:
        return cached

    try:
        t = yf.Ticker(sym)
        hist = t.history(period="5d")
        if hist.empty:
            raise HTTPException(status_code=404, detail=f"No quote data found for {sym}")

        curr = float(hist['Close'].iloc[-1])
        prev = float(hist['Close'].iloc[-2]) if len(hist) > 1 else curr
        chg = curr - prev
        chg_pct = (chg / prev) * 100 if prev else 0.0
        vol = int(hist['Volume'].iloc[-1]) if 'Volume' in hist else 0

        info = {}
        try:
            info = t.info or {}
        except Exception:
            pass

        data = {
            "symbol": sym,
            "price": round(curr, 2),
            "change": round(chg, 2),
            "change_pct": round(chg_pct, 2),
            "volume": vol,
            "fifty_two_week_high": info.get("fiftyTwoWeekHigh"),
            "fifty_two_week_low": info.get("fiftyTwoWeekLow"),
            "pe_ratio": info.get("trailingPE"),
            "market_cap": info.get("marketCap"),
        }
        set_cached(cache_key, data)
        return data
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch quote: {str(e)}")


@app.get("/market/chart")
def get_market_chart(symbol: str, period: str = "1mo"):
    sym = symbol.strip().upper()
    period_map = {
        "1w": "5d",
        "1m": "1mo",
        "3m": "3mo",
        "6m": "6mo",
        "1y": "1y",
    }
    yf_period = period_map.get(period.lower(), period.lower())
    cache_key = f"chart_{sym}_{yf_period}"
    cached = get_cached(cache_key, ttl=120)
    if cached:
        return cached

    try:
        t = yf.Ticker(sym)
        hist = t.history(period=yf_period)
        if hist.empty:
            raise HTTPException(status_code=404, detail=f"No historical chart data for {sym}")

        points = []
        for idx, row in hist.iterrows():
            points.append({
                "date": idx.strftime("%b %d"),
                "price": round(float(row["Close"]), 2),
                "volume": int(row["Volume"]) if "Volume" in row else 0,
            })

        first_p = points[0]["price"] if points else 0
        last_p = points[-1]["price"] if points else 0
        chg = last_p - first_p
        chg_pct = round((chg / first_p) * 100, 2) if first_p else 0.0

        data = {
            "symbol": sym,
            "period": period,
            "points": points,
            "first_price": first_p,
            "last_price": last_p,
            "change": round(chg, 2),
            "change_pct": chg_pct,
        }
        set_cached(cache_key, data)
        return data
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch chart: {str(e)}")


# =========================================
# Chat & RAG Endpoints
# =========================================

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        result = run_agent(request.query)
        return {
            "response": result["answer"]
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@app.post("/research/query")
def query_document(request: ResearchRequest):
    try:
        result = answer_question(request.query)
        return {
            "status": "success",
            "answer": result["answer"]
        }
    except Exception as e:
        traceback.print_exc()
        return {
            "status": "error",
            "message": str(e)
        }

UPLOAD_DIR = "data/raw"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/research/upload")
async def upload_document(file: UploadFile = File(...)):
    try:
        file_path = os.path.join(
            UPLOAD_DIR,
            file.filename
        )
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        ingest_document(file_path)

        return {
            "status": "success",
            "filename": file.filename,
            "message": "Document uploaded and indexed successfully"
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
