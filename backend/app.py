import os
import shutil

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.agents.graph import run_agent
from src.rag.qa_chain import answer_question
from src.rag.ingestion import ingest_document

# =========================================
# FastAPI App Initialization
# =========================================

app = FastAPI()

# =========================================
# CORS Configuration
# =========================================

origins = [
    "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "https://finsight-ai-fawn.vercel.app",
        "https://www.finsightai.space",
        "https://finsightai.space",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# =========================================
# Request Schema
# =========================================

class ChatRequest(BaseModel):
    query: str

# =========================================
# Root Endpoint
# =========================================

@app.get("/")
def home():
    return {
        "message": "FinSight AI Backend Running"
    }

# =========================================
# Test Endpoint
# =========================================

@app.get("/test")
def test():
    return {
        "status": "success",
        "message": "Frontend connected successfully"
    }

# =========================================
# Chat Endpoint
# =========================================

@app.post("/chat")
async def chat(request: ChatRequest):

    try:

        result = run_agent(request.query)

        return {
            "response": result["answer"]
        }

    except Exception as e:

        return {
            "response": f"Error: {str(e)}"
        }
    
class ResearchRequest(BaseModel):
    query: str

@app.post("/research/query")
def query_document(request: ResearchRequest):

    try:

        result = answer_question(request.query)

        return {
            "status": "success",
            "answer": result["answer"]
        }

    except Exception as e:

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

        return {
            "status": "error",
            "message": str(e)
        }