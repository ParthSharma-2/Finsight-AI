from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.agents.graph import run_agent

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
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
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