from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


app = FastAPI()

origins = [
    "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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



class ChatRequest(BaseModel):
    query: str


@app.post("/chat")
def chat(request: ChatRequest):

    user_query = request.query

    response = f"AI Response to: {user_query}"

    return {
        "response": response
    }