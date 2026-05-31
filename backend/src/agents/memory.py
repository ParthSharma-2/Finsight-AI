import os
import sys
sys.path.insert(0, os.path.abspath("."))
import json
from datetime import datetime
from pathlib import Path

MEMORY_PATH = "data/processed/conversation_memory.json"

def load_memory() -> list:
    path = Path(MEMORY_PATH)
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return []

def save_memory(memory: list):
    Path(MEMORY_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(MEMORY_PATH, "w") as f:
        json.dump(memory, f, indent=2)

def add_to_memory(question: str, answer: str):
    memory = load_memory()
    memory.append({
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "answer": answer
    })
    if len(memory) > 20:
        memory = memory[-20:]
    save_memory(memory)
    print(f"Memory saved. Total entries: {len(memory)}")

def get_recent_context(n: int = 5) -> str:
    memory = load_memory()
    recent = memory[-n:] if len(memory) >= n else memory
    if not recent:
        return "No previous conversation history."
    context = "\n".join([
        f"Q: {entry['question']}\nA: {entry['answer'][:200]}..."
        for entry in recent
    ])
    return context

def clear_memory():
    save_memory([])
    print("Memory cleared.")

if __name__ == "__main__":
    add_to_memory(
        "What was Infosys revenue?",
        "Infosys total revenue was ₹153,670 crore in FY2024."
    )
    print("\nRecent context:")
    print(get_recent_context())