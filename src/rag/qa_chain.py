import os
import sys
sys.path.insert(0, os.path.abspath("."))
from dotenv import load_dotenv
from langchain_ollama import OllamaLLM
from src.rag.retriever import retrieve_documents, format_context


load_dotenv()
def get_llm():
    return OllamaLLM(
        model="phi",
        temperature=0.1,
    )

def answer_question(question: str, k: int = 5) -> dict:
    print(f"Retrieving relevant context for: {question}")
    docs = retrieve_documents(question, k=k)
    docs = docs[:5]  # hard cap
    context = format_context(docs)
    print("\n--- CONTEXT ---\n", context[:1000])

    prompt = f"""
You are a senior equity research analyst.

Rules:
- Use ONLY provided context
- Extract exact numbers
- Mention page numbers
- Show calculations if applicable
- If missing → say "Not available in document"

Context:
{context}

Question: {question}

Answer:
"""

    llm = get_llm()
    answer = llm.invoke(prompt)

    return {
        "question": question,
        "answer": answer,
        "source_documents": docs,
        "num_sources": len(docs)
    }

if __name__ == "__main__":
    questions = [
        "What was Infosys total revenue in 2024?",
        "Who is the CEO of Infosys?",
        "What are the main business segments of Infosys?"
    ]

    for q in questions:
        print(f"\n{'='*60}")
        result = answer_question(q)
        print(f"Q: {result['question']}")
        print(f"A: {result['answer']}")
        print(f"Sources used: {result['num_sources']} chunks")