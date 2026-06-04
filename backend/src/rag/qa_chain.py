import os
import sys

sys.path.insert(0, os.path.abspath("."))

from dotenv import load_dotenv
from langchain_groq import ChatGroq

from src.rag.retriever import (
    retrieve_documents,
    format_context
)

load_dotenv()


def get_llm():

    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.1,
        groq_api_key=os.getenv("GROQ_API_KEY")
    )


def answer_question(question: str, k: int = 5) -> dict:

    print(f"Retrieving relevant context for: {question}")

    docs = retrieve_documents(question, k=k)

    docs = docs[:5]

    context = format_context(docs)

    print("\n--- CONTEXT ---\n", context[:1000])

    prompt = f"""
You are FinSight AI, a senior equity research analyst.

Instructions:
- Use ONLY the provided context.
- Extract exact numbers whenever available.
- Mention evidence from the document.
- Be concise and professional.
- If information is unavailable, clearly say:
  "Not available in the provided document."

Context:
{context}

Question:
{question}

Answer:
"""

    llm = get_llm()

    response = llm.invoke(prompt)

    return {{
        "question": question,
        "answer": response.content,
        "source_documents": docs,
        "num_sources": len(docs)
    }}


if __name__ == "__main__":

    questions = [
        "What was Infosys total revenue in 2024?",
        "Who is the CEO of Infosys?",
        "What are the main business segments of Infosys?"
    ]

    for q in questions:

        print(f"\n{{'='*60}}")

        result = answer_question(q)

        print(f"Q: {{result['question']}}")
        print(f"A: {{result['answer']}}")
        print(f"Sources used: {{result['num_sources']}} chunks")