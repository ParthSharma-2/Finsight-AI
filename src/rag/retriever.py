import os
import sys
sys.path.insert(0, os.path.abspath("."))
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

load_dotenv()
CHROMA_PATH = "data/processed/chroma_db"

EMBEDDING_MODEL = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": "cpu"}
)

def get_vectorstore():
    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=EMBEDDING_MODEL
    )
    return vectorstore

def retrieve_documents(query: str, k: int = 5) -> list[Document]:
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )
    query = query.strip()
    docs = retriever.invoke(query)
    return docs

def format_context(docs: list[Document]) -> str:
    context = "\n\n---\n\n".join([
        f"Source (Page {doc.metadata.get('page', 'unknown')}):\n{doc.page_content}"
        for doc in docs
    ])
    return context

if __name__ == "__main__":
    query = "What was Infosys revenue in 2024?"
    print(f"Query: {query}\n")
    docs = retrieve_documents(query, k=3)
    print(f"Found {len(docs)} relevant chunks:\n")
    for i, doc in enumerate(docs):
        print(f"--- Chunk {i+1} (Page {doc.metadata.get('page', '?')}) ---")
        print(doc.page_content[:300])
        print()