import os
import sys
from pathlib import Path
from langchain.tools import tool

sys.path.insert(0, os.path.abspath("."))

from src.rag.retriever import retrieve_documents, format_context, CHROMA_PATH


@tool
def search_uploaded_documents(query: str) -> str:
    """Search and retrieve information from uploaded financial filings, annual reports (10-K, 10-Q, 8-K), and earnings documents.
    Use this tool whenever the user asks about information contained in uploaded documents, filings, or company annual reports."""
    try:
        chroma_dir = Path(CHROMA_PATH)
        if not chroma_dir.exists() or not any(chroma_dir.iterdir()):
            return "No documents have been uploaded or indexed yet. Please upload a filing PDF first in the Research Workbench."

        docs = retrieve_documents(query, k=4)
        if not docs:
            return "No relevant sections found in the uploaded documents for this query."

        return format_context(docs)
    except Exception as e:
        return f"Error searching documents: {str(e)}"
