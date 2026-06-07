import os
import sys

sys.path.insert(0, os.path.abspath("."))

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

load_dotenv()

CHROMA_PATH = "data/processed/chroma_db"

_embedding_model = None


def get_embedding_model():
    global _embedding_model

    if _embedding_model is None:
        print("Loading embedding model...")

        _embedding_model = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={"device": "cpu"}
        )

    return _embedding_model


def get_vectorstore():

    embeddings = get_embedding_model()

    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )

    return vectorstore


def retrieve_documents(query: str, k: int = 5) -> list[Document]:

    vectorstore = get_vectorstore()

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )

    docs = retriever.invoke(query.strip())

    return docs


def format_context(docs: list[Document]) -> str:

    context = "\n\n---\n\n".join([
        f"Source (Page {doc.metadata.get('page', 'unknown')}):\n{doc.page_content}"
        for doc in docs
    ])

    return context