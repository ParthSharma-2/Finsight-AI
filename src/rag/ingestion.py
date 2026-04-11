import warnings
warnings.filterwarnings("ignore")
import os
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
load_dotenv()

CHROMA_PATH = "data/processed/chroma_db"
DATA_PATH = "data/raw"

EMBEDDING_MODEL = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": "cpu"}
)

def load_pdf(file_path: str):
    print(f"Loading PDF: {file_path}")
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    print(f"Loaded {len(documents)} pages")
    return documents

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")
    return chunks

def create_vectorstore(chunks):
    print("Creating embeddings locally — no rate limits...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=EMBEDDING_MODEL,
        persist_directory=CHROMA_PATH
    )
    print(f"Vectorstore created at {CHROMA_PATH}")
    return vectorstore

def load_vectorstore():
    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=EMBEDDING_MODEL
    )
    return vectorstore

def ingest_document(file_path: str):
    documents = load_pdf(file_path)
    chunks = split_documents(documents)
    vectorstore = create_vectorstore(chunks)
    return vectorstore

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python ingestion.py <path_to_pdf>")
        sys.exit(1)
    ingest_document(sys.argv[1])