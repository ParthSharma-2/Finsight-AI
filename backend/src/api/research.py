from fastapi import APIRouter, UploadFile, File
import shutil
import os

from src.rag.ingestion import ingest_document

router = APIRouter(prefix="/research")

UPLOAD_DIR = "data/raw"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):

    filepath = f"{UPLOAD_DIR}/{file.filename}"

    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    ingest_document(filepath)

    return {
        "status": "success",
        "filename": file.filename
    }