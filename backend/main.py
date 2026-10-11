"""
FastAPI entrypoint for Knowledge Desk.

Endpoints
---------
POST   /api/upload             Upload and ingest a PDF/TXT/MD file
POST   /api/ask                Ask a question against the ingested knowledge base
GET    /api/documents          List ingested documents + chunk counts
DELETE /api/documents/{doc_id} Remove one document and re-save the index
DELETE /api/documents          Clear the entire knowledge base
GET    /api/health             Basic health/config check

The frontend/ folder is served as static files at "/".
"""
import shutil
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import config
from .rag import RAGPipeline

app = FastAPI(title="Knowledge Desk")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md"}

# The pipeline loads the persisted TF-IDF index eagerly — unlike the
# embeddings version of this project there's no API client to construct
# lazily, so there's no reason to defer it.
pipeline = RAGPipeline()


class AskRequest(BaseModel):
    question: str
    api_key: Optional[str] = None  # Hugging Face token pasted in the sidebar
    top_k: Optional[int] = None


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "hf_key_configured": bool(config.HF_API_KEY),
        "documents_indexed": pipeline.document_count(),
        "chunks_indexed": pipeline.total_chunks(),
    }


@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )

    dest_path = config.UPLOADS_DIR / file.filename
    with dest_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        result = pipeline.ingest_file(dest_path)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "id": result["id"],
        "filename": file.filename,
        "chunks_added": result["chunks_added"],
        "total_documents": pipeline.document_count(),
    }


@app.post("/api/ask")
def ask_question(request: AskRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        return pipeline.ask(request.question, api_key=request.api_key, top_k=request.top_k)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/documents")
def list_documents():
    return {
        "documents": pipeline.list_documents(),
        "total_chunks": pipeline.total_chunks(),
    }


@app.delete("/api/documents/{doc_id}")
def remove_document(doc_id: str):
    if not pipeline.remove_document(doc_id):
        raise HTTPException(status_code=404, detail="Document not found.")
    return {"removed": doc_id, "total_documents": pipeline.document_count()}


@app.delete("/api/documents")
def clear_documents():
    pipeline.clear()
    return {"total_documents": pipeline.document_count()}


# Serve the frontend last so it doesn't shadow the /api routes above.
frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
