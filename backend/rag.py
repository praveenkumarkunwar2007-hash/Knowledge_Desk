"""
Ties together ingestion, TF-IDF retrieval, and the Hugging Face chat model
into a single question-answering pipeline.
"""
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import requests

from . import config
from .ingest import load_and_chunk, tokenize
from .tfidf_store import TfidfStore

SYSTEM_PROMPT = """You are an internal knowledge assistant. Answer the user's \
question using ONLY the context passages provided below. Each passage is \
tagged with its source document.

Rules:
- If the answer is not contained in the context, say you don't have enough \
information in the knowledge base to answer, rather than guessing.
- Be concise and direct.
- When you use a passage, mention which source document it came from.
"""


class RAGPipeline:
    def __init__(self):
        self.store = TfidfStore.load()

    # -- ingestion -----------------------------------------------------------
    def ingest_file(self, file_path: Path) -> Dict:
        """Chunk and index a single file. Returns its document id + chunk count."""
        chunks = load_and_chunk(file_path)
        if not chunks:
            raise ValueError(f"No extractable text found in \"{file_path.name}\".")

        doc_id = str(uuid.uuid4())
        added_at = datetime.now(timezone.utc).isoformat()
        self.store.add_document(doc_id, file_path.name, added_at, chunks)
        self.store.save()
        return {"id": doc_id, "chunks_added": len(chunks)}

    def remove_document(self, doc_id: str) -> bool:
        removed = self.store.remove_document(doc_id)
        if removed:
            self.store.save()
        return removed

    def clear(self) -> None:
        self.store.clear()
        self.store.save()

    def list_documents(self) -> List[dict]:
        return self.store.list_documents()

    def document_count(self) -> int:
        return len(self.store)

    def total_chunks(self) -> int:
        return self.store.total_chunks()

    # -- generation ------------------------------------------------------------
    def _call_hf(self, api_key: str, question: str, context_blocks: List[str]) -> str:
        context = "\n\n---\n\n".join(context_blocks)
        user_prompt = f"Context:\n{context}\n\nQuestion: {question}"

        response = requests.post(
            config.HF_API_URL,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            json={
                "model": config.HF_MODEL,
                "max_tokens": 1000,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            },
            timeout=60,
        )

        if not response.ok:
            detail = ""
            try:
                detail = response.json().get("error", {}).get("message", "")
            except Exception:
                pass
            if response.status_code == 401:
                detail = detail or "Invalid Hugging Face API token."
            raise RuntimeError(detail or f"Request failed ({response.status_code})")

        data = response.json()
        choices = data.get("choices") or []
        text = (choices[0].get("message", {}).get("content") if choices else "") or ""
        return text or "I couldn't generate an answer."

    # -- retrieval + generation ----------------------------------------------
    def ask(self, question: str, api_key: Optional[str] = None, top_k: Optional[int] = None) -> Dict:
        if self.document_count() == 0:
            return {"answer": "The knowledge base is empty. Upload a document first.", "sources": []}

        key = api_key or config.HF_API_KEY
        if not key:
            raise RuntimeError(
                "No Hugging Face API token set. Paste one in the sidebar, or set "
                "HF_API_KEY in the server's .env file."
            )

        results = self.store.retrieve(tokenize(question), k=top_k or config.TOP_K)
        if not results:
            return {
                "answer": "None of the filed documents seem relevant to that question.",
                "sources": [],
            }

        context_blocks = [
            f"[Source: {chunk['source']} | chunk {chunk['chunk_index']}]\n{chunk['text']}"
            for chunk, _score in results
        ]
        sources = [
            {
                "source": chunk["source"],
                "chunk_index": chunk["chunk_index"],
                "score": round(score, 4),
                "text": chunk["text"],
            }
            for chunk, score in results
        ]

        answer = self._call_hf(key, question, context_blocks)
        return {"answer": answer, "sources": sources}
