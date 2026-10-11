"""
A thin, persistent TF-IDF index — the retrieval half of the pipeline.

No embedding API and no vector database: each question and each chunk is
scored by term frequency / inverse document frequency and compared with
cosine similarity, computed fresh over the current corpus on every query.
This keeps the whole assistant runnable with nothing but a free Hugging
Face token for the generation step.
"""
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from . import config
from .ingest import Chunk


class TfidfStore:
    def __init__(self):
        # One entry per uploaded document:
        # {id, name, added_at, chunks: [{id, chunk_index, text, tokens}]}
        self.documents: List[dict] = []

    # -- writing -----------------------------------------------------------
    def add_document(self, doc_id: str, name: str, added_at: str, chunks: List[Chunk]) -> None:
        self.documents.append({
            "id": doc_id,
            "name": name,
            "added_at": added_at,
            "chunks": [
                {
                    "id": c.id,
                    "chunk_index": c.chunk_index,
                    "text": c.text,
                    "tokens": c.tokens,
                }
                for c in chunks
            ],
        })

    def remove_document(self, doc_id: str) -> bool:
        before = len(self.documents)
        self.documents = [d for d in self.documents if d["id"] != doc_id]
        return len(self.documents) != before

    def clear(self) -> None:
        self.documents = []

    # -- reading -------------------------------------------------------------
    def list_documents(self) -> List[dict]:
        return [
            {"id": d["id"], "name": d["name"], "added_at": d["added_at"], "chunks": len(d["chunks"])}
            for d in self.documents
        ]

    def total_chunks(self) -> int:
        return sum(len(d["chunks"]) for d in self.documents)

    def _all_chunks(self) -> List[dict]:
        all_chunks = []
        for doc in self.documents:
            for c in doc["chunks"]:
                all_chunks.append({
                    "source": doc["name"],
                    "chunk_index": c["chunk_index"],
                    "text": c["text"],
                    "tokens": c["tokens"],
                })
        return all_chunks

    # -- TF-IDF scoring --------------------------------------------------------
    @staticmethod
    def _document_frequencies(all_chunks: List[dict]) -> Dict[str, int]:
        df: Dict[str, int] = {}
        for c in all_chunks:
            for term in set(c["tokens"]):
                df[term] = df.get(term, 0) + 1
        return df

    @staticmethod
    def _idf(term: str, df: Dict[str, int], n: int) -> float:
        d = df.get(term, 0)
        return math.log((1 + n) / (1 + d)) + 1

    @classmethod
    def _vectorize(cls, tokens: List[str], df: Dict[str, int], n: int) -> Tuple[Dict[str, float], float]:
        if not tokens:
            return {}, 0.0
        tf: Dict[str, int] = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
        vec: Dict[str, float] = {}
        norm_sq = 0.0
        for term, count in tf.items():
            if term not in df:
                continue
            weight = (count / len(tokens)) * cls._idf(term, df, n)
            vec[term] = weight
            norm_sq += weight * weight
        return vec, math.sqrt(norm_sq)

    @staticmethod
    def _cosine(vec_a: Dict[str, float], norm_a: float, vec_b: Dict[str, float], norm_b: float) -> float:
        if norm_a == 0 or norm_b == 0:
            return 0.0
        small, large = (vec_a, vec_b) if len(vec_a) < len(vec_b) else (vec_b, vec_a)
        dot = sum(weight * large[term] for term, weight in small.items() if term in large)
        return dot / (norm_a * norm_b)

    def retrieve(self, question_tokens: List[str], k: int = config.TOP_K) -> List[Tuple[dict, float]]:
        all_chunks = self._all_chunks()
        if not all_chunks:
            return []

        df = self._document_frequencies(all_chunks)
        n = len(all_chunks)
        q_vec, q_norm = self._vectorize(question_tokens, df, n)

        scored = []
        for chunk in all_chunks:
            c_vec, c_norm = self._vectorize(chunk["tokens"], df, n)
            score = self._cosine(q_vec, q_norm, c_vec, c_norm)
            if score > 0:
                scored.append((chunk, score))

        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:k]

    def __len__(self) -> int:
        return len(self.documents)

    # -- persistence -----------------------------------------------------------
    def save(self, path: Path = config.INDEX_PATH) -> None:
        path.write_text(json.dumps(self.documents, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path = config.INDEX_PATH) -> "TfidfStore":
        store = cls()
        if path.exists():
            store.documents = json.loads(path.read_text(encoding="utf-8"))
        return store
