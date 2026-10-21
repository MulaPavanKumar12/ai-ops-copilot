"""Embeddings and vector stores for the incident knowledge base.

Embeddings: the default backend is **LSA** (TF-IDF over words and bigrams, then
truncated SVD to a dense vector). It runs fully offline, fits in milliseconds,
and works well on log-heavy text where exact tokens ("QueuePool",
"replication slot") carry most of the meaning. A ``sentence-transformers``
backend can be switched on with ``EMBEDDING_BACKEND`` for more semantic
matching on free-text reports.

Vector store: **ChromaDB** (persistent, on disk) when installed, otherwise a
small in-memory NumPy store with the same interface. Embeddings are always
computed by us and passed in, so Chroma never downloads a model.
"""

from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import joblib
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import Normalizer

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ embedders
class Embedder(Protocol):
    """Anything that turns texts into L2-normalised vectors."""

    def fit(self, texts: list[str]) -> Embedder: ...
    def embed(self, texts: list[str]) -> np.ndarray: ...
    def save(self, path: Path) -> None: ...


class LSAEmbedder:
    """TF-IDF + truncated SVD ("latent semantic analysis"), offline and fast."""

    def __init__(self, dims: int = 128) -> None:
        self.dims = dims
        self.pipeline: Pipeline | None = None

    def fit(self, texts: list[str]) -> LSAEmbedder:
        """Fit the vocabulary and projection on the corpus."""
        if len(texts) < 3:
            raise ValueError("need at least 3 documents to fit LSA embeddings")
        tfidf = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1,
                                token_pattern=r"(?u)\b[\w./:-]{2,}\b")
        n_terms = len(tfidf.fit(texts).vocabulary_)
        dims = max(2, min(self.dims, len(texts) - 1, n_terms - 1))
        self.pipeline = make_pipeline(tfidf, TruncatedSVD(dims, random_state=42), Normalizer(copy=False))
        self.pipeline.fit(texts)
        return self

    def embed(self, texts: list[str]) -> np.ndarray:
        """Vectors of shape (n, dims)."""
        if self.pipeline is None:
            raise RuntimeError("embedder is not fitted")
        return np.asarray(self.pipeline.transform(texts), dtype=np.float32)

    def save(self, path: Path) -> None:
        """Persist the fitted pipeline."""
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"kind": "lsa", "pipeline": self.pipeline, "dims": self.dims}, path)


class SentenceTransformerEmbedder:
    """Wrapper for ``sentence-transformers`` (optional dependency)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer  # optional

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def fit(self, texts: list[str]) -> SentenceTransformerEmbedder:
        """Pre-trained: nothing to fit."""
        return self

    def embed(self, texts: list[str]) -> np.ndarray:
        """Normalised sentence embeddings."""
        return np.asarray(self.model.encode(texts, normalize_embeddings=True), dtype=np.float32)

    def save(self, path: Path) -> None:
        """Store only the model name."""
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"kind": "sentence-transformers", "model_name": self.model_name}, path)


def make_embedder(backend: str) -> Embedder:
    """Create an embedder for ``backend`` ("lsa" or "sentence-transformers")."""
    if backend == "sentence-transformers":
        return SentenceTransformerEmbedder()
    return LSAEmbedder()


def load_embedder(path: Path) -> Embedder:
    """Load an embedder saved with ``save``."""
    data = joblib.load(path)
    if data["kind"] == "sentence-transformers":
        return SentenceTransformerEmbedder(data["model_name"])
    emb = LSAEmbedder(int(data["dims"]))
    emb.pipeline = data["pipeline"]
    return emb


# --------------------------------------------------------------- vector stores
@dataclass
class Hit:
    """One search result."""

    doc_id: str
    score: float
    metadata: dict[str, Any]
    document: str


class VectorStore(Protocol):
    """Minimal vector store interface used by the knowledge base."""

    def reset(self) -> None: ...
    def add(self, ids: list[str], embeddings: np.ndarray, metadatas: list[dict[str, Any]],
            documents: list[str]) -> None: ...
    def query(self, embedding: np.ndarray, k: int, where: dict[str, Any] | None = None) -> list[Hit]: ...
    def count(self) -> int: ...


class MemoryStore:
    """Exact cosine search in NumPy, persisted with joblib."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self.ids: list[str] = []
        self.vectors = np.zeros((0, 0), dtype=np.float32)
        self.metadatas: list[dict[str, Any]] = []
        self.documents: list[str] = []
        if path is not None and path.exists():
            data = joblib.load(path)
            self.ids, self.vectors = data["ids"], data["vectors"]
            self.metadatas, self.documents = data["metadatas"], data["documents"]

    def reset(self) -> None:
        """Remove everything."""
        self.__init__(None)  # type: ignore[misc]

    def add(self, ids: list[str], embeddings: np.ndarray, metadatas: list[dict[str, Any]],
            documents: list[str]) -> None:
        """Append documents."""
        emb = np.asarray(embeddings, dtype=np.float32)
        self.vectors = emb if self.vectors.size == 0 else np.vstack([self.vectors, emb])
        self.ids += ids
        self.metadatas += metadatas
        self.documents += documents

    def persist(self, path: Path) -> None:
        """Write to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"ids": self.ids, "vectors": self.vectors, "metadatas": self.metadatas,
                     "documents": self.documents}, path)

    def query(self, embedding: np.ndarray, k: int, where: dict[str, Any] | None = None) -> list[Hit]:
        """Top-k by cosine similarity, with optional exact-match metadata filter."""
        if not self.ids:
            return []
        sims = self.vectors @ np.asarray(embedding, dtype=np.float32).reshape(-1)
        order = np.argsort(-sims)
        hits: list[Hit] = []
        for i in order:
            meta = self.metadatas[i]
            if where and any(meta.get(key) != val for key, val in where.items()):
                continue
            hits.append(Hit(self.ids[i], float(sims[i]), meta, self.documents[i]))
            if len(hits) >= k:
                break
        return hits

    def count(self) -> int:
        """Number of stored documents."""
        return len(self.ids)


class ChromaStore:
    """Persistent ChromaDB collection using cosine distance and our own embeddings."""

    def __init__(self, directory: Path, collection: str = "incident_knowledge") -> None:
        import chromadb  # optional dependency

        directory.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(directory))
        self._name = collection
        self._collection = self._client.get_or_create_collection(
            collection, metadata={"hnsw:space": "cosine"}, embedding_function=None)

    def reset(self) -> None:
        """Drop and recreate the collection."""
        with contextlib.suppress(Exception):  # the collection may not exist yet
            self._client.delete_collection(self._name)
        self._collection = self._client.get_or_create_collection(
            self._name, metadata={"hnsw:space": "cosine"}, embedding_function=None)

    def add(self, ids: list[str], embeddings: np.ndarray, metadatas: list[dict[str, Any]],
            documents: list[str]) -> None:
        """Insert documents in batches."""
        for s in range(0, len(ids), 500):
            self._collection.add(ids=ids[s:s + 500], embeddings=np.asarray(embeddings[s:s + 500]).tolist(),
                                 metadatas=metadatas[s:s + 500],  # type: ignore[arg-type]
                                 documents=documents[s:s + 500])

    def query(self, embedding: np.ndarray, k: int, where: dict[str, Any] | None = None) -> list[Hit]:
        """Top-k nearest documents (similarity = 1 - cosine distance)."""
        if self.count() == 0:
            return []
        res = self._collection.query(query_embeddings=[np.asarray(embedding).tolist()],
                                     n_results=min(k, self.count()), where=where or None)
        hits = []
        distances, metas, docs = res["distances"] or [[]], res["metadatas"] or [[]], res["documents"] or [[]]
        for doc_id, dist, meta, doc in zip(res["ids"][0], distances[0], metas[0], docs[0], strict=True):
            hits.append(Hit(doc_id, 1.0 - float(dist), dict(meta), doc))
        return hits

    def count(self) -> int:
        """Number of stored documents."""
        return int(self._collection.count())


def make_store(kind: str, directory: Path) -> VectorStore:
    """Chroma if requested and importable, otherwise the in-memory store."""
    if kind == "chroma":
        try:
            return ChromaStore(directory / "chroma")
        except ImportError:
            logger.warning("chromadb not installed; using the in-memory vector store")
    return MemoryStore(directory / "memory_store.joblib")
