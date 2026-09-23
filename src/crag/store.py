"""Document loading, chunking, embedding and a FAISS flat inner-product index."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import faiss
import numpy as np


@dataclass
class Doc:
    id: str
    text: str
    source: str  # file name for internal docs, URL for web results
    kind: str = "internal"  # internal | web
    score: float = 0.0
    meta: dict = field(default_factory=dict)


def load_kb(kb_dir: Path) -> list[Doc]:
    """One chunk per paragraph, prefixed with the doc title so chunks stay self-describing."""
    docs = []
    for path in sorted(kb_dir.glob("*.md")):
        raw = path.read_text(encoding="utf-8").strip()
        title, _, body = raw.partition("\n")
        title = title.lstrip("# ").strip()
        for i, para in enumerate(p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()):
            docs.append(Doc(id=f"{path.stem}#{i}", text=f"{title}: {para}", source=path.name))
    return docs


class Embedder:
    """Sentence-transformers if available; TF-IDF fallback keeps the project runnable offline."""

    def __init__(self, name: str, corpus: list[str]):
        self.name = name
        if name == "tfidf":
            from sklearn.feature_extraction.text import TfidfVectorizer

            self._tfidf = TfidfVectorizer(stop_words="english").fit(corpus)
        else:
            from sentence_transformers import SentenceTransformer

            self._st = SentenceTransformer(name)

    def encode(self, texts: list[str]) -> np.ndarray:
        if self.name == "tfidf":
            vecs = self._tfidf.transform(texts).toarray()
        else:
            vecs = self._st.encode(texts)
        vecs = np.asarray(vecs, dtype="float32")
        faiss.normalize_L2(vecs)  # cosine similarity via inner product
        return vecs


class VectorStore:
    def __init__(self, docs: list[Doc], embedder: Embedder):
        self.docs = docs
        self.embedder = embedder
        vecs = embedder.encode([d.text for d in docs])
        self.index = faiss.IndexFlatIP(vecs.shape[1])
        self.index.add(vecs)

    def search(self, query: str, k: int) -> list[Doc]:
        scores, idx = self.index.search(self.embedder.encode([query]), k)
        return [
            Doc(**{**self.docs[i].__dict__, "score": float(s)})
            for s, i in zip(scores[0], idx[0])
            if i >= 0
        ]
