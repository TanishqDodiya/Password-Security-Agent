"""
agent/rag.py - Lightweight local RAG for security knowledge retrieval.

Architecture (MVP):
Documents -> Chunking -> TF-IDF Embeddings -> In-memory Vector Store -> Retriever

- No cloud vector DB, no external dependencies beyond scikit-learn (already present)
- Only knowledge-base content is embedded, never user passwords
- Retrieval query is built from security findings, not raw password
- Deterministic, lightweight, explainable
"""

import os
import re
from typing import List, Dict, Any
from pathlib import Path

# Global cache for vector store (in-memory, no persistence of passwords)
_VECTORIZER = None
_CHUNKS: List[Dict[str, str]] = []
_TFIDF_MATRIX = None

KNOWLEDGE_DIR = os.path.join(os.path.dirname(__file__), "..", "knowledge")

def _load_documents() -> List[Dict[str, str]]:
    """Load markdown documents from knowledge/ and chunk by paragraphs."""
    docs = []
    knowledge_path = Path(KNOWLEDGE_DIR)
    if not knowledge_path.exists():
        # Try alternative path
        knowledge_path = Path("knowledge")
    if not knowledge_path.exists():
        return docs
    for md_file in sorted(knowledge_path.glob("*.md")):
        try:
            text = md_file.read_text(encoding="utf-8")
            # Chunk by double newline (paragraphs) and also keep full doc
            paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
            for idx, para in enumerate(paragraphs):
                # Skip very short chunks
                if len(para) < 20:
                    continue
                docs.append({
                    "content": para,
                    "source": f"{md_file.name}#chunk{idx}",
                    "doc": md_file.name,
                })
            # Also add full document as a chunk for broader retrieval
            # But to keep concise, we use paragraphs only
        except Exception:
            continue
    return docs

def _build_vector_store():
    global _VECTORIZER, _CHUNKS, _TFIDF_MATRIX
    if _TFIDF_MATRIX is not None:
        return
    chunks = _load_documents()
    if not chunks:
        _CHUNKS = []
        return
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError:
        # Fallback: simple keyword store without embeddings
        _CHUNKS = chunks
        _VECTORIZER = None
        _TFIDF_MATRIX = None
        return

    texts = [c["content"] for c in chunks]
    vectorizer = TfidfVectorizer(stop_words="english", max_features=500)
    matrix = vectorizer.fit_transform(texts)
    _VECTORIZER = vectorizer
    _CHUNKS = chunks
    _TFIDF_MATRIX = matrix

def _build_retrieval_query(findings: Dict[str, Any]) -> str:
    """
    Build retrieval query from security findings only.
    Never includes raw password.
    Findings include: risk_level, heuristic_score, issues, warnings, features summary.
    """
    parts = []
    risk_level = findings.get("risk_level", "")
    if risk_level:
        parts.append(f"risk {risk_level}")

    # Add issue/warning codes and messages (already aggregated, no password)
    for bucket in ["issues", "warnings", "positive_signals"]:
        for item in findings.get(bucket, []):
            code = item.get("code", "")
            if code:
                parts.append(code.replace("_", " "))
            # Also add message keywords but truncated
            msg = item.get("message", "")
            if msg:
                # Take first 10 words to avoid verbosity
                parts.append(" ".join(msg.split()[:8]))

    # Add feature-based hints (length, types) without password
    features = findings.get("features", {})
    if features:
        n = features.get("length", 0)
        if n is not None:
            if n < 6:
                parts.append("very short password length")
            elif n >= 12:
                parts.append("long password length")
        type_count = features.get("unique_character_type_count", 0)
        if type_count == 1:
            parts.append("single character type")
        elif type_count >= 3:
            parts.append("multiple character types")

    # Add ML note if present (without password)
    ml = findings.get("ml", {})
    if ml and ml.get("available"):
        parts.append("dataset model evidence")

    # Fallback if no findings
    if not parts:
        parts.append("password security best practices")

    query = " ".join(parts)
    # Ensure query does not contain password-like long random string - it is derived from findings only
    return query

def retrieve_knowledge(findings: Dict[str, Any], top_k: int = 3) -> Dict[str, Any]:
    """
    Retrieve relevant security knowledge based on findings.

    Args:
        findings: dict with risk_level, issues, warnings, features, ml (no password)
        top_k: number of chunks to return

    Returns:
        {
            "query": str (findings-derived, no password),
            "results": [{"content": str, "source": str, "score": float}, ...],
            "retrieval_available": bool
        }
    """
    _build_vector_store()
    query = _build_retrieval_query(findings)

    if not _CHUNKS:
        return {"query": query, "results": [], "retrieval_available": False}

    # If vectorizer not available, use simple keyword matching fallback
    if _VECTORIZER is None or _TFIDF_MATRIX is None:
        # Simple keyword overlap fallback
        query_words = set(query.lower().split())
        scored = []
        for chunk in _CHUNKS:
            content_words = set(chunk["content"].lower().split())
            overlap = len(query_words & content_words)
            scored.append((overlap, chunk))
        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for score, chunk in scored[:top_k]:
            if score > 0:
                results.append({"content": chunk["content"], "source": chunk["source"], "score": float(score)})
        return {"query": query, "results": results, "retrieval_available": True}

    # TF-IDF cosine similarity (vectors are L2 normalized, dot product = cosine)
    try:
        from sklearn.metrics.pairwise import cosine_similarity
        q_vec = _VECTORIZER.transform([query])
        sims = cosine_similarity(q_vec, _TFIDF_MATRIX).flatten()
        # Get top_k indices sorted by similarity
        top_indices = sims.argsort()[::-1][:top_k]
        results = []
        for idx in top_indices:
            score = float(sims[idx])
            if score > 0.05:  # threshold to avoid irrelevant
                chunk = _CHUNKS[idx]
                results.append({"content": chunk["content"], "source": chunk["source"], "score": score})
        # If none above threshold, return top 1 anyway for context
        if not results and len(top_indices) > 0:
            idx = top_indices[0]
            chunk = _CHUNKS[idx]
            results.append({"content": chunk["content"], "source": chunk["source"], "score": float(sims[idx])})
        return {"query": query, "results": results, "retrieval_available": True}
    except Exception:
        return {"query": query, "results": [], "retrieval_available": False}

def knowledge_base_available() -> bool:
    _build_vector_store()
    return len(_CHUNKS) > 0

def get_knowledge_stats() -> Dict[str, Any]:
    _build_vector_store()
    return {"num_chunks": len(_CHUNKS), "num_docs": len(set(c["doc"] for c in _CHUNKS)) if _CHUNKS else 0}

# For testing: ensure query does not contain password
def _query_contains_password(query: str, password: str) -> bool:
    if not password or len(password) < 4:
        return False
    return password in query
