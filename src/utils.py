"""
Utilities for semantic retrieval using hnswlib, BM25, TF-IDF, keyword, and hybrid retrieval.
This version uses hnswlib.
"""
import os
import ujson
import re
import numpy as np
import hnswlib
import traceback
from sentence_transformers import SentenceTransformer



# Keyword retrieval
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

INDEX_DIR = os.path.join("data", "index")
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
HNSW_INDEX_PATH = os.path.join(INDEX_DIR, "hnsw_index.bin")
METADATA_PATH = os.path.join(INDEX_DIR, "metadatas.jsonl")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-2.1")
CLAUDE_MAX_TOKENS = int(os.getenv("CLAUDE_MAX_TOKENS", "400"))

# caches
_model = None
_hnsw_index = None
_meta = None

_bm25 = None
_bm25_docs = None

_tfidf_vec = None
_tfidf_matrix = None
_tfidf_meta = None

def load_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL_NAME)
    return _model

def embed_text_local(text: str):
    model = load_model()
    vec = model.encode(text)
    v = np.array(vec, dtype=np.float32)
    # normalize row for cosine
    norm = np.linalg.norm(v)
    if norm == 0: norm = 1.0
    v = v / norm
    return v

def load_hnsw_and_meta():
    global _hnsw_index, _meta
    if _hnsw_index is None:
        if not os.path.exists(HNSW_INDEX_PATH):
            raise FileNotFoundError(f"hnsw index not found at {HNSW_INDEX_PATH}. Run build_index.py first.")
        # create dummy index and load actual index (hnswlib will override)
        _hnsw_index = hnswlib.Index(space='cosine', dim=1)
        _hnsw_index.load_index(HNSW_INDEX_PATH)
    if _meta is None:
        _meta = []
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            for line in f:
                _meta.append(ujson.loads(line))
    return _hnsw_index, _meta

def semantic_retrieve(query: str, top_k: int = 5):
    idx, meta = load_hnsw_and_meta()
    v = embed_text_local(query)
    # hnswlib expects shape (1, dim)
    labels, distances = idx.knn_query(v, k=top_k)
    results = []
    for lbl, dist in zip(labels[0], distances[0]):
        if lbl < 0 or lbl >= len(meta):
            continue
        # For 'cosine' space, hnswlib returns distance = 1 - cosine_sim
        sim = 1.0 - float(dist)
        m = meta[int(lbl)]
        results.append({
            "score": sim,
            "text": m.get("text", ""),
            "source": m.get("url", ""),
            "title": m.get("title", "")
        })
    return results

# -------------------------
# Keyword retrieval (new)
# -------------------------
def keyword_retrieve(query: str, top_k: int = 5):
    """
    Simple case-insensitive substring keyword retrieval.
    Scoring: primary = count of occurrences of the exact query string,
    tiebreaker = earlier first occurrence (smaller index better).
    Returns top_k results sorted by score desc, then position asc.
    """
    _, meta = load_hnsw_and_meta()
    q = query.lower().strip()
    if q == "":
        return []

    candidates = []
    for m in meta:
        text = (m.get("text") or "").lower()
        pos = text.find(q)
        if pos >= 0:
            count = text.count(q)
            # record (count, pos) for ranking
            candidates.append((count, pos if pos >= 0 else 10**9, m))

    # sort by count desc, pos asc
    candidates.sort(key=lambda x: (-x[0], x[1]))

    results = []
    for count, pos, m in candidates[:top_k]:
        results.append({
            "score": float(count),
            "text": m.get("text", ""),
            "source": m.get("url", ""),
            "title": m.get("title", "")
        })
    return results

# -------------------------
# BM25 helpers (unchanged)
# -------------------------
def build_bm25(meta_path=None):
    global _bm25, _bm25_docs
    if _bm25 is not None:
        return _bm25
    meta_path = meta_path or METADATA_PATH
    docs = []
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"metadatas not found at {meta_path}")
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            m = ujson.loads(line)
            text = m.get("text", "")
            tokens = text.split()
            docs.append(tokens)
    _bm25_docs = docs
    _bm25 = BM25Okapi(docs)
    return _bm25

def bm25_retrieve(query: str, top_k: int = 5):
    bm25 = build_bm25()
    tokens = query.split()
    scores = bm25.get_scores(tokens)
    ranked_idx = scores.argsort()[::-1][:top_k]
    meta = []
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            meta.append(ujson.loads(line))
    results = []
    for idx in ranked_idx:
        results.append({
            "score": float(scores[idx]),
            "text": meta[idx].get("text", ""),
            "source": meta[idx].get("url", ""),
            "title": meta[idx].get("title", "")
        })
    return results

# -------------------------
# TF-IDF helpers (unchanged)
# -------------------------
def build_tfidf(meta_path=None):
    global _tfidf_vec, _tfidf_matrix, _tfidf_meta
    if _tfidf_matrix is not None:
        return _tfidf_vec, _tfidf_matrix
    meta_path = meta_path or METADATA_PATH
    docs = []
    metas = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            m = ujson.loads(line)
            metas.append(m)
            docs.append(m.get("text", ""))
    _tfidf_vec = TfidfVectorizer(max_features=20000)
    _tfidf_matrix = _tfidf_vec.fit_transform(docs)
    _tfidf_meta = metas
    return _tfidf_vec, _tfidf_matrix

def tfidf_retrieve(query: str, top_k: int = 5):
    vec, mat = build_tfidf()
    qv = vec.transform([query])
    sims = cosine_similarity(qv, mat).flatten()
    idxs = sims.argsort()[::-1][:top_k]
    results = []
    for i in idxs:
        m = _tfidf_meta[i]
        results.append({
            "score": float(sims[i]),
            "text": m.get("text", ""),
            "source": m.get("url", ""),
            "title": m.get("title", "")
        })
    return results

# -------------------------
# Hybrid retrieval (unchanged)
# -------------------------
def hybrid_retrieve(query: str, top_k: int = 5, alpha: float = 0.6):
    candidate_k = max(200, top_k * 40)
    sem = semantic_retrieve(query, top_k=candidate_k)
    bm = bm25_retrieve(query, top_k=candidate_k)

    merged = {}
    sem_scores = [s["score"] for s in sem] or [0.0]
    bm_scores = [s["score"] for s in bm] or [0.0]

    def normalize_list(xs):
        xs = np.array(xs, dtype=float)
        if xs.max() == xs.min():
            if xs.max() == 0:
                return xs
            return xs / (xs.max() + 1e-9)
        return (xs - xs.min()) / (xs.max() - xs.min() + 1e-9)

    sem_norm = normalize_list(sem_scores)
    bm_norm = normalize_list(bm_scores)

    for i, s in enumerate(sem):
        key = (s["source"], s["text"][:80])
        merged.setdefault(key, {"text": s["text"], "source": s["source"], "sem": 0.0, "bm": 0.0})
        merged[key]["sem"] += float(sem_norm[i])

    for i, b in enumerate(bm):
        key = (b["source"], b["text"][:80])
        merged.setdefault(key, {"text": b["text"], "source": b["source"], "sem": 0.0, "bm": 0.0})
        merged[key]["bm"] += float(bm_norm[i])

    combined = []
    for key, v in merged.items():
        score = alpha * v["sem"] + (1.0 - alpha) * v["bm"]
        combined.append({"score": float(score), "text": v["text"], "source": v["source"]})

    combined = sorted(combined, key=lambda x: x["score"], reverse=True)[:top_k]
    return combined


# -------------------------
# Unified retrieve func
# -------------------------
def retrieve(query: str, top_k: int = 5, method: str = "semantic"):
    method = (method or "semantic").lower()
    if method == "semantic":
        return semantic_retrieve(query, top_k=top_k)
    elif method == "keyword":
        return keyword_retrieve(query, top_k=top_k)
    elif method == "bm25":
        return bm25_retrieve(query, top_k=top_k)
    elif method == "tfidf":
        return tfidf_retrieve(query, top_k=top_k)
    elif method == "hybrid":
        return hybrid_retrieve(query, top_k=top_k)
    else:
        raise ValueError(f"Unknown retrieval method: {method}")
    

# lazy import helper for Anthropic
def _call_claude(prompt: str, model: str = None, max_tokens: int = None):
    try:
        from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
    except Exception as e:
        raise RuntimeError("anthropic package not installed. Install with `pip install anthropic`.") from e

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY environment variable is not set.")

    model = model or CLAUDE_MODEL
    max_tokens = int(max_tokens or CLAUDE_MAX_TOKENS)

    client = Anthropic(api_key=api_key)
    full_prompt = HUMAN_PROMPT + prompt + AI_PROMPT

    try:
        resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=0.0,
        messages=[
            {"role": "user", "content": full_prompt}
        ]
        )
        if not resp.content:
                return ""
            # get first text block
        return resp.content[0].text.strip()
    except Exception as e:
        tb = traceback.format_exc()
        raise RuntimeError(f"Claude API call failed: {e}\n{tb}") from e


def _truncate_text(text: str, max_chars: int = 2000):
    """
    Truncate text to max_chars while:
      - preferring to cut at a sentence end if possible,
      - removing any leading partial word at the start of the returned slice
      - trimming trailing incomplete words at the cut point
    """
    if not text:
        return text
    if len(text) <= max_chars:
        # also cleanup leading/trailing whitespace
        return text.strip()

    # Primary slice
    slice_ = text[:max_chars]

    # Try to extend to the end of the last complete sentence inside slice_
    # (look for last sentence-ending punctuation within the slice)
    last_sent_end = max(slice_.rfind('.'), slice_.rfind('!'), slice_.rfind('?'))
    if last_sent_end and last_sent_end > int(0.6 * max_chars):
        # keep up to that sentence end (prefer longer but complete sentence)
        slice_ = slice_[: last_sent_end + 1]
    else:
        # otherwise try to cut at nearest previous whitespace so we don't break a word at the end
        last_space = slice_.rfind(' ')
        if last_space > int(0.4 * max_chars):
            slice_ = slice_[: last_space]

    # Remove any leading partial word: if the slice starts with a lowercase run (like 'ng '),
    # drop characters until the first whitespace to align to a word boundary.
    # But be conservative: only strip a leading fragment up to 20 chars.
    if re.match(r'^[a-z]{1,20}\S', slice_):
        # find first whitespace
        sp = re.search(r'\s', slice_)
        if sp:
            slice_ = slice_[sp.start():].lstrip()
    # also remove leading punctuation/spurious chars
    slice_ = re.sub(r'^[\s\-\–\—\:\;\,]+', '', slice_)

    # Final cleanup: normalize whitespace and trim
    slice_ = re.sub(r'\s+', ' ', slice_).strip()
    # if it's still too long, shorten defensively
    if len(slice_) > max_chars:
        slice_ = slice_[:max_chars].rsplit(' ', 1)[0].rstrip() + " ..."
    return slice_



def _build_claude_prompt(query: str, retrieved: list, top_k: int = 5):
 
    instructions = (
        "You are given a user's question and a list of numbered source passages. "
        "Produce a concise answer in 1–6 BULLET POINTS (each bullet 1–2 short sentences). "
        "Use ONLY the information in the provided passages — do NOT invent facts or use external knowledge. "
        "For factual claims, include inline numeric citations using passage numbers in brackets (e.g., [1], [2]). "
        "If multiple passages support a claim, include all relevant citations (e.g., [1,3]). "
        "Return ONLY the bullet points and nothing else (no sources list, no passages, no extra commentary).\n\n"
    )

    passages = []
    for i, item in enumerate(retrieved[:top_k], start=1):
        text = item.get("text", "") or ""
        url = item.get("source") or item.get("url") or ""
        title = item.get("title") or ""
        text_trunc = _truncate_text(text, max_chars=1500)
        # include the URL inline so Claude can refer to it if needed, but we won't ask for a separate map
        passages.append(f"[{i}] Title: {title}\nURL: {url}\nPassage: {text_trunc}")

    passages_block = "\n\n".join(passages)

    prompt = (
        instructions
        + f"USER QUESTION:\n{query}\n\n"
        + "SOURCE PASSAGES:\n" + passages_block + "\n\n"
        + "INSTRUCTIONS:\n- Use ONLY the passages above.\n"
        + "- Answer in bullets; each factual claim should include numeric citation(s) like [1] or [1,3].\n"
        + "- Place citations as bracketed passage numbers (e.g. [1]) immediately after claims.\n"
        + "- If the passages don't answer the question, respond: 'I don't know based on the provided sources.'\n\n"
        + "Answer now:"
    )

    return prompt


def generate_from_retrieval(query: str, top_k: int = 5, method: str = "semantic"):
    """
    Claude-only generator wrapper:
      - Retrieves passages using retrieve(...)
      - Builds a strict prompt and calls Claude to produce the final answer.
    Returns:
      {
        "answer": str,
        "retrieved": [...],   # full passages (with URLs) so grader can verify citations
        "paraphrase_used": True
      }
    Raises RuntimeError on any failure (no fallback).
    """
    retrieved = retrieve(query, top_k=top_k, method=method)


    if not retrieved:
        raise RuntimeError("No passages retrieved for the query; nothing to generate from.")

    prompt = _build_claude_prompt(query, retrieved, top_k=top_k)
    completion = _call_claude(prompt, model=CLAUDE_MODEL, max_tokens=CLAUDE_MAX_TOKENS)

    return {
        "answer": completion,
        "retrieved": retrieved,
        "paraphrase_used": True
    }