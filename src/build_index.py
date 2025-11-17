"""
Build an HNSWlib index from data/raw_pages.jsonl and save:
  data/index/hnsw_index.bin
  data/index/metadatas.jsonl
"""
import os
import json
from sentence_transformers import SentenceTransformer
import numpy as np 
from tqdm import tqdm
import hnswlib

RAW = os.path.join("data", "raw_pages.jsonl")
INDEX_DIR = os.path.join("data", "index")
EMBED_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 450
CHUNK_OVERLAP = 80
HNSW_INDEX_PATH = os.path.join(INDEX_DIR, "hnsw_index.bin")
METADATA_PATH = os.path.join(INDEX_DIR, "metadatas.jsonl")

def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    text = text.replace("\r", " ")
    chunks = []
    start = 0
    L = len(text)
    while start < L:
        end = min(start + size, L)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == L:
            break
        start = max(0, end - overlap)
    return chunks

def load_pages(path=RAW):
    pages = []
    if not os.path.exists(path):
        raise FileNotFoundError(f"raw pages not found at {path}")
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            pages.append(json.loads(line))
    return pages

def main():
    os.makedirs(INDEX_DIR, exist_ok=True)
    pages = load_pages()
    model = SentenceTransformer(EMBED_MODEL)
    embeddings = []
    metadatas = []

    print("Chunking pages and computing embeddings (this may take a few minutes)...")
    for p in tqdm(pages, desc="pages"):
        url = p.get("url", "")
        title = p.get("title", "")
        text = p.get("text", "")
        chunks = chunk_text(text)
        for i, c in enumerate(chunks):
            meta = {"url": url, "title": title, "chunk_id": i, "text": c}
            metadatas.append(meta)
            emb = model.encode(c)  # shape (dim,)
            embeddings.append(np.array(emb, dtype=np.float32))

    if len(embeddings) == 0:
        raise RuntimeError("No embeddings were created; check raw_pages.jsonl content.")

    X = np.vstack(embeddings)  # shape (N, dim)
    # normalize rows for cosine space in hnswlib
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    X = X / norms

    num_elements, dim = X.shape
    print(f"Total vectors: {num_elements}, dimension: {dim}")

    # Initialize HNSW index (cosine). For cosine, hnswlib expects normalized vectors and 'cosine' space.
    p = hnswlib.Index(space='cosine', dim=dim)
    p.init_index(max_elements=num_elements, ef_construction=200, M=16)
    ids = np.arange(num_elements)
    p.add_items(X, ids)
    p.set_ef(50)  # ef for query time (higher -> more accurate)
    # save index
    p.save_index(HNSW_INDEX_PATH)
    print(f"Saved HNSW index to {HNSW_INDEX_PATH}")

    # save metadata aligned with ids (same order)
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        for m in metadatas:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    print(f"Saved metadata ({len(metadatas)} entries) to {METADATA_PATH}")

if __name__ == "__main__":
    main()
