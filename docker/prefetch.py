"""One-time model prefetch — run ONLINE to populate the `model-cache` volume.

After this, api/worker/ui run fully offline (HF_HUB_OFFLINE=1) against the cache.
This is the only step that touches the network, and it downloads only public
model/vocab files — never any document content.

    docker compose --profile prefetch run --rm --build prefetch

Unlike stage 1 there is no parser pipeline to warm here: the dense encoder is the
only downloaded asset on the default path. BM25 is arithmetic and needs nothing.
The optional reranker is fetched only if RAG_EMBED_RERANK_ENABLED=true, so an
offline deployment that later flips that flag gets a clear failure rather than a
surprise download attempt.

The warm-up deliberately *encodes a sentence* instead of just constructing the
model: sentence-transformers resolves tokenizer/pooling assets lazily, and a
construct-only prefetch leaves those out of the cache — which then fails at
runtime with LocalEntryNotFound. (Same class of bug stage 1 hit with Docling.)
"""

from __future__ import annotations

import sys

from rag_embedding import EmbeddingConfig


def _prefetch_dense(cfg: EmbeddingConfig) -> None:
    if cfg.embedding_provider.lower() == "azure":
        print("[prefetch] embedding_provider=azure — nothing to download")
        return
    print(f"[prefetch] dense embedding model: {cfg.local_embedding_model}")
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(cfg.local_embedding_model, device="cpu")
    vector = model.encode(["warm up the tokenizer and pooling modules"], normalize_embeddings=True)
    print(f"[prefetch] dense model cached (dim={len(vector[0])})")


def _prefetch_reranker(cfg: EmbeddingConfig) -> None:
    if not cfg.rerank_enabled:
        print("[prefetch] rerank_enabled=false — skipping reranker")
        return
    print(f"[prefetch] reranker: {cfg.rerank_model}")
    from sentence_transformers import CrossEncoder

    CrossEncoder(cfg.rerank_model, device="cpu").predict([("warm", "up")])
    print("[prefetch] reranker cached")


def main() -> None:
    cfg = EmbeddingConfig()
    if cfg.offline:
        print(
            "[prefetch] RAG_EMBED_OFFLINE=true in this container — prefetch must run ONLINE.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    _prefetch_dense(cfg)
    _prefetch_reranker(cfg)
    print("[prefetch] done")


if __name__ == "__main__":
    main()
