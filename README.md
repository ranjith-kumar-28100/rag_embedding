# rag_embedding

Embedding, hybrid indexing & retrieval. **Stage 2** of a RAG pipeline, downstream of
[`rag_ingestion`](../rag_ingestion).

One job: **`list[Chunk]` (JSONL) in → embedded, hybrid-indexed, retrievable context out.**
It does not parse or chunk documents (that is stage 1), and it does not synthesize
answers.

```
rag_ingestion ──▶ chunks.jsonl ──▶ rag_embedding ──▶ Qdrant (dense + sparse)
 (stage 1)          (contract)       │  embed children      + parent docstore
                                     │  index parents/kids        │
                                     ▼                            ▼
                              Search API / UI  ◀── hybrid retrieval + parent lookup
                                     │
                                     └──▶ RAGAS evaluation + reindex controls
```

The two stages share a **wire contract, not code**: chunk JSONL is uploaded to this
service's API. There is no shared volume and nothing here imports stage 1.

Everything runs locally by default — sentence-transformers for dense vectors, BM25
(pure arithmetic) for sparse, a local Qdrant container, an Ollama judge for
evaluation. Azure is opt-in and rejected outright when `offline=true`.

---

## Quickstart (Docker)

```bash
cp .env.example .env          # optional; the defaults work as-is

make prefetch                 # ONCE, online: downloads the embedding model into
                              # the model-cache volume
make up                       # qdrant + redis + api + worker + ui, fully offline

# UI:  http://localhost:8502
# API: http://localhost:8001/docs
# Qdrant dashboard: http://localhost:6333/dashboard
```

Then upload a stage-1 `chunks.jsonl` on the **Index** tab and search it on the
**Search** tab. A sample file that exercises every block type lives in
[`examples/sample_chunks.jsonl`](examples/sample_chunks.jsonl):

```bash
curl -F "files=@examples/sample_chunks.jsonl" http://localhost:8001/index
curl -X POST http://localhost:8001/search \
     -H 'content-type: application/json' \
     -d '{"query": "how are dense and sparse signals combined?"}'
```

> If port 6333 is already taken by another Qdrant on your machine, either stop it or
> remap the port in `docker-compose.yml` and set `RAG_EMBED_QDRANT_URL` to match.

## Quickstart (library)

```python
from qdrant_client import QdrantClient
from sqlalchemy import create_engine

from rag_embedding import EmbeddingConfig, Indexer, Retriever, SearchRequest
from rag_embedding.embedders.factory import build_dense_embedder, build_sparse_embedder
from rag_embedding.store.docstore import SqliteParentDocStore
from rag_embedding.store.qdrant_store import QdrantStore, build_client

cfg = EmbeddingConfig()                       # local providers, no credentials
store = QdrantStore(build_client(cfg), cfg)
docstore = SqliteParentDocStore(create_engine("sqlite:///parents.db"))
docstore.create_schema()

indexer = Indexer(
    cfg=cfg, store=store, docstore=docstore,
    dense=build_dense_embedder(cfg), sparse=build_sparse_embedder(cfg),
)
indexer.ensure_collection("rag_v1")
indexer.index_file("rag_v1", "chunks.jsonl")
indexer.activate("rag_v1")                    # point the `rag` alias at it

retriever = Retriever(
    cfg=cfg, store=store, docstore=docstore,
    dense=build_dense_embedder(cfg), sparse=build_sparse_embedder(cfg),
)
response = retriever.search(SearchRequest(query="reciprocal rank fusion"))
for parent in response.parents:               # small-to-big: parents are the context
    print(parent.score, parent.text[:80])
```

---

## Input contract

The unit of exchange is the stage-1 `Chunk`, one JSON object per line
(`.jsonl` or `.jsonl.gz`). Exact wire format = `Chunk.model_dump(mode="json")`.

| Field | Use here |
|---|---|
| `chunk_id` | → Qdrant point id, `uuid5(chunk_id)` (idempotent upserts) |
| `doc_id` | payload + filter key |
| `parent_id` | links child → parent for parent lookup |
| `role` | `child` is embedded; `parent` goes to the docstore |
| `block_type` | payload + optional filter |
| `text` | embedded (dense) + tokenized (BM25) |
| `section_path` | payload; indexed element-wise for filtering |
| `source_uri`, `file_type` | payload (+ `file_type` filter) |
| `page` / `slide_number` / `sheet_name` / `row_range` | payload (provenance) |
| `token_count`, `metadata` | payload (metadata flattened to scalars) |

The chunk is treated as **opaque beyond these fields** — structure is never
re-derived, and unknown fields are ignored, so a new stage-1 block type needs zero
changes here.

---

## Embeddings

### Dense (semantic)

| Provider | Default model | Dim | Notes |
|---|---|---|---|
| `sentence_transformers` (default, offline) | `all-MiniLM-L6-v2` | 384 | L2-normalized, batched |
| `azure` (online, opt-in) | `text-embedding-3-large` | 3072 | needs endpoint + key |

Vectors are L2-normalized and the collection uses cosine distance. The model id and
dim are recorded on the collection, and the query path **hard-errors** on a mismatch
rather than searching across embedding spaces.

### Sparse (lexical) — BM25

`rag_embedding/embedders/sparse_bm25.py` is an in-repo BM25 encoder: no model
download, nothing to prefetch. It emits only the term-frequency half of BM25,

```
w(t, d) = tf * (k1 + 1) / (tf + k1 * (1 - b + b * |d| / avgdl))
```

and the collection declares `Modifier.IDF`, so **Qdrant supplies the IDF term at
query time** from its own live corpus statistics. That split is what makes encoding
streaming: a batch can be indexed without knowing the corpus, and IDF stays correct
as the corpus grows. Queries carry weight `1.0` per unique term.

`avgdl` is the one statistic a streaming encoder cannot know, so it is config
(`bm25_avg_len`, default 256) rather than a measured mean — it only affects how
strongly length is normalised.

Tokenization is lowercase → split → stopword removal → **conservative** suffix
stripping. It is deliberately not Porter: over-stemming conflates distinct
identifiers, which is the opposite of what the lexical branch exists for. What
matters is that documents and queries are normalised identically.

---

## Vector store — Qdrant + HNSW

One collection per (model, dim), with two named vectors:

```python
vectors_config = {"dense": VectorParams(
    size=dim, distance=Cosine,
    hnsw_config=HnswConfigDiff(m=16, ef_construct=128),
    on_disk=True,
    quantization_config=ScalarQuantization(int8, always_ram=True))}
sparse_vectors_config = {"bm25": SparseVectorParams(modifier=Modifier.IDF)}
```

**Storage effectiveness**, all config-driven:

- `on_disk=True` for dense vectors, with **int8 scalar quantization kept
  `always_ram`** — candidate scoring stays in memory at ~4× less of it, exact
  rescoring reads the full vector from disk;
- `on_disk_payload=True`, so payload size does not inflate RAM;
- keyword payload indexes on `doc_id`, `file_type`, `block_type`, `section_path` —
  a filtered search is an index intersection, not a full scan;
- deterministic `uuid5(chunk_id)` point ids — re-indexing a chunk overwrites it,
  never duplicates;
- reindex via aliases, so storage churn is bounded to one extra collection during a
  rebuild.

`section_path` is stored as a list, which Qdrant indexes element-wise — you can
filter by an ancestor heading without knowing the leaf.

### Parent docstore

Parents are **stored, never embedded**. Default backend is SQLite via SQLAlchemy
(`parent_chunks`), so parent writes are transactional alongside job state and one URL
swap moves both to Postgres. `RAG_EMBED_DOCSTORE=qdrant` instead uses a payload-only
`rag__parents` collection for deployments that would rather run one datastore.

Parents are **version independent**: a reindex builds a new vector collection and
reuses the same parents.

---

## Retrieval

`POST /search` with `{query, top_k, mode, filters, with_parents}`.

- **`dense`** — HNSW ANN over the dense vector.
- **`sparse`** — BM25 sparse query.
- **`hybrid`** (default) — Qdrant Query API: prefetch dense + prefetch sparse, fused
  **server-side** with RRF (`fusion=dbsf` for distribution-based fusion). One round
  trip, one place the filter is applied.

**Parent lookup (small-to-big).** Hybrid search ranks *children*; their `parent_id`s
are deduped (keeping the best child score), parents are fetched from the docstore,
and **parents are returned as the context** with their matched children attached.
A ranked child with no parent row becomes its own context rather than vanishing.
`with_parents=false` returns raw child hits.

**Filtering** on the four indexed payload keys. Within a field, values are ORed;
across fields, ANDed.

`POST /search/compare` runs all three modes for one query — three real queries, not a
client-side re-fusion — which is what the UI's raw-vs-fused view shows.

### Reranking (optional, local, open-source)

A cross-encoder re-scores the fused candidates before parent lookup. The dense
branch is a *bi-encoder*: it embeds query and chunk independently, so a chunk's
vector is fixed before anyone asks a question. A cross-encoder reads the pair
**together**, which is strictly more informative and correspondingly slower — one
forward pass per candidate. That is why it runs on the fused top-N, not the corpus.

| Model | Licence | Size | CPU latency @10 / @50 candidates |
|---|---|---|---|
| `BAAI/bge-reranker-base` (default) | MIT | ~280 MB | ~140 ms / ~610 ms |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | Apache-2.0 | ~90 MB | ~25 ms / ~95 ms |

Those are measured, not estimated, and they are the reason this is off by default:
the fused search alone is ~10–20 ms, so reranking 50 candidates with the base model
makes the request roughly **30× slower**. Size `rerank_candidates` against a latency
budget, or take the MiniLM model. A GPU (`rerank_device=cuda`) changes the picture
entirely.

Both run through sentence-transformers' `CrossEncoder`, already a core dependency,
so enabling reranking **adds no packages** — only a one-time model download that
`make prefetch` picks up when `rerank_enabled=true`.

```bash
RAG_EMBED_RERANK_ENABLED=true make prefetch   # cache the cross-encoder, once
RAG_EMBED_RERANK_ENABLED=true make up
```

Two details that decide whether it does anything at all:

- **Candidate widening.** With reranking on, the store is asked for
  `rerank_candidates` (default 50) instead of `top_k`, then truncated back to
  `top_k` afterwards. Fetching exactly `top_k` would leave nothing to reorder, so
  the config rejects `rerank_candidates < top_k` outright.
- **Two scales.** Cross-encoder outputs are unbounded logits, unrelated to RRF's
  reciprocal ranks. The reranked value replaces `score` and the fused value is kept
  as `retrieval_score`, so both are always visible and neither silently stands in
  for the other. `source` becomes e.g. `hybrid+rerank` and the response carries
  `reranked: true`.

Reranking is switchable **per request** (`{"rerank": true｜false}` on `/search`) and
per eval run, so you can A/B it on one query or measure it against your eval set
rather than taking it on faith. The Search tab shows fused-vs-reranked side by side
and counts how many positions moved. Asking for `rerank: true` on a service with no
reranker configured is a 400, not a silent pass-through — a silent no-op reads as
"the reranker didn't help".

---

## Reindexing (zero downtime)

Serving always goes through a Qdrant **alias** (`rag` by default) pointing at a
physical collection (`rag_v1`, `rag_v2`, …).

1. `POST /reindex` builds `rag_vN+1` from the current collection's **stored
   payloads** — child text lives in the Qdrant payload and parents in the docstore, so
   nothing is re-uploaded and stage 1 is not involved.
2. The job runs async with progress; the old collection keeps serving throughout.
3. `activate_on_success: true`, or `POST /collections/{name}/activate`, repoints the
   alias atomically. The same endpoint is the rollback path.

A `PARTIAL` build never takes over the alias. Each reindex job stores the old-vs-new
`config_diff` and a free-text `reason` for auditability.

The intended loop: reindex → evaluate the *candidate* collection
(`POST /evaluate` with `collection=rag_v2`) → compare against the live one → activate
only if it actually improved.

---

## Evaluation

Two tiers, kept separate because they are not equally trustworthy.

**Deterministic (always available with `ground_truth_contexts`)** — `hit_rate@k`,
`mrr`, `context_precision@k`, `context_recall`. Pure arithmetic over ranks; identical
on every run. **These are the numbers to steer by offline.**

**LLM-judged (opt-in)** — RAGAS with a local judge (Ollama/Mistral by default). With
a small local model these are **indicative, not authoritative**; the report says so in
`warnings` rather than dressing the numbers up. Needs the optional extra:

```bash
uv pip install -e ".[eval]"
```

A missing or failing judge degrades to a warning and never loses the deterministic
results.

Answer-based metrics (faithfulness, answer_relevancy) need generation, which is
deferred; supplying a `reference_answer` produces an explicit "skipped" warning rather
than a silent omission.

**Reference matching** is the one judgement call: a reference may be a `chunk_id`
(exact match) or literal context text (rarely byte-identical to the chunk). The
matcher tries id equality → substring containment → token overlap coefficient
(threshold 0.8).

Eval sets are `[{question, ground_truth_contexts?, reference_answer?}]` as JSON or
JSONL. Runs persist with a config snapshot so scores stay comparable across reindexes.

---

## API

| Method | Path | Purpose |
|---|---|---|
| POST | `/index` | upload chunk JSONL (multipart) → `{index_job_id}` |
| GET | `/index`, `/index/{id}` | indexing progress (percent) + stats |
| POST | `/search` | hybrid/dense/sparse retrieval (+ parents, filters) |
| POST | `/search/compare` | the same query through all three modes |
| POST | `/evaluate` | run evaluation over a dataset |
| GET | `/evaluate`, `/evaluate/{id}` | eval progress + metrics |
| POST | `/reindex` | build a new collection, optional alias swap |
| GET | `/collections` | registry: models, dims, counts, active alias |
| POST | `/collections/{name}/activate` | manual alias swap / rollback |
| GET | `/health` | liveness (checks Qdrant + Redis) |

Indexing / reindex / evaluate are async (Celery); search is synchronous.

Progress is `processed_children / total_children`, written per batch.

---

## Configuration

Library settings use the **`RAG_EMBED_`** prefix; the microservice's infra settings use
**`RAG_EMBED_SVC_`**. See [`.env.example`](.env.example) for the annotated full list.

| Setting | Default | Purpose |
|---|---|---|
| `embedding_provider` | `sentence_transformers` | `sentence_transformers` \| `azure` |
| `local_embedding_model` | `all-MiniLM-L6-v2` | local dense model |
| `embedding_batch_size` | `64` | dense batch |
| `sparse_enabled` | `true` | compute BM25 |
| `bm25_tokenizer` / `bm25_k1` / `bm25_b` / `bm25_avg_len` | `default` / 1.2 / 0.75 / 256 | BM25 |
| `qdrant_url` | `http://qdrant:6333` | Qdrant endpoint |
| `collection_prefix` | `rag` | alias + base collection name |
| `hnsw_m` / `hnsw_ef_construct` / `hnsw_ef` | 16 / 128 / 64 | HNSW |
| `quantization` | `int8` | `none` \| `int8` \| `product` |
| `on_disk_vectors` / `on_disk_payload` | `true` | storage |
| `fusion` | `rrf` | `rrf` \| `dbsf` |
| `top_k` / `child_prefetch_k` | 10 / 50 | retrieval sizes |
| `with_parents` | `true` | small-to-big |
| `rerank_enabled` / `rerank_model` | `false` / `bge-reranker-base` | local cross-encoder |
| `rerank_candidates` / `rerank_batch_size` | 50 / 32 | rerank sizing |
| `docstore` | `sqlite` | `sqlite` \| `qdrant` |
| `ragas_llm_provider` / `ragas_model` | `ollama` / `mistral` | judge |
| `chunk_source` | `upload` | `upload` \| `path` (dev-only) |
| `max_upload_mb` | `500` | per-request JSONL cap |
| `offline` | `false` | air-gap enforcement (rejects Azure providers) |

---

## Air-gapped / offline operation

Default providers keep every byte of document text on the machine: local
sentence-transformers, in-repo BM25, a local Qdrant container, a local Ollama judge.

The **only** network touch is a one-time prefetch of the public embedding model into
the model-cache volume:

```bash
make prefetch          # online, once
make up                # HF_HUB_OFFLINE=1 + RAG_EMBED_OFFLINE=true from here on
```

`RAG_EMBED_OFFLINE=true` forces HF/transformers offline and disables telemetry, so a
missing model **fails loudly** instead of being silently downloaded — and it rejects
any `azure` provider at config load, before a single byte could egress.

The prefetch script deliberately *encodes a sentence* rather than just constructing
the model: sentence-transformers resolves tokenizer/pooling assets lazily, and a
construct-only prefetch leaves them out of the cache, which then fails at runtime.

---

## Architecture

```
rag_embedding/           # the reusable library (mypy --strict, provider-agnostic)
  config.py              # EmbeddingConfig (RAG_EMBED_*) + air-gap guardrail
  models.py              # Chunk contract, Search/Eval DTOs, CollectionSpec
  chunks.py              # JSONL loader, parent/child split, payload round-trip
  indexer.py             # index + reindex pipelines
  retrieval.py           # hybrid fusion + parent lookup + filters
  rerank.py              # local cross-encoder reranking
  embedders/             # base (Protocols), dense_local, dense_azure, sparse_bm25, factory
  store/                 # qdrant_store (lifecycle/upsert/query/aliases), docstore
  eval/                  # metrics (deterministic), ragas_eval (harness)
service_embed/           # FastAPI + Celery + SQLAlchemy around the library
ui/app.py                # Streamlit: index / search / evaluate / collections
docker/                  # Dockerfile, prefetch
alembic/                 # migrations (service tables + parent_chunks)
```

The library is provider- and transport-agnostic and fully unit-testable with a fake
embedder and an in-process Qdrant. The service adds jobs, persistence and HTTP — it
never re-implements embedding, indexing or retrieval.

---

## Testing

```bash
make test               # hermetic: fake embedder + in-process Qdrant, no network
make typecheck          # mypy --strict over library, service, ui, tests
make qdrant             # start a real Qdrant on :6333
make test-integration   # opt-in round-trip against it
```

Hermetic tests use `QdrantClient(":memory:")`, which really implements named vectors,
sparse vectors, Query API fusion and aliases — so the store assertions are behavioural,
not mock assertions. The integration tests cover what in-process mode cannot: that
quantization/on-disk settings and payload indexes are actually applied, and that HNSW
returns the right neighbour from a real index.

The dense embedder in tests is a deterministic fake built from a **token bag** rather
than a hash of the whole string, so texts sharing words land near each other — a pure
hash would make every vector orthogonal and every ranking assertion meaningless.

---

## Local dev (bare processes)

Needs a local Redis and Qdrant.

```bash
uv venv && uv pip install --torch-backend=cpu -e ".[service,dev]"
make qdrant
make migrate
make dev-api      # :8001
make dev-worker
make dev-ui       # :8502
```

## Out of scope

Document parsing/chunking (that is `rag_ingestion`), answer generation and chat,
multi-tenant auth, cross-machine clustering, GPU orchestration.
