"""Streamlit UI for the rag_embedding service (spec §12).

Four tabs: Index, Search playground, Evaluate, Collections/reindex. It talks to
the FastAPI service over HTTP only — no embedding, no Qdrant client, no library
imports. That keeps the UI container light and means anything you can do here you
can do with curl.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

import httpx
import streamlit as st

API_BASE = os.environ.get(
    "RAG_EMBED_SVC_API_BASE_URL", "http://localhost:8001")

st.set_page_config(page_title="rag_embedding", page_icon="🔎", layout="wide")


def _client() -> httpx.Client:
    return httpx.Client(base_url=API_BASE, timeout=120.0)


def _unwrap(resp: httpx.Response) -> Any:
    """Return the JSON body, or surface the API's own error text and stop.

    The service answers bad requests with a specific, actionable ``detail``
    (dimension mismatch, no active collection, no reranker configured). Letting
    httpx raise would replace all of that with a stack trace, so the message is
    shown as-is instead.
    """
    if resp.is_success:
        return resp.json()
    try:
        detail = resp.json().get("detail", resp.text)
    except ValueError:
        detail = resp.text
    st.error(f"{resp.status_code} from {resp.request.url.path}: {detail}")
    st.stop()


def api_get(path: str, **params: Any) -> Any:
    with _client() as client:
        return _unwrap(client.get(path, params=params or None))


def api_post(path: str, **kwargs: Any) -> Any:
    with _client() as client:
        return _unwrap(client.post(path, **kwargs))


def health() -> dict[str, Any] | None:
    try:
        with _client() as client:
            resp = client.get("/health")
            return dict(resp.json()) if resp.status_code == 200 else None
    except httpx.HTTPError:
        return None


# --------------------------------------------------------------------------- #
st.title("🔎 rag_embedding")
st.caption(
    "Hybrid indexing & retrieval over stage-1 chunks — nothing leaves this machine.")

_status = health()
if _status is None:
    st.error(
        f"API not reachable at {API_BASE}. Start it (see README) and reload.")
    st.stop()
# st.stop() raises internally but is not typed NoReturn, so re-bind for the checker.
status: dict[str, Any] = _status or {}

# None unless RAG_EMBED_RERANK_ENABLED=true on the API/worker.
RERANKER: str | None = status.get("reranker")

cols = st.columns(5)
cols[0].metric("API", status["status"])
cols[1].metric("Qdrant", status["qdrant"].split(":")[0])
cols[2].metric("Redis", status["redis"].split(":")[0])
cols[3].metric("Serving", status["active_collection"] or "—")
cols[4].metric(
    "Reranker",
    RERANKER.split(":")[-1].split("/")[-1] if RERANKER else "off",
    help=RERANKER or "set RAG_EMBED_RERANK_ENABLED=true to enable",
)

tab_index, tab_search, tab_eval, tab_collections = st.tabs(
    ["📥 Index", "🔍 Search", "📊 Evaluate", "🗃️ Collections"]
)


# --------------------------------------------------------------------------- #
# Index
# --------------------------------------------------------------------------- #
with tab_index:
    st.subheader("Index stage-1 chunks")
    st.caption(
        "Upload the `chunks.jsonl` (or `.jsonl.gz`) that rag_ingestion produced.")

    uploads = st.file_uploader(
        "Chunk files", type=["jsonl", "gz", "ndjson", "json"], accept_multiple_files=True
    )
    if st.button("Index", type="primary", disabled=not uploads):
        multipart = [("files", (f.name, f.getvalue()))
                     for f in (uploads or [])]
        created = api_post("/index", files=multipart)
        st.session_state["index_job_id"] = created["index_job_id"]
        st.success(
            f"Job {created['index_job_id'][:8]} queued → {created['collection']}")

    manual = st.text_input("…or open an existing index job id")
    if manual:
        st.session_state["index_job_id"] = manual.strip()

    job_id = st.session_state.get("index_job_id")
    if job_id:
        job = api_get(f"/index/{job_id}")
        st.markdown(
            f"**Job `{job_id[:12]}`** — {job['kind']} → `{job['collection']}` — {job['status']}")
        st.progress(job["progress"] / 100.0, text=f"{job['progress']}%")

        m = st.columns(4)
        m[0].metric("Files", job["total_files"])
        m[1].metric("Children indexed", job["children_indexed"])
        m[2].metric("Parents stored", job["parents_stored"])
        m[3].metric("Failed", job["files_failed"])

        st.dataframe(
            [
                {
                    "file": f["filename"],
                    "status": f["status"],
                    "progress": f"{f['progress']}%",
                    "children": f"{f['processed_children']}/{f['total_children']}",
                    "parents": f["parents_stored"],
                    "skipped": f["skipped"],
                    "seconds": f["seconds"],
                    "warnings": "; ".join(f["warnings"]),
                    "error": f["error"] or "",
                }
                for f in job["files"]
            ],
            use_container_width=True,
            hide_index=True,
        )

        if job["status"] not in ("done", "partial", "failed"):
            time.sleep(1.5)
            st.rerun()


# --------------------------------------------------------------------------- #
# Search playground
# --------------------------------------------------------------------------- #
def _render_hits(hits: list[dict[str, Any]]) -> None:
    if not hits:
        st.info("no hits")
        return
    st.dataframe(
        [
            {
                "#": i + 1,
                "score": round(h["score"], 4),
                # Different scale from `score` once a cross-encoder has run.
                "fused": (
                    round(h["retrieval_score"], 4)
                    if h.get("retrieval_score") is not None
                    else ""
                ),
                "chunk_id": h["chunk_id"],
                "type": h["payload"].get("block_type", ""),
                "section": h["payload"].get("section_path_text", ""),
                "text": h["text"][:160].replace("\n", " "),
            }
            for i, h in enumerate(hits)
        ],
        use_container_width=True,
        hide_index=True,
    )


with tab_search:
    st.subheader("Search playground")
    query = st.text_input(
        "Query", placeholder="how does retrieval fuse dense and sparse?")

    c1, c2, c3, c4, c5 = st.columns([2, 1, 1, 1, 1])
    mode = c1.radio("Mode", ["hybrid", "dense", "sparse"], horizontal=True)
    top_k = c2.number_input("top_k", min_value=1, max_value=100, value=10)
    with_parents = c3.checkbox("Parent context", value=True)
    compare = c4.checkbox("Compare modes", value=False)
    # Offering "on" against a service with no reranker would only produce a 400.
    rerank_choice = c5.selectbox(
        "Rerank",
        ["default", "on", "off"] if RERANKER else ["default"],
        help=(
            f"cross-encoder: {RERANKER}"
            if RERANKER
            else "no reranker configured — set RAG_EMBED_RERANK_ENABLED=true and "
            "re-run `make prefetch` to cache the model"
        ),
        disabled=RERANKER is None,
    )

    with st.expander("Filters"):
        f1, f2, f3, f4 = st.columns(4)
        filters = {
            "doc_id": [x for x in f1.text_input("doc_id (comma-sep)").split(",") if x.strip()],
            "file_type": [x for x in f2.text_input("file_type").split(",") if x.strip()],
            "block_type": [x for x in f3.text_input("block_type").split(",") if x.strip()],
            "section_path": [x for x in f4.text_input("section_path").split(",") if x.strip()],
        }
        filters = {k: [v.strip() for v in vals] for k, vals in filters.items()}

    if st.button("Search", type="primary", disabled=not query):
        body: dict[str, Any] = {
            "query": query,
            "mode": mode,
            "top_k": int(top_k),
            "with_parents": with_parents,
            "filters": filters,
        }
        if rerank_choice != "default":
            body["rerank"] = rerank_choice == "on"
        if compare:
            # Three real queries, one per mode — what each would return in prod.
            results = api_post("/search/compare", json=body)
            columns = st.columns(len(results))
            all_ids = {m: {h["chunk_id"] for h in r["hits"]}
                       for m, r in results.items()}
            for col, (m, resp) in zip(columns, results.items(), strict=True):
                with col:
                    st.markdown(
                        f"**{m}** — {resp['timings'].get('search_ms', 0)} ms")
                    others = set().union(*(v for k, v in all_ids.items() if k != m))
                    for i, hit in enumerate(resp["hits"], start=1):
                        unique = "🆕 " if hit["chunk_id"] not in others else ""
                        st.markdown(
                            f"{i}. {unique}`{hit['chunk_id'][:12]}` "
                            f"({hit['score']:.4f})<br/><small>{hit['text'][:120]}</small>",
                            unsafe_allow_html=True,
                        )
            st.caption(
                "🆕 marks a chunk only that mode found — the hybrid benefit, made visible.")
        elif rerank_choice == "on":
            # A/B the cross-encoder on this exact query: same candidates, one
            # reordered by the reranker, so the effect is visible rather than
            # asserted.
            before = api_post("/search", json={**body, "rerank": False})
            after = api_post("/search", json={**body, "rerank": True})
            left, right = st.columns(2)
            with left:
                st.markdown(
                    f"**fused only** — {before['timings'].get('search_ms', 0)} ms")
                _render_hits(before["hits"])
            with right:
                st.markdown(
                    f"**+ cross-encoder** — {after['timings'].get('rerank_ms', 0)} ms rerank"
                )
                _render_hits(after["hits"])
            moved = sum(
                1
                for i, h in enumerate(after["hits"])
                if i >= len(before["hits"]) or before["hits"][i]["chunk_id"] != h["chunk_id"]
            )
            st.caption(
                f"{moved} of {len(after['hits'])} positions changed after reranking.")
        else:
            resp = api_post("/search", json=body)
            st.caption(
                f"collection `{resp['collection']}` · "
                + ("reranked · " if resp.get("reranked") else "")
                + " · ".join(f"{k} {v}ms" for k, v in resp["timings"].items())
            )
            st.markdown("#### Child hits")
            _render_hits(resp["hits"])
            if resp["parents"]:
                st.markdown("#### Parent context (small-to-big)")
                for parent in resp["parents"]:
                    with st.expander(
                        f"{parent['chunk_id'][:12]} · score {parent['score']:.4f} · "
                        f"{len(parent['children'])} matched child(ren)"
                    ):
                        st.write(parent["text"])
                        st.caption(
                            "matched children: "
                            + ", ".join(c["chunk_id"][:12]
                                        for c in parent["children"])
                        )


# --------------------------------------------------------------------------- #
# Evaluate
# --------------------------------------------------------------------------- #
with tab_eval:
    st.subheader("Evaluate retrieval")
    st.caption(
        "Upload `[{question, ground_truth_contexts?, reference_answer?}]` as JSON or JSONL. "
        "hit-rate/MRR/precision/recall need no LLM; the judged tier is opt-in and only "
        "indicative with a small local judge."
    )

    eval_file = st.file_uploader(
        "Eval set", type=["json", "jsonl"], key="evalset")
    e1, e2, e3, e4 = st.columns(4)
    eval_mode = e1.selectbox("Mode", ["hybrid", "dense", "sparse"])
    eval_k = e2.number_input("top_k", min_value=1,
                             max_value=100, value=10, key="evalk")
    eval_rerank = e3.selectbox(
        "Rerank",
        ["default", "on", "off"] if RERANKER else ["default"],
        key="evalrr",
        disabled=RERANKER is None,
    )
    llm_metrics = e4.checkbox("LLM-judged metrics (slow)", value=False)

    if st.button("Run evaluation", type="primary", disabled=eval_file is None):
        created = api_post(
            "/evaluate",
            files={"file": (eval_file.name, eval_file.getvalue())},
            data={
                "mode": eval_mode,
                "top_k": int(eval_k),
                "llm_metrics": str(llm_metrics).lower(),
                **(
                    {}
                    if eval_rerank == "default"
                    else {"rerank": str(eval_rerank == "on").lower()}
                ),
            },
        )
        st.session_state["eval_run_id"] = created["eval_run_id"]
        st.success(f"Eval run {created['eval_run_id'][:8]} queued")

    run_id = st.session_state.get("eval_run_id")
    if run_id:
        run = api_get(f"/evaluate/{run_id}")
        st.markdown(
            f"**Run `{run_id[:12]}`** — {run['status']} on `{run['collection']}`")
        if run["error"]:
            st.error(run["error"])
        if run["aggregate"]:
            metric_cols = st.columns(len(run["aggregate"]))
            for col, (name, value) in zip(metric_cols, run["aggregate"].items(), strict=True):
                col.metric(name, f"{value:.3f}")
        for warning in run["warnings"]:
            st.warning(warning)
        if run["items"]:
            st.dataframe(
                [
                    {
                        "question": item["question"],
                        **{k: round(v, 3) for k, v in item["metrics"].items()},
                        "retrieved": ", ".join(c[:10] for c in item["retrieved_chunk_ids"][:5]),
                        "note": item["note"] or "",
                    }
                    for item in run["items"]
                ],
                use_container_width=True,
                hide_index=True,
            )
        if run["status"] in ("queued", "running"):
            time.sleep(2.0)
            st.rerun()

    st.markdown("#### Trend across runs")
    runs = api_get("/evaluate", limit=20)
    scored = [r for r in runs if r["aggregate"]]
    if scored:
        st.dataframe(
            [
                {
                    "run": r["id"][:8],
                    "when": r["created_at"],
                    "collection": r["collection"],
                    "mode": r["mode"],
                    **{k: round(v, 3) for k, v in r["aggregate"].items()},
                }
                for r in scored
            ],
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "Compare a candidate collection's scores against the live one before swapping.")
    else:
        st.info("No scored runs yet.")


# --------------------------------------------------------------------------- #
# Collections / reindex
# --------------------------------------------------------------------------- #
with tab_collections:
    st.subheader("Collections")
    registry = api_get("/collections")
    st.caption(
        f"Serving alias `{registry['alias']}` → `{registry['active_collection'] or '—'}`")

    if registry["collections"]:
        st.dataframe(
            [
                {
                    "active": "✅" if c["is_active"] else "",
                    "name": c["name"],
                    "v": c["version"],
                    "model": c["model_id"],
                    "dim": c["dim"],
                    "points": c["point_count"],
                    "hnsw m/ef": f"{c['hnsw_m']}/{c['hnsw_ef_construct']}",
                    "quant": c["quantization"],
                    "sparse": c["sparse_enabled"],
                    "created": c["created_at"],
                }
                for c in registry["collections"]
            ],
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("#### Reindex")
    st.caption(
        "Rebuilds from stored payloads under the *current* config — change the model or HNSW "
        "settings in the environment first. The live collection keeps serving until you activate."
    )
    reason = st.text_input("Reason (stored on the job for auditability)")
    activate_now = st.checkbox(
        "Activate automatically when it finishes", value=False)
    if st.button("Start reindex", type="primary"):
        created = api_post(
            "/reindex", json={"reason": reason or None, "activate_on_success": activate_now}
        )
        st.session_state["index_job_id"] = created["index_job_id"]
        st.success(
            f"Reindex {created['index_job_id'][:8]} → `{created['collection']}`. "
            "Watch it on the Index tab."
        )

    st.markdown("#### Activate / roll back")
    names = [c["name"] for c in registry["collections"]]
    if names:
        target = st.selectbox("Collection", names)
        if st.button("Point the alias here"):
            out = api_post(f"/collections/{target}/activate")
            st.success(f"`{out['alias']}` → `{out['collection']}`")
            st.rerun()

    with st.expander("Raw registry JSON"):
        st.code(json.dumps(registry, indent=2), language="json")
