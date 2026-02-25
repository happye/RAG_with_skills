# v0.5 Summary

## Version Scope

v0.5 focuses on vector database integration as the core retrieval upgrade, while preserving simple-first usage and existing retriever compatibility.

## What Was Added

1. Qdrant retriever mode
- Added `--retriever qdrant` in `src/rag_baseline.py`.
- Added local Qdrant-backed semantic retrieval path:
  - embedding generation via sentence-transformers
  - vector storage/query via qdrant-client
  - top-k retrieval output in existing chunk format

2. Qdrant backend lifecycle (minimal)
- Added per-corpus collection naming (auto-generated when not specified).
- Added local backend controls:
  - `RAG_QDRANT_PATH`
  - `RAG_QDRANT_COLLECTION`
  - `RAG_QDRANT_RECREATE`
- Added in-process collection cache to avoid repeated setup in the same run.

3. End-to-end entrypoint integration
- `src/run.py`
  - Added `qdrant` to retriever choices.
  - Updated `quality` mode default retriever to `qdrant`.
- `src/benchmark_runner.py`
  - Added `qdrant` to retriever choices for benchmark A/B.

4. Dependency and docs updates
- Added `qdrant-client>=1.12.0` to `requirements.txt`.
- Updated README retriever docs and examples to include `qdrant`.

## Critical Fix During v0.5 Validation

Issue found:
- `qdrant-client` (installed version 1.17.0) no longer exposed `QdrantClient.search` in this environment.

Fix applied:
- Updated query path in `src/rag_baseline.py`:
  - Prefer `query_points` when available.
  - Keep fallback path for clients exposing `search`.
- Updated collection lifecycle:
  - Replaced deprecated `recreate_collection` flow with `create_collection` + optional explicit delete on recreate.

Impact:
- Restores runtime compatibility with newer qdrant-client API.
- Keeps behavior stable across mixed client versions.

## Validation Snapshot

1. Hugging Face cache behavior check
- Confirmed `HF_HOME` is read correctly in terminal environment.
- Confirmed model path exists under:
  - `C:\hf_cache\hub\models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2`

2. Offline model load check
- Ran with `HF_HUB_OFFLINE=1`.
- sentence-transformers model loaded from local cache successfully (no forced re-download).

3. Qdrant end-to-end smoke test
- Minimal corpus test with `--retriever qdrant` + `--provider retrieval-only` passed.
- Retrieval summary and evidence output returned correctly.

## Notes for Current Use

Recommended first command (quick verify):

```bash
python src/rag_baseline.py --query "聊斋志异是什么" --provider retrieval-only --retriever qdrant --top-k 2
```

If you need strict offline embedding behavior during testing:

```powershell
$env:HF_HUB_OFFLINE="1"
```

## Version Outcome

v0.5 achieves the milestone objective of introducing a vector DB backend (local Qdrant), keeps simple CLI usage, and resolves a real compatibility issue discovered during runtime verification.

## Next Candidate Goals (v0.6)

1. Reranker v2 (semantic/cross-encoder or provider rerank API).
2. Better retrieval precision tracking on expanded Chinese eval set.
3. Optional backend profiling report (index time/query latency/space) for data-driven tuning.
