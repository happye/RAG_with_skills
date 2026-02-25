# v0.5 Release Note (Public)

## Highlights

- Added a new vector retrieval backend: `qdrant`.
- Enabled backend comparison in existing benchmark flow.
- Preserved simple-first workflow and CLI compatibility.
- Fixed qdrant-client API compatibility discovered during runtime validation.

## What Changed

1. Retrieval backend upgrade
- New retriever option: `qdrant` in baseline query flow.
- Local vector storage and similarity query supported via qdrant-client.

2. Entry-point alignment
- `src/run.py`: supports `qdrant`; `quality` profile defaults to `qdrant`.
- `src/benchmark_runner.py`: supports `qdrant` for A/B evaluation.

3. Dependency update
- Added `qdrant-client` to requirements.

4. Compatibility fix
- For newer qdrant-client versions, use `query_points`.
- Keep fallback compatibility for clients exposing `search`.
- Collection lifecycle updated to avoid deprecated recreate flow.

## Validation Summary

- Confirmed Hugging Face cache works with existing `HF_HOME`.
- Confirmed offline model load path with `HF_HUB_OFFLINE=1`.
- Confirmed end-to-end smoke test with `retrieval-only + qdrant`.

## Usage Example

```bash
python src/rag_baseline.py --query "聊斋志异是什么" --provider retrieval-only --retriever qdrant --top-k 2
```

## Impact

- v0.5 completes the planned vector DB milestone with minimal UX burden.
- Existing retriever modes remain available for side-by-side comparison.
