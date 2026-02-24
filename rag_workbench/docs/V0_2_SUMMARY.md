# v0.2 Summary

## Version Scope

v0.2 focuses on improving retrieval quality and evaluation observability while keeping the generation layer stable.

## What Was Added

1. Embedding retriever mode
- Added `--retriever embedding` in `src/rag_baseline.py`.
- Uses sentence-transformers model:
  - default: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- Added embedding model/index cache for runtime reuse.

2. Lightweight reranker layer
- Added `--reranker` options:
  - `none`
  - `keyword`
  - `tfidf`
- Added `--rerank-pool` for candidate pool size before reranking.
- Integrated into both single-query and benchmark flows.

3. Benchmark diagnostics upgrade
- Added stage classification per case:
  - `ok`
  - `retrieval_miss`
  - `retrieval_low_signal`
  - `generation_empty_or_timeout`
- Added stage aggregate counts in summary output.
- Added failed/weak case export to `logs/failed_cases.jsonl`.

4. Eval path usability improvement
- If default `data/eval_set.jsonl` is missing, benchmark auto-fallbacks to `data_bak/eval_set.jsonl`.

5. Liaozhai-specific eval set
- Added `data/eval_set_liaozhai.jsonl` for Chinese corpus A/B testing.

## Chinese Corpus Readiness

v0.2 builds on prior Chinese fixes from v0.1.x stream:
- CJK-aware tokenization.
- CJK char n-gram TF-IDF retrieval mode.
- Auto text decoding (`utf-8-sig`, `utf-16`, `utf-16-le`, `utf-16-be`, `gb18030`).

## A/B Snapshot (Liaozhai, retrieval-only)

Dataset:
- `data/eval_set_liaozhai.jsonl`
- `eval_count = 5`

Results:
- `hybrid + tfidf reranker`
  - `retrieval_keyword_hit_avg = 0.9`
  - `answer_keyword_hit_avg = 0.9`
- `embedding + no reranker`
  - `retrieval_keyword_hit_avg = 0.8`
  - `answer_keyword_hit_avg = 0.9`

Interpretation:
- On this small Chinese test set, `hybrid + tfidf reranker` is currently the better retrieval default.
- Embedding mode is functional and ready for broader eval, but not yet superior on this specific set.

## Operational Notes

1. Embedding first-run downloads
- First run may download a large model from Hugging Face.
- Interrupted download can be resumed by rerunning benchmark.

2. Windows cache symlink warning
- Non-blocking warning on systems without symlink support.
- Functionality remains correct; disk usage may be higher.

## Version Control

- Release commit: `7811a7f`
- Git tag: `v0.2.0`
- Backup archive: `backups/rag_workbench_v0_2_20260224_165721.zip`

## Recommended Default for Current Corpus

For `data/《聊斋志异》白话文（经典完整版）.md`:
- retriever: `hybrid`
- reranker: `tfidf`
- top-k: `3`

Example:

```bash
python src/rag_baseline.py --query "请概括《聊斋志异》的主题和写作风格，并给出两点证据。" --provider kimi --model kimi-k2.5 --retriever hybrid --reranker tfidf --top-k 3
```

## Next Candidate Goals (v0.3)

1. Stronger reranker (semantic/cross-encoder or provider rerank API).
2. Better metrics than keyword-hit only (faithfulness and support attribution).
3. Automated prompt/context compression for latency and token control.
4. Expanded Chinese eval set for higher-confidence conclusions.
