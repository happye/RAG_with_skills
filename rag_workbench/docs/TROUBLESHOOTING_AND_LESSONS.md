# Troubleshooting and Lessons Learned (v0.1)

This document captures the major issues encountered during the initial build and how each one was resolved.

## 1. Validation Script Fails: `No module named yaml`

### Symptom

- Running `quick_validate.py` failed with:
  - `ModuleNotFoundError: No module named 'yaml'`

### Root Cause

- `PyYAML` was missing in the exact Python runtime used to execute the script.

### Fix

- Install with the same interpreter used for execution:

```bash
c:/python314/python.exe -m pip install PyYAML
```

### Prevention

- Always install dependencies using the same interpreter used for runtime commands.

## 2. API Key Parsed Incorrectly from `.env`

### Symptom

- Authentication looked correct but provider requests still failed.

### Root Cause

- API keys were wrapped in quotes in `.env` (for example `KIMI_API_KEY='sk-...'`).

### Fix

- Remove wrapping quotes from keys.
- Added environment value sanitizer in code to strip wrapping quotes safely.

### Prevention

- Keep `.env` values plain unless escaping is explicitly required.

## 3. Kimi 429 Errors

### Symptom

- `429 Client Error: Too Many Requests`

### Root Cause

- Rate/concurrency/quota limits at provider side.

### Fix

- Added retry with backoff behavior.
- Added clearer error messaging for 429.
- Recommended checking account quota and provider limit settings.

### Prevention

- Use small smoke runs first (`--max-questions 1`) before full benchmark loops.

## 4. Kimi 400: `invalid temperature: only 1 is allowed for this model`

### Symptom

- Kimi request failed for `kimi-k2.5`.

### Root Cause

- Model-specific parameter constraints were not respected.

### Fix

- Added provider/model-specific request logic:
  - For `kimi-k2.5`, enforce `temperature=1`.

### Prevention

- Keep provider/model specific request rules explicit in code.

## 5. Empty Answer with `finish_reason=length`

### Symptom

- Response returned:
  - `kimi returned empty content (finish_reason=length)`

### Root Cause

- Provider can end generation by length and return no usable content in edge cases.

### Fix

- Increased Kimi completion token defaults.
- Added one retry path with compact-answer instruction.
- Added fallback to retrieval evidence output when generation is empty.

### Prevention

- Tune `KIMI_MAX_COMPLETION_TOKENS` and `KIMI_MAX_COMPLETION_TOKENS_CAP` based on task length.

## 6. Chinese Retrieval Returned Zero Chunks

### Symptom

- `Retrieved chunks: 0` for Chinese questions on Chinese corpus.

### Root Cause

- Two combined issues:
  - Chinese tokenization was not retrieval-friendly.
  - Source file encoding mismatch (UTF-16 content read as UTF-8) caused corrupted text ingestion.

### Fix

- Added CJK-aware tokenization.
- Added char n-gram TF-IDF mode for CJK queries/corpora.
- Added automatic text decoding strategy (`utf-8-sig`, `utf-16`, `utf-16-le`, `utf-16-be`, `gb18030`).

### Prevention

- For Chinese corpora, use `--retriever tfidf` or `--retriever hybrid`.

## 7. Confusion About Which Files Are Indexed

### Symptom

- User expected a `data/bak` folder to be ignored.

### Root Cause

- Loader recursively indexes all `.md/.txt` files under the selected `--data-dir`.

### Fix

- Moved backup files outside active data root (`data_bak`).

### Prevention

- Keep active corpus directory clean.
- Use a separate folder and `--data-dir` for isolated tests.

## 8. Long or Stalled Runtime During Provider Calls

### Symptom

- Slow response or interrupted terminal runs.

### Root Cause

- Network/provider latency and long generation.

### Fix

- Added configurable HTTP timeouts:
  - `RAG_HTTP_CONNECT_TIMEOUT`
  - `RAG_HTTP_READ_TIMEOUT`
- Added smoke test strategy with `--max-questions 1`.

### Prevention

- Start with fast smoke tests, then scale to full benchmark runs.

## Recommended Debug Order

1. Confirm corpus loading and encoding.
2. Validate retrieval hit count first (`Retrieved chunks`).
3. Validate provider auth and model parameters.
4. Tune output limits and timeout settings.
5. Run benchmark and compare logs before/after changes.
