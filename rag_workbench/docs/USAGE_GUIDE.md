# RAG Workbench Usage Guide

This guide explains how to use the current project from quick testing to repeatable tuning.

## 1. Prerequisites

- Python available in your environment.
- Dependencies installed:

```bash
python -m pip install -r requirements.txt
```

- At least one provider API key configured (for generation mode).

## 2. Project Structure

- `src/rag_baseline.py`: single-query RAG execution.
- `src/benchmark_runner.py`: repeatable benchmark run on eval set.
- `src/tune_hybrid_alpha.py`: hybrid retriever weight sweep.
- `data/`: active corpus files (`.md`, `.txt`).
- `logs/`: benchmark and sweep outputs.

## 3. Basic Query Run

Run one RAG query:

```bash
python src/rag_baseline.py --query "What does this corpus focus on?" --provider kimi --model kimi-k2.5 --retriever hybrid
```

### Core CLI Parameters

- `--data-dir`: corpus root folder (default: `data`).
- `--query`: user question.
- `--provider`: model provider (`kimi`, `openai`, `gemini`, etc. or `retrieval-only`).
- `--model`: provider model override.
- `--retriever`: `keyword`, `tfidf`, or `hybrid`.
- `--top-k`: number of retrieved chunks.
- `--chunk-size`: chunk length in characters.
- `--overlap`: chunk overlap in characters.

## 4. Chinese Corpus Best Practice

For Chinese books/articles:

- Prefer:
  - `--retriever tfidf`
  - or `--retriever hybrid`
- Keep `top-k` small at first (2-3) to reduce noise and token cost.

Example:

```bash
python src/rag_baseline.py --query "请概括主题并给出证据" --provider kimi --model kimi-k2.5 --retriever hybrid --top-k 3
```

## 5. Isolated Corpus Testing

To test only one corpus folder:

```bash
python src/rag_baseline.py --data-dir ../books/liaozhai_only --query "..." --provider kimi --model kimi-k2.5 --retriever hybrid
```

Important:

- The loader recursively indexes all `.md/.txt` under `--data-dir`.

## 6. Benchmark Workflow

Prepare eval set:

- Edit `data/eval_set.jsonl` (one JSON object per line).
- Required field: `question`.
- Optional fields: `id`, `expected_keywords`.

Run benchmark:

```bash
python src/benchmark_runner.py --provider kimi --model kimi-k2.5 --retriever hybrid
```

Fast smoke run:

```bash
python src/benchmark_runner.py --provider kimi --model kimi-k2.5 --retriever hybrid --max-questions 1
```

Benchmark output is stored in:

- `logs/eval_runs.jsonl`

## 7. Hybrid Alpha Tuning

Run alpha sweep:

```bash
python src/tune_hybrid_alpha.py --provider kimi --model kimi-k2.5 --alphas 0.2,0.5,0.8
```

Fast sweep:

```bash
python src/tune_hybrid_alpha.py --provider kimi --model kimi-k2.5 --alphas 0.2,0.5,0.8 --max-questions 1
```

Sweep output is stored in:

- `logs/hybrid_alpha_sweep.jsonl`

## 8. Environment Variables

### Provider Keys

- `KIMI_API_KEY`
- `OPENAI_API_KEY`
- `GEMINI_API_KEY`
- `ANTHROPIC_API_KEY`
- and other provider keys as needed

### Kimi-Specific

- `KIMI_MODEL`
- `KIMI_TEMPERATURE` (for model families that allow tuning)
- `KIMI_MAX_COMPLETION_TOKENS`
- `KIMI_MAX_COMPLETION_TOKENS_CAP`

### Runtime Reliability

- `RAG_REQUEST_MAX_RETRIES`
- `RAG_HTTP_CONNECT_TIMEOUT`
- `RAG_HTTP_READ_TIMEOUT`

## 9. Interpreting Retrieval Summary

- `Indexed chunks`: total chunks created from corpus.
- `Retrieved chunks`: chunks selected for current question.
- If `Retrieved chunks = 0`:
  - check corpus encoding/content,
  - switch to `tfidf`/`hybrid`,
  - verify question-corpus overlap.

## 10. Suggested Default Profiles

### Fast Iteration

- `--retriever tfidf`
- `--top-k 2`
- `--chunk-size 350`
- `--overlap 40`
- benchmark with `--max-questions 1`

### Better Coverage

- `--retriever hybrid`
- `--top-k 3`
- `--chunk-size 500`
- `--overlap 80`

## 11. Troubleshooting Entry Point

If anything fails, check:

- `docs/TROUBLESHOOTING_AND_LESSONS.md`

## 12. v0.2 Progress Notes

- Completed in code:
  - `embedding` retriever mode
  - lightweight reranker (`none`, `keyword`, `tfidf`)
  - benchmark diagnostics (`stage_counts`, failed case export)
- Current blocker for embedding A/B:
  - first run needs to download a sentence-transformer model (hundreds of MB), which can be interrupted on unstable network.
- If interrupted, rerun the same benchmark command. Download resumes from cache in most cases.

## 13. Liaozhai A/B Snapshot (v0.2)

Dataset:

- `data/eval_set_liaozhai.jsonl`
- `eval_count = 5`

Retrieval-only comparison:

- `hybrid + tfidf reranker`
  - `retrieval_keyword_hit_avg = 0.9`
  - `answer_keyword_hit_avg = 0.9`
- `embedding + no reranker`
  - `retrieval_keyword_hit_avg = 0.8`
  - `answer_keyword_hit_avg = 0.9`

Interpretation:

- On this small Chinese eval set, `hybrid + tfidf reranker` currently gives better retrieval hit rate.
- Answer hit rate is tied, so keep `hybrid + tfidf reranker` as the default for this corpus.
