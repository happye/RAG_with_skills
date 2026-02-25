# RAG Workbench (0 -> 1 Baseline)

This folder contains a minimal, runnable RAG baseline for learning and iteration.

## Additional Docs

- Usage Guide: `docs/USAGE_GUIDE.md`
- Troubleshooting and Lessons: `docs/TROUBLESHOOTING_AND_LESSONS.md`
- v0.2 Summary: `docs/V0_2_SUMMARY.md`
- Milestones Roadmap: `docs/MILESTONES.md`

## What it does

- Loads local `.txt` and `.md` files from `data/`
- Splits documents into chunks
- Retrieves top-k chunks with pluggable strategies: `keyword`, `tfidf`, or `hybrid`
- Generates answers with multiple providers: Anthropic, OpenAI (ChatGPT), Gemini, DeepSeek, Qwen, Kimi, GLM, Doubao
- Falls back to grounded extractive output when no API key is present

## Quick start

1. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

2. Configure API keys (choose at least one provider):

- Option A: environment variables in shell
- Option B: copy `.env.example` to `.env` and fill values (auto-loaded by script)

Windows PowerShell examples:

```powershell
# Anthropic
$env:ANTHROPIC_API_KEY="your_anthropic_key"

# OpenAI / ChatGPT
$env:OPENAI_API_KEY="your_openai_key"

# Gemini
$env:GEMINI_API_KEY="your_gemini_key"

# DeepSeek
$env:DEEPSEEK_API_KEY="your_deepseek_key"

# Qwen
$env:QWEN_API_KEY="your_qwen_key"

# Kimi
$env:KIMI_API_KEY="your_kimi_key"

# GLM
$env:GLM_API_KEY="your_glm_key"

# Doubao
$env:DOUBAO_API_KEY="your_doubao_key"
```

3. Put source files in `data/`.

Encoding note:

- Text loading supports common encodings including UTF-8, UTF-16, and GB18030.
- This improves ingestion for Chinese book files exported from different tools.

4. Run a query:

```bash
python src/rag_baseline.py --query "What does the project focus on?" --top-k 3 --provider auto --retriever keyword
```

Retriever options:

- `keyword`: token overlap baseline
- `tfidf`: lexical-semantic retrieval with n-gram TF-IDF cosine similarity
- `hybrid`: weighted mix of keyword and tfidf (set `RAG_HYBRID_ALPHA` in env)
- `embedding`: semantic retrieval using sentence-transformers embeddings

Reranker options:

- `none`: no second-stage reranking
- `keyword`: token-overlap reranking over retrieved candidates
- `tfidf`: TF-IDF reranking over retrieved candidates

Embedding setup:

```bash
python -m pip install sentence-transformers
```

Optional embedding model override:

- `RAG_EMBEDDING_MODEL` (default: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`)

Chinese text note:

- The retriever now supports Chinese by using CJK-aware tokenization and char n-gram TF-IDF.
- For Chinese corpora, start with `--retriever tfidf` or `--retriever hybrid`.

5. Select provider explicitly (recommended for experiments):

```bash
python src/rag_baseline.py --query "..." --provider anthropic
python src/rag_baseline.py --query "..." --provider openai
python src/rag_baseline.py --query "..." --provider gemini
python src/rag_baseline.py --query "..." --provider deepseek
python src/rag_baseline.py --query "..." --provider qwen
python src/rag_baseline.py --query "..." --provider kimi
python src/rag_baseline.py --query "..." --provider glm
python src/rag_baseline.py --query "..." --provider doubao
python src/rag_baseline.py --query "..." --provider retrieval-only
python src/rag_baseline.py --query "..." --retriever embedding --reranker tfidf --top-k 3 --rerank-pool 12
```

6. Optional model override:

```bash
python src/rag_baseline.py --query "..." --provider openai --model gpt-4o
```

## Simple Run (Recommended)

Use `src/run.py` with one config file and mode profiles.

1. Edit `config.yaml` (set at least `query`).
2. Run:

```bash
python src/run.py
```

3. Optional mode switch:

```bash
python src/run.py --mode fast
python src/run.py --mode balanced
python src/run.py --mode quality
```

4. Optional one-off overrides:

```bash
python src/run.py --query "请概括《聊斋志异》的主题" --top-k 4 --retriever hybrid
```

Override priority:

- CLI args > `config.yaml` > mode defaults

## Provider env var map

- `anthropic`: `ANTHROPIC_API_KEY`, optional `ANTHROPIC_MODEL`
- `openai`: `OPENAI_API_KEY`, optional `OPENAI_MODEL`, `OPENAI_BASE_URL`
- `gemini`: `GEMINI_API_KEY`, optional `GEMINI_MODEL`
- `deepseek`: `DEEPSEEK_API_KEY`, optional `DEEPSEEK_MODEL`, `DEEPSEEK_BASE_URL`
- `qwen`: `QWEN_API_KEY`, optional `QWEN_MODEL`, `QWEN_BASE_URL`
- `kimi`: `KIMI_API_KEY`, optional `KIMI_MODEL`, `KIMI_BASE_URL`
- `glm`: `GLM_API_KEY`, optional `GLM_MODEL`, `GLM_BASE_URL`
- `doubao`: `DOUBAO_API_KEY`, optional `DOUBAO_MODEL`, `DOUBAO_BASE_URL`

## Kimi troubleshooting

- If you see `invalid temperature: only 1 is allowed for this model`, use `KIMI_MODEL=kimi-k2.5` with `KIMI_TEMPERATURE=1`.
- The script now applies this automatically for `kimi-k2.5` models.
- If you hit `429`, check account balance and rate/concurrency limits in Moonshot console.
- If you see `kimi returned empty content (finish_reason=length)`, increase:
	- `KIMI_MAX_COMPLETION_TOKENS` (default now `1200`)
	- `KIMI_MAX_COMPLETION_TOKENS_CAP` (default now `4000`)
	The script also retries once with a compact-answer instruction.

## Next tuning steps

- Replace lexical retrieval with embedding-based retrieval.
- Add reranking.
- Add a benchmark set and scorecard.
- Add experiment logging for each configuration change.

## Benchmark loop (Kimi-first)

Use the built-in benchmark runner to track iteration quality.

1. Edit `data/eval_set.jsonl` with your own questions and expected keywords.
	If this file is moved to `data_bak/eval_set.jsonl`, benchmark runner auto-fallbacks to that path.
2. Run benchmark with Kimi:

```bash
python src/benchmark_runner.py --provider kimi --model kimi-k2.5
python src/benchmark_runner.py --provider kimi --model kimi-k2.5 --retriever tfidf
python src/benchmark_runner.py --provider kimi --model kimi-k2.5 --retriever hybrid
python src/benchmark_runner.py --provider kimi --model kimi-k2.5 --retriever embedding
python src/benchmark_runner.py --provider kimi --model kimi-k2.5 --retriever keyword --max-questions 1
python src/benchmark_runner.py --provider kimi --model kimi-k2.5 --retriever hybrid --reranker tfidf --rerank-pool 12
```

3. Check summary in terminal and detailed logs in:

`logs/eval_runs.jsonl`

Failure/weak cases are also exported to:

`logs/failed_cases.jsonl`

Benchmark summary also includes traceability metrics:

- `citation_presence_avg`
- `citation_coverage_avg`

This gives you a repeatable score snapshot for each change to chunking, retrieval, or prompting.

## Hybrid alpha tuning

Find the best `hybrid` retriever weight automatically:

```bash
python src/tune_hybrid_alpha.py --provider kimi --model kimi-k2.5 --alphas 0.2,0.5,0.8
python src/tune_hybrid_alpha.py --provider kimi --model kimi-k2.5 --alphas 0.2,0.5,0.8 --max-questions 1
```

Result logs are stored in:

`logs/hybrid_alpha_sweep.jsonl`

## Compare Two Runs

Use the run comparator to check improvements/regressions between benchmark runs:

```bash
python src/compare_runs.py --list
python src/compare_runs.py --base-index -2 --target-index -1
python src/compare_runs.py --base-index -2 --target-index -1 --output-file logs/compare_last.json
```

## Offline Index Cache

Index artifacts are cached automatically to speed up repeated runs on unchanged corpora.

Environment controls:

- `RAG_USE_INDEX_CACHE=1` (default) enables cache reuse.
- `RAG_USE_INDEX_CACHE=0` disables cache and forces rebuild.
- `RAG_INDEX_CACHE_DIR=.cache/index` sets cache location.
