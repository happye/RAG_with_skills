# RAG Workbench Milestones

This roadmap defines a clear version-by-version plan to evolve the project toward a production-grade RAG system.

Guiding rule:

- Every version must deliver at least one concrete new feature or capability upgrade.

## Vision

Target state ("complete" RAG) includes:

- Personal high configurability with low-friction defaults
- Enterprise-grade retrieval capability and governance
- High-quality, grounded, and traceable answers
- High-quality retrieval (hybrid + semantic + rerank)
- Persistent index and vector database backend
- Repeatable evaluation and regression guardrails
- Cost/latency controls and observability
- Service/API deployment readiness

## Product Principles (Non-Negotiable)

1. Simple-first usage
- Default operation should be possible with one config file and one run command.
- Advanced settings must remain optional.

2. Layered configurability
- Three control layers must coexist:
  - mode defaults (`fast` / `balanced` / `quality`)
  - config overrides (`config.yaml`)
  - one-off CLI overrides

3. Enterprise capability without user burden
- Enterprise features (vector DB, ACL, observability, audit) should be pluggable and disabled by default.
- Individual users should not be forced into enterprise complexity.

4. Quality over appearance
- Every release must improve or maintain grounded answer quality and traceability.
- If quality regresses, release is blocked.

## Capability Tracks

The roadmap progresses in parallel across these tracks:

1. UX/Simplicity Track
- Simple config, profile modes, minimal command surface.

2. Retrieval/Enterprise Track
- Vector DB, hybrid retrieval, reranking, governance controls.

3. Quality/Trust Track
- Faithfulness checks, citation coverage, failure diagnostics, regression gates.

## Version Milestones

## v0.4 - Experiment Automation and Offline Index Foundation

Primary objective:

- Make experimentation faster and less manual.

Must-have features:

1. Run comparison tool (`compare_runs.py`)
- Compare two benchmark runs from `logs/eval_runs.jsonl`
- Output metric deltas and regression warnings

2. Offline index artifact (initial)
- Save/load chunk metadata and retrieval-ready artifacts to disk
- Avoid full rebuild on every run for unchanged corpus

3. Simple-run shell (initial)
- Add a minimal entrypoint (`run.py`) that supports mode-based execution.
- Preserve advanced controls via optional config/CLI overrides.

Success criteria:

- One command can produce a clear run-vs-run report
- First reusable index artifact created and consumed
- One-command run works for normal users without touching advanced parameters

## v0.5 - Vector Database Integration (Core Upgrade)

Primary objective:

- Move from in-memory embedding retrieval to persistent vector search backend.

Must-have features:

1. Vector DB backend (start with local Qdrant)
- Index embeddings into vector DB
- Query by vector similarity with top-k output

2. Backend switch support
- Config to choose retriever backend:
  - in-memory embedding
  - vector DB

3. Config UX guardrail
- Add human-readable config schema and examples for both personal and enterprise modes.

Success criteria:

- Same query can run against vector DB retriever successfully
- Benchmark can compare in-memory vs vector DB mode

## v0.6 - Reranker v2 and Retrieval Quality Lift

Primary objective:

- Improve precision of retrieved context.

Must-have features:

1. Strong reranker option
- Add semantic reranker (cross-encoder or provider rerank API)

2. Retrieval pipeline extension
- Candidate recall -> rerank -> final context selection

Success criteria:

- Reduced retrieval miss / low-signal cases on expanded Liaozhai eval set
- Better citation coverage with same or lower top-k

## v0.7 - Evaluation Maturity

Primary objective:

- Move beyond keyword-only proxies.

Must-have features:

1. Faithfulness and support checks
- Add answer-to-evidence support scoring

2. Expanded test sets
- Chinese long-form eval set expansion (20-50+ prompts)
- Add category tags by question type

3. Quality release gate
- Define minimum thresholds for answer grounding and citation metrics before release.

Success criteria:

- Stable evaluation protocol for release decisions
- Regression gate can block quality drops

## v0.8 - Cost and Latency Control

Primary objective:

- Improve efficiency under practical constraints.

Must-have features:

1. Context compression module
- Dedup, truncation strategy, sentence-level selection

2. Runtime profiles
- `fast`, `balanced`, `quality` profiles with defined defaults

Success criteria:

- Lower timeout/empty generation rate
- Reduced average prompt size while preserving answer quality

## v0.9 - Service Layer and Ops Readiness

Primary objective:

- Make the system consumable by external clients.

Must-have features:

1. API service (FastAPI)
- Query endpoint
- Health endpoint

2. Basic observability
- Structured logs
- Request/latency/error counters

Success criteria:

- End-to-end RAG available via API
- Basic operational telemetry available

## v1.0 - Stable Baseline Release

Primary objective:

- Deliver a stable, documented, reproducible RAG baseline.

Must-have features:

1. Release checklist completion
- Docs complete
- Repro commands validated
- Upgrade notes written

2. Quality gate
- No critical regressions on benchmark set
- Traceability metrics meet agreed thresholds

Success criteria:

- Tag and release as `v1.0.0`
- Team can reproduce setup and core results from scratch

## Working Rules per Version

For each version cycle:

1. Define one clear feature anchor
- Example: vector DB, reranker v2, eval upgrade

2. Run baseline before and after
- Store both runs in logs
- Compare with `compare_runs` (v0.4+)

3. Update docs and summary
- Add/refresh `docs/V0_X_SUMMARY.md`
- Record known issues and mitigations

4. Create backup and git tag
- Local backup zip
- Git tag per milestone

## Suggested Current Priority

Start now with:

1. v0.4 feature 1: `compare_runs.py`
2. v0.4 feature 2: offline index artifact

These two unlock faster iteration for all later versions.
