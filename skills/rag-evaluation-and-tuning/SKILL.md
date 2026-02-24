---
name: rag-evaluation-and-tuning
description: Evaluate and optimize existing RAG systems from baseline to robust performance. Use when you need structured diagnosis, metric-driven tuning, retrieval and generation error decomposition, experiment prioritization, and regression-safe iteration.
---

# RAG Evaluation and Tuning

Diagnose and improve RAG quality with controlled experiments.

## Workflow

1. Define evaluation protocol.
2. Decompose failures by stage.
3. Prioritize experiments by impact.
4. Tune one variable at a time.
5. Guard against regressions.
6. Log learnings and promote stable configs.

## Step 1: Define Evaluation Protocol

- Freeze a benchmark set and split by difficulty.
- Separate online metrics and offline metrics.
- Set pass/fail thresholds before tuning.

Read `references/eval-protocol.md`.

## Step 2: Decompose Failures by Stage

- Label each failure as indexing, retrieval, reranking, prompt, or model limitation.
- Quantify distribution of failure types.
- Focus first on the dominant failure class.

Read `references/error-taxonomy.md`.

## Step 3: Prioritize Experiments by Impact

- Estimate expected gain, implementation effort, and risk.
- Pick high-gain and low-risk experiments first.
- Avoid running many interacting changes simultaneously.

Use `references/experiment-priority-matrix.md`.

## Step 4: Tune One Variable at a Time

- Keep strict experiment logs.
- Compare against the same benchmark snapshot.
- Stop when change is statistically or practically insignificant.

Use `references/tuning-playbook.md`.

## Step 5: Guard Against Regressions

- Maintain a fixed canary set of historically hard questions.
- Block rollout when canary performance degrades.
- Store previous stable configuration for rollback.

Read `references/regression-guardrails.md`.

## Step 6: Log Learnings and Promote Stable Configs

- Promote only reproducible improvements.
- Record both winning and failed experiments.
- Convert repeated failure patterns into permanent checks.

Use `references/iteration-log-format.md` for each cycle.
