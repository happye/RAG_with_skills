---
name: rag-foundation-builder
description: Build Retrieval-Augmented Generation systems from zero to first working baseline. Use when you need to design, implement, and validate a first end-to-end RAG pipeline, including chunking, embeddings, vector storage, retrieval, prompt assembly, and baseline evaluation.
---

# RAG Foundation Builder

Build a first reliable RAG system with a clear, testable baseline.

## Workflow

1. Define the baseline target.
2. Build the indexing pipeline.
3. Build the retrieval pipeline.
4. Build answer generation.
5. Run baseline evaluation.
6. Record decisions and next experiments.

## Step 1: Define Baseline Target

- Write a concrete task statement: user type, question types, and expected answer quality.
- Define constraints: latency target, cost envelope, and acceptable failure modes.
- Create a small golden set of questions before implementation.

Read `references/foundation-checklist.md` and execute items in order.

## Step 2: Build Indexing Pipeline

- Normalize and clean documents.
- Choose chunk strategy using document structure first, token windows second.
- Store chunk metadata that enables filtering and source traceability.
- Select embeddings with a practical baseline, not the most complex option.

Read `references/chunking-playbook.md` to choose chunking defaults and exceptions.

## Step 3: Build Retrieval Pipeline

- Start with semantic retrieval top-k baseline.
- Add metadata filters only when needed by task constraints.
- Add reranking after baseline is measured.
- Return source spans with identifiers for debugging.

Use a fixed retrieval config for first baseline so evaluation is comparable.

## Step 4: Build Answer Generation

- Assemble context with strict token budgeting.
- Force citation-friendly formatting in prompts.
- Prefer abstention behavior over hallucinated certainty.

Read `references/prompting-patterns.md` for robust response constraints.

## Step 5: Run Baseline Evaluation

- Evaluate retrieval quality and answer quality separately.
- Keep one metric per failure dimension at minimum.
- Save failed examples with root-cause notes.

Read `references/baseline-eval.md` for metric definitions and minimal scorecard.

## Step 6: Record Decisions and Next Experiments

- Write what was tried, what changed, and what improved or regressed.
- Convert each unresolved issue into one experiment hypothesis.
- Keep experiment scope small and reversible.

Use `references/iteration-log-format.md` as the required log format.
