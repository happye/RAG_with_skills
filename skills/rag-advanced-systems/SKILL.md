---
name: rag-advanced-systems
description: Design and optimize advanced RAG architectures for scale, complexity, and long-term evolution. Use when extending mature RAG systems with hybrid retrieval, agentic retrieval planning, multi-index routing, knowledge graph augmentation, observability, and cost-latency governance.
---

# RAG Advanced Systems

Scale a mature RAG stack into a robust, adaptable system.

## Workflow

1. Select advanced architecture pattern.
2. Add retrieval diversity and routing.
3. Add control loops for quality and cost.
4. Harden observability and reliability.
5. Introduce long-term evolution strategy.

## Step 1: Select Advanced Architecture Pattern

- Match architecture to problem type: high-precision QA, exploratory search, multi-hop reasoning.
- Avoid adding complexity without measurable gain.

Read `references/architecture-patterns.md`.

## Step 2: Add Retrieval Diversity and Routing

- Combine semantic, lexical, and metadata-driven retrieval when needed.
- Route queries to specialized indexes by intent or domain.
- Add reranker ensembles only after baseline routing works.

Read `references/retrieval-routing.md`.

## Step 3: Add Control Loops for Quality and Cost

- Apply dynamic top-k and context compression.
- Use query difficulty estimation to select model tier.
- Add caching policies by query class.

Read `references/cost-latency-control.md`.

## Step 4: Harden Observability and Reliability

- Trace retrieval evidence and generation decisions per request.
- Monitor drift in query mix and document corpus.
- Add alert thresholds tied to user-impact metrics.

Read `references/observability.md`.

## Step 5: Introduce Long-Term Evolution Strategy

- Define deprecation path for outdated indexes and prompts.
- Use migration windows for schema and embedding upgrades.
- Keep backward compatibility for critical workflows.

Read `references/evolution-playbook.md`.
