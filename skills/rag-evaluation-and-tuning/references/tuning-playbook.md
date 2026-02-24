# Tuning Playbook

## Retrieval First

- Tune chunking and embeddings before prompt polish.
- Sweep top-k and reranker settings with fixed prompt.

## Generation Second

- Tune prompt instructions and output constraints.
- Evaluate citation behavior and abstention quality.

## Last-Mile

- Add query rewriting, hybrid retrieval, or caching only after core quality stabilizes.
