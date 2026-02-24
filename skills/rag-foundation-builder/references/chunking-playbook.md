# Chunking Playbook

## Default Rule

- Prefer structure-aware splitting first: headings, sections, tables, lists.
- Use token windows only when structure is weak.

## Baseline Defaults

- Chunk size: 300-600 tokens.
- Overlap: 10%-15%.
- Keep title and section path in metadata.

## Escalation Rules

- If retrieval misses key facts, reduce chunk size.
- If context is fragmented, increase chunk size or overlap.
- If noise is high, add document-type specific chunkers.

## Anti-Patterns

- Using one chunk strategy for all document types.
- Dropping metadata needed for filtering or citations.
- Tuning chunking before collecting failure examples.
