# Prompting Patterns

## Required Prompt Constraints

- Use only provided context.
- Cite source identifiers for factual claims.
- If evidence is insufficient, say so explicitly.
- Keep answer concise and task-directed.

## Minimal Template

System:
You are a grounded assistant. Answer only with evidence from context.

Developer:
Return:
1. Direct answer.
2. Supporting citations.
3. Confidence note.

User:
Question: {question}
Context: {retrieved_chunks}

## Failure Signals

- No citations attached to claims.
- Strong confidence with weak evidence.
- Repeating context without answering the question.
