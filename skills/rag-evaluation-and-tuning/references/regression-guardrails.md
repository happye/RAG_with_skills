# Regression Guardrails

## Hard Gates

- No deployment if correctness decreases on canary set.
- No deployment if faithfulness decreases beyond threshold.
- No deployment if latency or cost exceeds budget.

## Soft Gates

- Require review for major prompt rewrites.
- Require rollback path for index schema changes.
