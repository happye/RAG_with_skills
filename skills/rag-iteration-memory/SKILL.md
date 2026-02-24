---
name: rag-iteration-memory
description: Maintain disciplined RAG learning and optimization memory across long sessions. Use when running iterative RAG development, evaluating experiments, or making architecture decisions that must be recorded to prevent repeated mistakes and context loss.
---

# RAG Iteration Memory

Capture decisions, experiments, and lessons so the system improves continuously.

## Workflow

1. Record baseline before changes.
2. Log every experiment in a fixed format.
3. Extract reusable lessons.
4. Update active playbooks.
5. Plan the next iteration cycle.

## Step 1: Record Baseline Before Changes

- Save current config snapshot and key metrics.
- Record known risks and unresolved issues.

Use `references/session-baseline-template.md`.

## Step 2: Log Every Experiment

- Log one entry per experiment.
- Include hypothesis, change, result, and decision.
- Never merge multiple changes into one unlabeled entry.

Use `references/experiment-log-template.md`.

## Step 3: Extract Reusable Lessons

- Convert repeated failures into anti-pattern notes.
- Convert repeated wins into default heuristics.

Use `references/lessons-catalog.md`.

## Step 4: Update Active Playbooks

- Update chunking, retrieval, and prompt playbooks after validated changes.
- Keep old guidance only if still valid.

Use `references/playbook-update-checklist.md`.

## Step 5: Plan Next Iteration

- Select one primary bottleneck.
- Define 1-3 focused experiments.
- Set stop criteria before execution.

Use `references/next-iteration-planner.md`.
