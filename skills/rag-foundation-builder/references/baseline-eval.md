# Baseline Evaluation

## Retrieval Metrics

- Recall@k on evidence-bearing questions.
- MRR or nDCG for ranking quality.

## Generation Metrics

- Faithfulness to retrieved evidence.
- Answer correctness against expected answer.
- Citation precision.

## Minimal Scorecard

- Dataset size.
- Retrieval config.
- Prompt version.
- Metric values.
- Top 5 failure cases.

## Decision Rule

- Do not tune generation first if retrieval recall is below target.
- Prioritize the bottleneck with highest user-visible impact.
