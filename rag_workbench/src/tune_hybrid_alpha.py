import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from benchmark_runner import append_eval_log, keyword_hit_rate, load_eval_set
from rag_baseline import build_context, build_index, generate_answer, retrieve


def parse_alphas(text: str) -> List[float]:
    values = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        value = float(token)
        if value < 0 or value > 1:
            raise ValueError(f"Alpha must be in [0, 1], got {value}")
        values.append(value)
    if not values:
        raise ValueError("No valid alpha values provided")
    return values


def evaluate_once(
    index,
    eval_rows: List[Dict],
    provider: str,
    model: str,
    top_k: int,
    alpha: float,
) -> Dict:
    os.environ["RAG_HYBRID_ALPHA"] = str(alpha)

    retrieval_scores = []
    answer_scores = []
    details = []

    for row in eval_rows:
        question = row["question"]
        keywords = row.get("expected_keywords", [])
        retrieved = retrieve(index, question, top_k=top_k, retriever="hybrid")
        context = build_context(retrieved)
        answer = generate_answer(
            provider=provider,
            question=question,
            context=context,
            model_override=model,
        )

        retrieval_hit = keyword_hit_rate(context, keywords)
        answer_hit = keyword_hit_rate(answer, keywords)
        retrieval_scores.append(retrieval_hit)
        answer_scores.append(answer_hit)

        details.append(
            {
                "id": row.get("id", ""),
                "retrieved_chunks": len(retrieved),
                "retrieval_keyword_hit": round(retrieval_hit, 4),
                "answer_keyword_hit": round(answer_hit, 4),
                "answer_preview": answer[:220],
            }
        )

    retrieval_avg = sum(retrieval_scores) / len(retrieval_scores)
    answer_avg = sum(answer_scores) / len(answer_scores)

    return {
        "alpha": alpha,
        "retrieval_keyword_hit_avg": round(retrieval_avg, 4),
        "answer_keyword_hit_avg": round(answer_avg, 4),
        "details": details,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune hybrid retriever alpha")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--eval-file", default="data/eval_set.jsonl")
    parser.add_argument("--provider", default="kimi")
    parser.add_argument("--model", default="")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--alphas", default="0.2,0.5,0.8")
    parser.add_argument(
        "--max-questions",
        type=int,
        default=0,
        help="Limit number of eval questions for fast smoke tests (0 means all)",
    )
    parser.add_argument("--log-file", default="logs/hybrid_alpha_sweep.jsonl")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    eval_file = Path(args.eval_file)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    if not eval_file.exists():
        raise FileNotFoundError(f"Eval file not found: {eval_file}")

    alphas = parse_alphas(args.alphas)
    rows = load_eval_set(eval_file)
    if not rows:
        raise ValueError("Eval set is empty")
    if args.max_questions > 0:
        rows = rows[: args.max_questions]

    index = build_index(data_dir, chunk_size=500, overlap=80)
    if not index:
        raise ValueError("No chunks indexed. Add .txt or .md files to data directory.")

    experiments = []
    for alpha in alphas:
        experiments.append(
            evaluate_once(
                index=index,
                eval_rows=rows,
                provider=args.provider,
                model=args.model,
                top_k=args.top_k,
                alpha=alpha,
            )
        )

    experiments.sort(key=lambda x: x["answer_keyword_hit_avg"], reverse=True)
    best = experiments[0]

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "provider": args.provider,
        "model": args.model,
        "top_k": args.top_k,
        "alphas": alphas,
        "best_alpha": best["alpha"],
        "best_answer_keyword_hit_avg": best["answer_keyword_hit_avg"],
        "experiments": experiments,
    }

    append_eval_log(Path(args.log_file), summary)

    print("=== Hybrid Alpha Sweep ===")
    print(f"Provider: {args.provider}")
    print(f"Model: {args.model or '(default)'}")
    print(f"Alphas tested: {alphas}")
    print(f"Best alpha: {best['alpha']}")
    print(f"Best answer keyword hit avg: {best['answer_keyword_hit_avg']}")
    print(f"Log file: {args.log_file}")


if __name__ == "__main__":
    main()
