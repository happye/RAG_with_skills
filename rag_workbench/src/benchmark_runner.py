import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from rag_baseline import build_context, build_index, generate_answer, retrieve


def load_eval_set(path: Path) -> List[Dict]:
    rows = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at line {i}: {exc}") from exc
        if "question" not in row:
            raise ValueError(f"Missing 'question' at line {i}")
        row.setdefault("id", f"line-{i}")
        row.setdefault("expected_keywords", [])
        rows.append(row)
    return rows


def keyword_hit_rate(text: str, keywords: List[str]) -> float:
    if not keywords:
        return 1.0
    text_l = text.lower()
    hits = sum(1 for k in keywords if k.lower() in text_l)
    return hits / len(keywords)


def ensure_log_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def append_eval_log(log_path: Path, payload: Dict) -> None:
    ensure_log_dir(log_path.parent)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAG benchmark over an eval set")
    parser.add_argument("--data-dir", default="data", help="Directory with corpus documents")
    parser.add_argument("--eval-file", default="data/eval_set.jsonl", help="JSONL eval file")
    parser.add_argument("--provider", default="kimi", help="Model provider")
    parser.add_argument("--model", default="", help="Optional model override")
    parser.add_argument(
        "--retriever",
        choices=["keyword", "tfidf", "hybrid"],
        default="keyword",
        help="Retrieval strategy",
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--overlap", type=int, default=80)
    parser.add_argument(
        "--max-questions",
        type=int,
        default=0,
        help="Limit number of eval questions for fast smoke tests (0 means all)",
    )
    parser.add_argument("--log-file", default="logs/eval_runs.jsonl")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    eval_file = Path(args.eval_file)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    if not eval_file.exists():
        raise FileNotFoundError(f"Eval file not found: {eval_file}")

    index = build_index(data_dir, chunk_size=args.chunk_size, overlap=args.overlap)
    if not index:
        raise ValueError("No chunks indexed. Add .txt or .md files to data directory.")

    rows = load_eval_set(eval_file)
    if not rows:
        raise ValueError("Eval set is empty.")
    if args.max_questions > 0:
        rows = rows[: args.max_questions]

    results = []
    retrieval_scores = []
    answer_scores = []

    for row in rows:
        qid = row["id"]
        question = row["question"]
        keywords = row.get("expected_keywords", [])

        retrieved = retrieve(index, question, top_k=args.top_k, retriever=args.retriever)
        context = build_context(retrieved)
        answer = generate_answer(
            provider=args.provider,
            question=question,
            context=context,
            model_override=args.model,
        )

        retrieval_hit = keyword_hit_rate(context, keywords)
        answer_hit = keyword_hit_rate(answer, keywords)
        retrieval_scores.append(retrieval_hit)
        answer_scores.append(answer_hit)

        results.append(
            {
                "id": qid,
                "question": question,
                "retrieved_chunks": len(retrieved),
                "retrieval_keyword_hit": round(retrieval_hit, 4),
                "answer_keyword_hit": round(answer_hit, 4),
                "answer_preview": answer[:280],
            }
        )

    retrieval_avg = sum(retrieval_scores) / len(retrieval_scores)
    answer_avg = sum(answer_scores) / len(answer_scores)

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "provider": args.provider,
        "model": args.model,
        "retriever": args.retriever,
        "top_k": args.top_k,
        "chunk_size": args.chunk_size,
        "overlap": args.overlap,
        "eval_count": len(rows),
        "retrieval_keyword_hit_avg": round(retrieval_avg, 4),
        "answer_keyword_hit_avg": round(answer_avg, 4),
        "results": results,
    }

    append_eval_log(Path(args.log_file), summary)

    print("=== Evaluation Summary ===")
    print(f"Provider: {args.provider}")
    print(f"Model: {args.model or '(default)'}")
    print(f"Retriever: {args.retriever}")
    print(f"Eval count: {len(rows)}")
    print(f"Retrieval keyword hit avg: {summary['retrieval_keyword_hit_avg']}")
    print(f"Answer keyword hit avg: {summary['answer_keyword_hit_avg']}")
    print(f"Log file: {args.log_file}")


if __name__ == "__main__":
    main()
