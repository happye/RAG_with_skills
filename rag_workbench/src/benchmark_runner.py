import argparse
import json
import re
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


def classify_case(retrieved_chunks: int, answer: str) -> str:
    text = (answer or "").lower()
    if retrieved_chunks == 0:
        return "retrieval_miss"
    if "returned empty content" in text or "timed out" in text:
        return "generation_empty_or_timeout"
    if "no relevant evidence found" in text:
        return "retrieval_low_signal"
    return "ok"


def extract_source_citations(answer: str) -> List[str]:
    if not answer:
        return []
    # Capture inline tags like [source=...; chunk=...; score=...]
    return re.findall(r"\[source=.*?\]", answer)


def citation_metrics(answer: str, retrieved_chunks: int) -> Dict[str, float]:
    cites = extract_source_citations(answer)
    has_citation = 1.0 if len(cites) > 0 else 0.0
    if retrieved_chunks <= 0:
        coverage = 0.0
    else:
        coverage = min(1.0, len(cites) / float(retrieved_chunks))
    return {
        "has_citation": has_citation,
        "citation_coverage": coverage,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAG benchmark over an eval set")
    parser.add_argument("--data-dir", default="data", help="Directory with corpus documents")
    parser.add_argument("--eval-file", default="data/eval_set.jsonl", help="JSONL eval file")
    parser.add_argument("--provider", default="kimi", help="Model provider")
    parser.add_argument("--model", default="", help="Optional model override")
    parser.add_argument(
        "--retriever",
        choices=["keyword", "tfidf", "hybrid", "embedding"],
        default="keyword",
        help="Retrieval strategy",
    )
    parser.add_argument(
        "--reranker",
        choices=["none", "keyword", "tfidf"],
        default="none",
        help="Optional second-stage reranker",
    )
    parser.add_argument(
        "--rerank-pool",
        type=int,
        default=0,
        help="Candidate pool size before reranking (0 uses dynamic default)",
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
    parser.add_argument(
        "--failed-file",
        default="logs/failed_cases.jsonl",
        help="JSONL file for failed/weak cases",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    eval_file = Path(args.eval_file)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    if not eval_file.exists():
        fallback_eval_file = Path("data_bak/eval_set.jsonl")
        if args.eval_file == "data/eval_set.jsonl" and fallback_eval_file.exists():
            print(
                f"[info] Eval file not found at {eval_file}, fallback to {fallback_eval_file}."
            )
            eval_file = fallback_eval_file
        else:
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
    citation_presence_scores = []
    citation_coverage_scores = []
    stage_counts = {
        "ok": 0,
        "retrieval_miss": 0,
        "retrieval_low_signal": 0,
        "generation_empty_or_timeout": 0,
        "ungrounded_answer": 0,
    }
    failed_rows = []

    for row in rows:
        qid = row["id"]
        question = row["question"]
        keywords = row.get("expected_keywords", [])

        retrieved = retrieve(
            index,
            question,
            top_k=args.top_k,
            retriever=args.retriever,
            reranker=args.reranker,
            rerank_pool=args.rerank_pool,
        )
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
        cite_stats = citation_metrics(answer, len(retrieved))
        citation_presence_scores.append(cite_stats["has_citation"])
        citation_coverage_scores.append(cite_stats["citation_coverage"])
        stage = classify_case(len(retrieved), answer)
        if stage == "ok" and len(retrieved) > 0 and cite_stats["has_citation"] == 0.0:
            stage = "ungrounded_answer"
        stage_counts[stage] = stage_counts.get(stage, 0) + 1

        results.append(
            {
                "id": qid,
                "question": question,
                "retrieved_chunks": len(retrieved),
                "retrieval_keyword_hit": round(retrieval_hit, 4),
                "answer_keyword_hit": round(answer_hit, 4),
                "has_citation": bool(cite_stats["has_citation"]),
                "citation_coverage": round(cite_stats["citation_coverage"], 4),
                "stage": stage,
                "answer_preview": answer[:280],
            }
        )

        if stage != "ok" or answer_hit < 0.5:
            failed_rows.append(
                {
                    "id": qid,
                    "question": question,
                    "retriever": args.retriever,
                    "reranker": args.reranker,
                    "provider": args.provider,
                    "model": args.model,
                    "retrieved_chunks": len(retrieved),
                    "retrieval_keyword_hit": round(retrieval_hit, 4),
                    "answer_keyword_hit": round(answer_hit, 4),
                    "has_citation": bool(cite_stats["has_citation"]),
                    "citation_coverage": round(cite_stats["citation_coverage"], 4),
                    "stage": stage,
                    "answer_preview": answer[:400],
                }
            )

    retrieval_avg = sum(retrieval_scores) / len(retrieval_scores)
    answer_avg = sum(answer_scores) / len(answer_scores)
    citation_presence_avg = sum(citation_presence_scores) / len(citation_presence_scores)
    citation_coverage_avg = sum(citation_coverage_scores) / len(citation_coverage_scores)

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "provider": args.provider,
        "model": args.model,
        "retriever": args.retriever,
        "reranker": args.reranker,
        "rerank_pool": args.rerank_pool,
        "top_k": args.top_k,
        "chunk_size": args.chunk_size,
        "overlap": args.overlap,
        "eval_count": len(rows),
        "retrieval_keyword_hit_avg": round(retrieval_avg, 4),
        "answer_keyword_hit_avg": round(answer_avg, 4),
        "citation_presence_avg": round(citation_presence_avg, 4),
        "citation_coverage_avg": round(citation_coverage_avg, 4),
        "stage_counts": stage_counts,
        "results": results,
    }

    append_eval_log(Path(args.log_file), summary)
    for row in failed_rows:
        append_eval_log(Path(args.failed_file), row)

    print("=== Evaluation Summary ===")
    print(f"Provider: {args.provider}")
    print(f"Model: {args.model or '(default)'}")
    print(f"Retriever: {args.retriever}")
    print(f"Reranker: {args.reranker}")
    print(f"Eval count: {len(rows)}")
    print(f"Retrieval keyword hit avg: {summary['retrieval_keyword_hit_avg']}")
    print(f"Answer keyword hit avg: {summary['answer_keyword_hit_avg']}")
    print(f"Citation presence avg: {summary['citation_presence_avg']}")
    print(f"Citation coverage avg: {summary['citation_coverage_avg']}")
    print(f"Stage counts: {stage_counts}")
    print(f"Log file: {args.log_file}")
    print(f"Failed cases file: {args.failed_file}")


if __name__ == "__main__":
    main()
