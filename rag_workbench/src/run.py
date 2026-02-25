import argparse
import json
from pathlib import Path
from typing import Any, Dict

import yaml

from rag_baseline import (
    available_generation_provider,
    build_context,
    build_index,
    generate_answer,
    retrieve,
)

MODE_DEFAULTS = {
    "fast": {
        "retriever": "tfidf",
        "reranker": "none",
        "rerank_pool": 0,
        "top_k": 2,
        "chunk_size": 350,
        "overlap": 40,
        "provider": "auto",
        "model": "",
    },
    "balanced": {
        "retriever": "hybrid",
        "reranker": "tfidf",
        "rerank_pool": 12,
        "top_k": 3,
        "chunk_size": 500,
        "overlap": 80,
        "provider": "auto",
        "model": "",
    },
    "quality": {
        "retriever": "qdrant",
        "reranker": "tfidf",
        "rerank_pool": 20,
        "top_k": 4,
        "chunk_size": 600,
        "overlap": 120,
        "provider": "auto",
        "model": "",
    },
}


def load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def pick(cli_value, cfg: Dict[str, Any], key: str, default):
    if cli_value is not None:
        return cli_value
    value = cfg.get(key)
    if value in (None, ""):
        return default
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple run wrapper for RAG Workbench")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config")
    parser.add_argument("--mode", choices=["fast", "balanced", "quality"], default=None)
    parser.add_argument("--query", default=None)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--provider", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--retriever", choices=["keyword", "tfidf", "hybrid", "embedding", "qdrant"], default=None
    )
    parser.add_argument(
        "--reranker", choices=["none", "keyword", "tfidf"], default=None
    )
    parser.add_argument("--rerank-pool", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--overlap", type=int, default=None)
    parser.add_argument("--show-config", action="store_true")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    mode = pick(args.mode, cfg, "mode", "balanced")
    mode_defaults = MODE_DEFAULTS[mode]

    effective = {
        "mode": mode,
        "query": pick(args.query, cfg, "query", ""),
        "data_dir": pick(args.data_dir, cfg, "data_dir", "data"),
        "provider": pick(args.provider, cfg, "provider", mode_defaults["provider"]),
        "model": pick(args.model, cfg, "model", mode_defaults["model"]),
        "retriever": pick(args.retriever, cfg, "retriever", mode_defaults["retriever"]),
        "reranker": pick(args.reranker, cfg, "reranker", mode_defaults["reranker"]),
        "rerank_pool": int(
            pick(args.rerank_pool, cfg, "rerank_pool", mode_defaults["rerank_pool"])
        ),
        "top_k": int(pick(args.top_k, cfg, "top_k", mode_defaults["top_k"])),
        "chunk_size": int(
            pick(args.chunk_size, cfg, "chunk_size", mode_defaults["chunk_size"])
        ),
        "overlap": int(pick(args.overlap, cfg, "overlap", mode_defaults["overlap"])),
    }

    if not effective["query"].strip():
        raise ValueError("Query is required. Set it in config.yaml or pass --query.")

    if args.show_config:
        print("=== Effective Config ===")
        print(json.dumps(effective, ensure_ascii=False, indent=2))
        print()

    data_dir = Path(effective["data_dir"])
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    index = build_index(data_dir, chunk_size=effective["chunk_size"], overlap=effective["overlap"])
    if not index:
        raise ValueError("No chunks were indexed. Add .txt or .md files under data directory.")

    retrieved = retrieve(
        index,
        effective["query"],
        top_k=effective["top_k"],
        retriever=effective["retriever"],
        reranker=effective["reranker"],
        rerank_pool=effective["rerank_pool"],
    )
    context = build_context(retrieved)

    selected_provider = effective["provider"]
    if selected_provider == "auto":
        selected_provider = available_generation_provider()

    answer = generate_answer(
        provider=selected_provider,
        question=effective["query"],
        context=context,
        model_override=effective["model"],
    )

    print("=== Simple Run Summary ===")
    print(f"Mode: {effective['mode']}")
    print(f"Retriever: {effective['retriever']}")
    print(f"Reranker: {effective['reranker']}")
    print(f"Indexed chunks: {len(index)}")
    print(f"Retrieved chunks: {len(retrieved)}")
    print(f"Provider: {selected_provider}")
    print()
    print("=== Answer ===")
    print(answer)


if __name__ == "__main__":
    main()
