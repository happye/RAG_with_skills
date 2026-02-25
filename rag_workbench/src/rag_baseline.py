import argparse
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import requests

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    # Keep runtime dependency optional; env vars can still be set in shell.
    pass


OPENAI_COMPAT_PROVIDERS = {
    "openai": {
        "api_key_env": "OPENAI_API_KEY",
        "base_url_env": "OPENAI_BASE_URL",
        "default_base_url": "https://api.openai.com/v1",
        "model_env": "OPENAI_MODEL",
        "default_model": "gpt-4o-mini",
    },
    "deepseek": {
        "api_key_env": "DEEPSEEK_API_KEY",
        "base_url_env": "DEEPSEEK_BASE_URL",
        "default_base_url": "https://api.deepseek.com/v1",
        "model_env": "DEEPSEEK_MODEL",
        "default_model": "deepseek-chat",
    },
    "qwen": {
        "api_key_env": "QWEN_API_KEY",
        "base_url_env": "QWEN_BASE_URL",
        "default_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model_env": "QWEN_MODEL",
        "default_model": "qwen-plus",
    },
    "kimi": {
        "api_key_env": "KIMI_API_KEY",
        "base_url_env": "KIMI_BASE_URL",
        "default_base_url": "https://api.moonshot.cn/v1",
        "model_env": "KIMI_MODEL",
        "default_model": "moonshot-v1-8k",
    },
    "glm": {
        "api_key_env": "GLM_API_KEY",
        "base_url_env": "GLM_BASE_URL",
        "default_base_url": "https://open.bigmodel.cn/api/paas/v4",
        "model_env": "GLM_MODEL",
        "default_model": "glm-4-plus",
    },
    "doubao": {
        "api_key_env": "DOUBAO_API_KEY",
        "base_url_env": "DOUBAO_BASE_URL",
        "default_base_url": "https://ark.cn-beijing.volces.com/api/v3",
        "model_env": "DOUBAO_MODEL",
        "default_model": "doubao-seed-1-6-250615",
    },
}

_EMBED_MODEL_CACHE: Dict[str, object] = {}
_EMBED_INDEX_CACHE: Dict[Tuple[int, str], object] = {}
_QDRANT_COLLECTION_CACHE: Dict[Tuple[int, str, str], bool] = {}


def build_grounded_prompt(question: str, context: str) -> str:
    return (
        "You are a grounded RAG assistant. Use only the provided context. "
        "If evidence is insufficient, say so clearly. "
        "Cite source tags in square brackets. "
        "Keep the answer concise (prefer <= 220 Chinese characters or <= 120 English words).\n\n"
        f"Question: {question}\n\n"
        f"Context:\n{context}\n"
    )


def get_env_clean(name: str, default: str = "") -> str:
    raw = os.getenv(name, default)
    if raw is None:
        return default
    value = raw.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    return value.strip()


def get_http_timeout() -> Tuple[float, float]:
    connect_timeout = float(get_env_clean("RAG_HTTP_CONNECT_TIMEOUT", "10") or "10")
    read_timeout = float(get_env_clean("RAG_HTTP_READ_TIMEOUT", "45") or "45")
    return connect_timeout, read_timeout


def normalize_text_content(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                if isinstance(item.get("text"), str):
                    parts.append(item["text"])
                    continue
                if item.get("type") == "text" and isinstance(item.get("text"), str):
                    parts.append(item["text"])
                    continue
        return "\n".join(p.strip() for p in parts if p and p.strip()).strip()
    return str(content).strip()


@dataclass
class Chunk:
    source: str
    chunk_id: int
    text: str


def tokenize(text: str) -> List[str]:
    text_l = text.lower()
    # Keep English-like tokens.
    en_tokens = re.findall(r"[a-zA-Z0-9_]+", text_l)
    # Keep CJK character-level tokens to improve Chinese matching.
    zh_tokens = re.findall(r"[\u4e00-\u9fff]", text)
    return en_tokens + zh_tokens


def contains_cjk(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text))


def read_text_auto(path: Path) -> str:
    raw = path.read_bytes()
    encodings = ["utf-8-sig", "utf-16", "utf-16-le", "utf-16-be", "gb18030"]
    best_text = ""
    best_score = -1

    for enc in encodings:
        try:
            text = raw.decode(enc)
        except Exception:
            continue
        # Prefer decodes that contain meaningful visible characters and less null noise.
        visible = sum(1 for ch in text if ch.isprintable())
        cjk = len(re.findall(r"[\u4e00-\u9fff]", text))
        nulls = text.count("\x00")
        score = visible + (cjk * 5) - (nulls * 10)
        if score > best_score:
            best_score = score
            best_text = text

    if best_text:
        return best_text
    return raw.decode("utf-8", errors="ignore")


def load_documents(data_dir: Path) -> List[Tuple[str, str]]:
    docs = []
    for path in sorted(data_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".txt", ".md"}:
            docs.append((str(path), read_text_auto(path)))
    return docs


def _indexable_files(data_dir: Path) -> List[Path]:
    return [
        path
        for path in sorted(data_dir.rglob("*"))
        if path.is_file() and path.suffix.lower() in {".txt", ".md"}
    ]


def _corpus_signature(data_dir: Path, chunk_size: int, overlap: int) -> str:
    items = []
    for path in _indexable_files(data_dir):
        stat = path.stat()
        items.append(
            {
                "rel": str(path.relative_to(data_dir)),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    payload = {
        "chunk_size": chunk_size,
        "overlap": overlap,
        "files": items,
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def _index_cache_path(data_dir: Path, chunk_size: int, overlap: int) -> Path:
    cache_root = Path(get_env_clean("RAG_INDEX_CACHE_DIR", ".cache/index"))
    signature = _corpus_signature(data_dir, chunk_size, overlap)
    return cache_root / f"index_{signature}_c{chunk_size}_o{overlap}.jsonl"


def _save_index_cache(cache_path: Path, index: List[Chunk]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as f:
        for chunk in index:
            f.write(
                json.dumps(
                    {
                        "source": chunk.source,
                        "chunk_id": chunk.chunk_id,
                        "text": chunk.text,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def _load_index_cache(cache_path: Path) -> List[Chunk]:
    rows = []
    for line in cache_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        rows.append(
            Chunk(source=obj["source"], chunk_id=int(obj["chunk_id"]), text=obj["text"])
        )
    return rows


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 80) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(text[start:end])
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks


def build_index(data_dir: Path, chunk_size: int, overlap: int) -> List[Chunk]:
    use_cache = get_env_clean("RAG_USE_INDEX_CACHE", "1") != "0"
    cache_path = _index_cache_path(data_dir, chunk_size, overlap)
    if use_cache and cache_path.exists():
        try:
            return _load_index_cache(cache_path)
        except Exception:
            # Rebuild if cache is corrupted or incompatible.
            pass

    index = []
    docs = load_documents(data_dir)
    for source, content in docs:
        pieces = chunk_text(content, chunk_size=chunk_size, overlap=overlap)
        for i, piece in enumerate(pieces):
            index.append(Chunk(source=source, chunk_id=i, text=piece))

    if use_cache:
        try:
            _save_index_cache(cache_path, index)
        except Exception:
            # Cache write failure should never break core retrieval flow.
            pass

    return index


def score_query_chunk(query_tokens: set, chunk_text_value: str) -> float:
    chunk_tokens = set(tokenize(chunk_text_value))
    if not chunk_tokens:
        return 0.0
    overlap = query_tokens.intersection(chunk_tokens)
    return len(overlap) / (len(query_tokens) + 1e-9)


def _top_scored_chunks(
    index: List[Chunk], scores: List[float], top_k: int
) -> List[Tuple[Chunk, float]]:
    paired = list(zip(index, scores))
    paired.sort(key=lambda x: x[1], reverse=True)
    return [item for item in paired[:top_k] if item[1] > 0]


def _tfidf_scores_for_chunks(chunks: List[Chunk], query: str) -> List[float]:
    from sklearn.feature_extraction.text import TfidfVectorizer

    corpus = [chunk.text for chunk in chunks]
    if not corpus:
        return []
    has_cjk = contains_cjk(query) or any(contains_cjk(chunk.text) for chunk in chunks[:20])
    if has_cjk:
        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1)
    else:
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    matrix = vectorizer.fit_transform(corpus + [query])
    doc_matrix = matrix[:-1]
    query_vec = matrix[-1]
    return (doc_matrix @ query_vec.T).toarray().ravel().tolist()


def retrieve_keyword(index: List[Chunk], query: str, top_k: int) -> List[Tuple[Chunk, float]]:
    q_tokens = set(tokenize(query))
    scores = [score_query_chunk(q_tokens, chunk.text) for chunk in index]
    return _top_scored_chunks(index, scores, top_k)


def retrieve_tfidf(index: List[Chunk], query: str, top_k: int) -> List[Tuple[Chunk, float]]:
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
    except Exception as exc:
        raise RuntimeError(
            "TF-IDF retriever requires scikit-learn. Install requirements first."
        ) from exc

    if not index:
        return []
    similarities = _tfidf_scores_for_chunks(index, query)
    return _top_scored_chunks(index, similarities, top_k)


def retrieve_hybrid(index: List[Chunk], query: str, top_k: int) -> List[Tuple[Chunk, float]]:
    keyword = retrieve_keyword(index, query, top_k=len(index))
    tfidf = retrieve_tfidf(index, query, top_k=len(index))

    k_map = {id(chunk): score for chunk, score in keyword}
    t_map = {id(chunk): score for chunk, score in tfidf}

    # Balanced hybrid score with simple weighted sum.
    alpha = float(get_env_clean("RAG_HYBRID_ALPHA", "0.5") or "0.5")
    alpha = min(max(alpha, 0.0), 1.0)
    scores = []
    for chunk in index:
        score = alpha * k_map.get(id(chunk), 0.0) + (1 - alpha) * t_map.get(id(chunk), 0.0)
        scores.append(score)
    return _top_scored_chunks(index, scores, top_k)


def get_embedding_model_name() -> str:
    return get_env_clean(
        "RAG_EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )


def _get_sentence_transformer(model_name: str):
    if model_name in _EMBED_MODEL_CACHE:
        return _EMBED_MODEL_CACHE[model_name]
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as exc:
        raise RuntimeError(
            "Embedding retriever requires sentence-transformers. "
            "Install with: python -m pip install sentence-transformers"
        ) from exc
    model = SentenceTransformer(model_name)
    _EMBED_MODEL_CACHE[model_name] = model
    return model


def _l2_normalize(matrix):
    import numpy as np

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def _get_embedding_index(index: List[Chunk], model_name: str):
    cache_key = (id(index), model_name)
    if cache_key in _EMBED_INDEX_CACHE:
        return _EMBED_INDEX_CACHE[cache_key]

    model = _get_sentence_transformer(model_name)
    corpus = [chunk.text for chunk in index]
    batch_size = int(get_env_clean("RAG_EMBEDDING_BATCH_SIZE", "32") or "32")
    embeddings = model.encode(corpus, batch_size=batch_size, show_progress_bar=False)
    embeddings = _l2_normalize(embeddings)
    _EMBED_INDEX_CACHE[cache_key] = embeddings
    return embeddings


def retrieve_embedding(index: List[Chunk], query: str, top_k: int) -> List[Tuple[Chunk, float]]:
    import numpy as np

    if not index:
        return []
    model_name = get_embedding_model_name()
    model = _get_sentence_transformer(model_name)
    doc_embeddings = _get_embedding_index(index, model_name)
    query_embedding = model.encode([query], show_progress_bar=False)
    query_embedding = _l2_normalize(query_embedding)[0]
    scores = np.dot(doc_embeddings, query_embedding).tolist()
    return _top_scored_chunks(index, scores, top_k)


def _default_qdrant_collection(index: List[Chunk], model_name: str) -> str:
    hasher = hashlib.sha1()
    hasher.update(model_name.encode("utf-8"))
    sample_count = min(len(index), 50)
    for chunk in index[:sample_count]:
        hasher.update(chunk.source.encode("utf-8", errors="ignore"))
        hasher.update(str(chunk.chunk_id).encode("utf-8"))
        hasher.update(chunk.text[:200].encode("utf-8", errors="ignore"))
    return f"rag_{hasher.hexdigest()[:12]}"


def _ensure_qdrant_collection(index: List[Chunk], model_name: str):
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, PointStruct, VectorParams
    except Exception as exc:
        raise RuntimeError(
            "Qdrant retriever requires qdrant-client. "
            "Install with: python -m pip install qdrant-client"
        ) from exc

    if not index:
        return None, ""

    qdrant_path = get_env_clean("RAG_QDRANT_PATH", ".cache/qdrant")
    model = _get_sentence_transformer(model_name)
    embeddings = _get_embedding_index(index, model_name)
    vector_size = int(embeddings.shape[1])

    collection_name = get_env_clean("RAG_QDRANT_COLLECTION", "")
    if not collection_name:
        collection_name = _default_qdrant_collection(index, model_name)

    cache_key = (id(index), model_name, collection_name)
    client = QdrantClient(path=qdrant_path)
    if cache_key in _QDRANT_COLLECTION_CACHE:
        return client, collection_name

    recreate = get_env_clean("RAG_QDRANT_RECREATE", "0") == "1"
    try:
        exists = client.collection_exists(collection_name=collection_name)
    except Exception:
        exists = False

    if recreate and exists:
        client.delete_collection(collection_name=collection_name)
        exists = False

    if not exists:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        points = [
            PointStruct(id=i, vector=embeddings[i].tolist(), payload={"chunk_idx": i})
            for i in range(len(index))
        ]
        client.upsert(collection_name=collection_name, points=points)
    _QDRANT_COLLECTION_CACHE[cache_key] = True
    return client, collection_name


def retrieve_qdrant(index: List[Chunk], query: str, top_k: int) -> List[Tuple[Chunk, float]]:
    if not index:
        return []

    model_name = get_embedding_model_name()
    model = _get_sentence_transformer(model_name)
    client, collection_name = _ensure_qdrant_collection(index, model_name)
    query_embedding = model.encode([query], show_progress_bar=False)
    query_embedding = _l2_normalize(query_embedding)[0].tolist()

    try:
        if hasattr(client, "query_points"):
            query_result = client.query_points(
                collection_name=collection_name,
                query=query_embedding,
                limit=max(1, top_k),
                with_payload=True,
            )
            hits = getattr(query_result, "points", query_result)
        else:
            hits = client.search(
                collection_name=collection_name,
                query_vector=query_embedding,
                limit=max(1, top_k),
                with_payload=True,
            )
    except Exception as exc:
        raise RuntimeError(f"Qdrant search failed: {exc}") from exc

    results: List[Tuple[Chunk, float]] = []
    for hit in hits:
        idx = None
        payload = getattr(hit, "payload", None) or {}
        if isinstance(payload, dict) and "chunk_idx" in payload:
            idx = int(payload["chunk_idx"])
        elif getattr(hit, "id", None) is not None:
            idx = int(hit.id)
        if idx is None or idx < 0 or idx >= len(index):
            continue
        score = float(getattr(hit, "score", 0.0) or 0.0)
        if score <= 0:
            continue
        results.append((index[idx], score))
    return results[:top_k]


def rerank(
    candidates: List[Tuple[Chunk, float]], query: str, top_k: int, reranker: str = "none"
) -> List[Tuple[Chunk, float]]:
    if reranker == "none":
        return candidates[:top_k]
    chunks = [chunk for chunk, _ in candidates]
    if not chunks:
        return []
    if reranker == "keyword":
        q_tokens = set(tokenize(query))
        scores = [score_query_chunk(q_tokens, chunk.text) for chunk in chunks]
        return _top_scored_chunks(chunks, scores, top_k)
    if reranker == "tfidf":
        scores = _tfidf_scores_for_chunks(chunks, query)
        return _top_scored_chunks(chunks, scores, top_k)
    raise ValueError(f"Unknown reranker: {reranker}")


def retrieve(
    index: List[Chunk],
    query: str,
    top_k: int,
    retriever: str = "keyword",
    reranker: str = "none",
    rerank_pool: int = 0,
) -> List[Tuple[Chunk, float]]:
    effective_pool = rerank_pool if rerank_pool > 0 else max(top_k, top_k * 3)
    if retriever == "keyword":
        initial = retrieve_keyword(index, query, effective_pool)
        return rerank(initial, query, top_k, reranker=reranker)
    if retriever == "tfidf":
        initial = retrieve_tfidf(index, query, effective_pool)
        return rerank(initial, query, top_k, reranker=reranker)
    if retriever == "hybrid":
        initial = retrieve_hybrid(index, query, effective_pool)
        return rerank(initial, query, top_k, reranker=reranker)
    if retriever == "embedding":
        initial = retrieve_embedding(index, query, effective_pool)
        return rerank(initial, query, top_k, reranker=reranker)
    if retriever == "qdrant":
        initial = retrieve_qdrant(index, query, effective_pool)
        return rerank(initial, query, top_k, reranker=reranker)
    raise ValueError(f"Unknown retriever: {retriever}")


def build_context(retrieved: List[Tuple[Chunk, float]]) -> str:
    blocks = []
    for chunk, score in retrieved:
        blocks.append(
            f"[source={chunk.source}; chunk={chunk.chunk_id}; score={score:.3f}]\n{chunk.text}"
        )
    return "\n\n".join(blocks)


def generate_with_anthropic(question: str, context: str, model_override: str = "") -> str:
    try:
        from anthropic import Anthropic
    except Exception:
        return "Anthropic SDK is not installed. Install requirements first."

    api_key = get_env_clean("ANTHROPIC_API_KEY")
    if not api_key:
        return "ANTHROPIC_API_KEY is not set."

    client = Anthropic(api_key=api_key)
    prompt = build_grounded_prompt(question, context)
    model = model_override or get_env_clean("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest")

    response = client.messages.create(
        model=model,
        max_tokens=500,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )

    parts = []
    for block in response.content:
        if hasattr(block, "text"):
            text = normalize_text_content(block.text)
            if text:
                parts.append(text)
    final_text = "\n".join(parts).strip()
    if final_text:
        return final_text
    return "Anthropic returned an empty text response."


def generate_openai_compatible(
    provider: str, question: str, context: str, model_override: str = ""
) -> str:
    cfg = OPENAI_COMPAT_PROVIDERS[provider]
    api_key = get_env_clean(cfg["api_key_env"], "")
    if not api_key:
        return f"{cfg['api_key_env']} is not set."

    base_url = get_env_clean(cfg["base_url_env"], cfg["default_base_url"]).rstrip("/")
    model = model_override or get_env_clean(cfg["model_env"], cfg["default_model"])
    prompt = build_grounded_prompt(question, context)
    max_retries = int(get_env_clean("RAG_REQUEST_MAX_RETRIES", "2") or "2")

    url = f"{base_url}/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }

    # Provider/model-specific sampling constraints.
    if provider == "kimi":
        # Moonshot docs: kimi-k2.5 has fixed sampling behavior.
        if model.startswith("kimi-k2.5"):
            payload["temperature"] = 1
        else:
            payload["temperature"] = float(get_env_clean("KIMI_TEMPERATURE", "0.6") or "0.6")
        payload["max_completion_tokens"] = int(
            get_env_clean("KIMI_MAX_COMPLETION_TOKENS", "1200") or "1200"
        )
    else:
        temp_env = f"{provider.upper()}_TEMPERATURE"
        payload["temperature"] = float(get_env_clean(temp_env, "0") or "0")
        payload["max_tokens"] = int(get_env_clean("RAG_MAX_TOKENS", "500") or "500")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    http_timeout = get_http_timeout()

    length_retry_done = False
    for attempt in range(max_retries + 1):
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=http_timeout)
            if response.status_code == 429 and attempt < max_retries:
                retry_after = response.headers.get("Retry-After", "")
                wait_seconds = float(retry_after) if retry_after else float(2 ** attempt)
                time.sleep(wait_seconds)
                continue

            response.raise_for_status()
            data = response.json()
            choices = data.get("choices", [])
            if not choices:
                return f"{provider} returned no choices."
            message = choices[0].get("message", {})
            content = normalize_text_content(message.get("content"))
            if content:
                return content
            finish_reason = choices[0].get("finish_reason", "")

            if (
                provider == "kimi"
                and finish_reason == "length"
                and not length_retry_done
                and "max_completion_tokens" in payload
            ):
                payload["max_completion_tokens"] = min(
                    int(payload["max_completion_tokens"]) * 2,
                    int(get_env_clean("KIMI_MAX_COMPLETION_TOKENS_CAP", "4000") or "4000"),
                )
                # Recovery path for providers that return empty content with finish_reason=length.
                payload["messages"] = [
                    {
                        "role": "user",
                        "content": prompt
                        + "\n\nIf the answer is long, return a compact summary with at most 5 bullets.",
                    }
                ]
                length_retry_done = True
                continue

            return (
                f"{provider} returned empty content"
                + (f" (finish_reason={finish_reason})" if finish_reason else "")
                + "."
            )
        except requests.HTTPError:
            body_text = ""
            err_type = ""
            err_msg = ""
            try:
                err = response.json().get("error", {})
                err_type = err.get("type", "")
                err_msg = err.get("message", "")
                body_text = f"type={err_type}; message={err_msg}"
            except Exception:
                body_text = response.text[:500]
            if response.status_code == 429:
                return (
                    f"{provider} rate limited (429). {body_text}. "
                    "Check provider quota/concurrency/RPM limits and retry later."
                )
            return f"{provider} HTTP {response.status_code}: {body_text}"
        except requests.Timeout:
            if attempt < max_retries:
                time.sleep(float(2 ** attempt))
                continue
            return (
                f"{provider} request timed out after retries "
                f"(connect={http_timeout[0]}s, read={http_timeout[1]}s)."
            )
        except Exception as exc:
            if attempt < max_retries:
                time.sleep(float(2 ** attempt))
                continue
            return f"{provider} request failed: {exc}"

    return f"{provider} request failed after retries."


def generate_with_gemini(question: str, context: str, model_override: str = "") -> str:
    api_key = get_env_clean("GEMINI_API_KEY", "")
    if not api_key:
        return "GEMINI_API_KEY is not set."

    model = model_override or get_env_clean("GEMINI_MODEL", "gemini-2.0-flash")
    prompt = build_grounded_prompt(question, context)
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        f"?key={api_key}"
    )
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0, "maxOutputTokens": 500},
    }
    http_timeout = get_http_timeout()

    try:
        response = requests.post(url, json=payload, timeout=http_timeout)
        response.raise_for_status()
        data = response.json()
        candidates = data.get("candidates", [])
        if not candidates:
            return "gemini returned no candidates."
        parts = candidates[0].get("content", {}).get("parts", [])
        texts = [normalize_text_content(p.get("text", "")) for p in parts if p.get("text")]
        final_text = "\n".join(t for t in texts if t).strip()
        if final_text:
            return final_text
        finish_reason = candidates[0].get("finishReason", "")
        return (
            "gemini returned empty content"
            + (f" (finishReason={finish_reason})" if finish_reason else "")
            + "."
        )
    except requests.Timeout:
        return (
            "gemini request timed out "
            f"(connect={http_timeout[0]}s, read={http_timeout[1]}s)."
        )
    except Exception as exc:
        return f"gemini request failed: {exc}"


def available_generation_provider() -> str:
    if get_env_clean("ANTHROPIC_API_KEY"):
        return "anthropic"
    if get_env_clean("OPENAI_API_KEY"):
        return "openai"
    if get_env_clean("GEMINI_API_KEY"):
        return "gemini"
    for provider, cfg in OPENAI_COMPAT_PROVIDERS.items():
        if provider == "openai":
            continue
        if get_env_clean(cfg["api_key_env"]):
            return provider
    return "retrieval-only"


def generate_answer(provider: str, question: str, context: str, model_override: str = "") -> str:
    generated = ""
    if provider == "retrieval-only":
        return retrieval_only_answer(question, context)
    if provider == "anthropic":
        generated = generate_with_anthropic(question, context, model_override=model_override)
    elif provider == "gemini":
        generated = generate_with_gemini(question, context, model_override=model_override)
    elif provider in OPENAI_COMPAT_PROVIDERS:
        generated = generate_openai_compatible(
            provider, question, context, model_override=model_override
        )
    else:
        return f"Unknown provider: {provider}"

    if generated and generated.strip() and "returned empty content" not in generated:
        return generated

    fallback = retrieval_only_answer(question, context)
    return (
        "Provider returned empty output. Falling back to retrieval evidence.\n\n"
        f"{fallback}"
    )


def retrieval_only_answer(question: str, context: str) -> str:
    if not context.strip():
        return "No relevant evidence found in the indexed documents."
    return (
        "Retrieval-only mode: evidence found below. "
        "Set one provider API key to enable generated answers.\n\n"
        f"Question: {question}\n\n"
        f"Evidence:\n{context}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal RAG baseline")
    parser.add_argument("--data-dir", default="data", help="Directory with .txt/.md files")
    parser.add_argument("--query", required=True, help="User question")
    parser.add_argument("--top-k", type=int, default=3, help="Number of chunks to retrieve")
    parser.add_argument(
        "--retriever",
        choices=["keyword", "tfidf", "hybrid", "embedding", "qdrant"],
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
    parser.add_argument("--chunk-size", type=int, default=500, help="Chunk size in characters")
    parser.add_argument("--overlap", type=int, default=80, help="Chunk overlap in characters")
    parser.add_argument(
        "--provider",
        choices=[
            "auto",
            "retrieval-only",
            "anthropic",
            "openai",
            "gemini",
            "deepseek",
            "qwen",
            "kimi",
            "glm",
            "doubao",
        ],
        default="auto",
        help="Model provider",
    )
    parser.add_argument("--model", default="", help="Optional model override for selected provider")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    index = build_index(data_dir, chunk_size=args.chunk_size, overlap=args.overlap)
    if not index:
        raise ValueError("No chunks were indexed. Add .txt or .md files under data directory.")

    retrieved = retrieve(
        index,
        args.query,
        top_k=args.top_k,
        retriever=args.retriever,
        reranker=args.reranker,
        rerank_pool=args.rerank_pool,
    )
    context = build_context(retrieved)

    print("=== Retrieval Summary ===")
    print(f"Retriever: {args.retriever}")
    print(f"Reranker: {args.reranker}")
    print(f"Indexed chunks: {len(index)}")
    print(f"Retrieved chunks: {len(retrieved)}")
    print()

    selected_provider = args.provider
    if selected_provider == "auto":
        selected_provider = available_generation_provider()

    print(f"Selected provider: {selected_provider}")
    print()

    answer = generate_answer(
        provider=selected_provider,
        question=args.query,
        context=context,
        model_override=args.model,
    )

    print("=== Answer ===")
    print(answer)


if __name__ == "__main__":
    main()
