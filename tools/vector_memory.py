"""Persistent lightweight vector memory for cross-query retention."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List

TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")
DEFAULT_DIM = 256
DEFAULT_MEMORY_PATH = Path("data/vector_memory.json")


@dataclass
class MemoryRecord:
    id: str
    created_at: str
    query: str
    answer: str
    notes: List[str]
    sources: List[str]
    embedding: List[float]


@dataclass
class MemoryHit:
    score: float
    query: str
    answer: str
    notes: List[str]
    sources: List[str]
    created_at: str


def _tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text.lower())


def _hash_index(token: str, dim: int) -> int:
    digest = hashlib.blake2s(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % dim


def _hash_sign(token: str) -> float:
    digest = hashlib.blake2s(token.encode("utf-8"), digest_size=1).digest()
    return 1.0 if digest[0] % 2 == 0 else -1.0


def embed_text(text: str, dim: int = DEFAULT_DIM) -> List[float]:
    vector = [0.0] * dim
    tokens = _tokenize(text)
    for token in tokens:
        index = _hash_index(token, dim)
        vector[index] += _hash_sign(token)

    norm_sq = sum(value * value for value in vector)
    if norm_sq == 0.0:
        return vector
    norm = math.sqrt(norm_sq)
    return [value / norm for value in vector]


def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    if len(vec_a) != len(vec_b):
        return 0.0
    return sum(a * b for a, b in zip(vec_a, vec_b))


def _load_records(memory_path: Path) -> List[MemoryRecord]:
    if not memory_path.exists():
        return []
    raw = json.loads(memory_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        return []

    records: List[MemoryRecord] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            records.append(MemoryRecord(**item))
        except TypeError:
            continue
    return records


def _save_records(memory_path: Path, records: List[MemoryRecord]) -> None:
    memory_path.parent.mkdir(parents=True, exist_ok=True)
    memory_path.write_text(
        json.dumps([asdict(record) for record in records], indent=2),
        encoding="utf-8",
    )


def save_memory(
    query: str,
    answer: str,
    notes: List[str],
    sources: List[str],
    memory_path: Path = DEFAULT_MEMORY_PATH,
    dim: int = DEFAULT_DIM,
) -> None:
    records = _load_records(memory_path)
    memory_text = "\n".join([query, answer] + notes[:5] + sources[:5])
    record = MemoryRecord(
        id=hashlib.blake2s(f"{query}-{datetime.now(tz=timezone.utc).isoformat()}".encode("utf-8")).hexdigest(),
        created_at=datetime.now(tz=timezone.utc).isoformat(),
        query=query,
        answer=answer,
        notes=notes[:8],
        sources=sources[:8],
        embedding=embed_text(memory_text, dim=dim),
    )
    records.append(record)
    _save_records(memory_path, records)


def retrieve_memory(
    query: str,
    top_k: int = 3,
    min_score: float = 0.2,
    memory_path: Path = DEFAULT_MEMORY_PATH,
    dim: int = DEFAULT_DIM,
) -> List[MemoryHit]:
    records = _load_records(memory_path)
    if not records:
        return []

    query_vector = embed_text(query, dim=dim)
    scored = []
    for record in records:
        score = _cosine_similarity(query_vector, record.embedding)
        if score >= min_score:
            scored.append(
                MemoryHit(
                    score=score,
                    query=record.query,
                    answer=record.answer,
                    notes=record.notes,
                    sources=record.sources,
                    created_at=record.created_at,
                )
            )

    scored.sort(key=lambda hit: hit.score, reverse=True)
    return scored[:top_k]


def format_memory_context(memory_hits: List[MemoryHit]) -> str:
    if not memory_hits:
        return ""

    chunks: List[str] = []
    for index, hit in enumerate(memory_hits, start=1):
        notes = "\n".join(f"- {note}" for note in hit.notes[:3]) if hit.notes else "- No notes"
        sources = "\n".join(f"- {source}" for source in hit.sources[:3]) if hit.sources else "- No sources"
        chunks.append(
            (
                f"Memory {index} (score={hit.score:.3f}, at={hit.created_at})\n"
                f"Past query: {hit.query}\n"
                f"Past answer excerpt: {hit.answer[:400]}\n"
                f"Past notes:\n{notes}\n"
                f"Past sources:\n{sources}"
            )
        )

    return "\n\n".join(chunks)