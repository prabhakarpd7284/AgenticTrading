"""VoyageAI embedder (or any HTTP-based embedding API). Example implementation."""
from __future__ import annotations

import os

import httpx


class VoyageEmbedder:
    dim = 1024

    def __init__(self, api_key: str | None = None, model: str = "voyage-3"):
        self.api_key = api_key or os.environ.get("VOYAGE_API_KEY", "")
        self.model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not self.api_key:
            # Deterministic fallback for dev: hash-based pseudo-embedding
            import hashlib, struct
            out = []
            for t in texts:
                h = hashlib.blake2b(t.encode(), digest_size=self.dim * 4).digest()
                vec = list(struct.unpack(f"{self.dim}f", h))
                out.append(vec)
            return out
        r = httpx.post(
            "https://api.voyageai.com/v1/embeddings",
            json={"input": texts, "model": self.model},
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=30,
        )
        r.raise_for_status()
        return [d["embedding"] for d in r.json()["data"]]
