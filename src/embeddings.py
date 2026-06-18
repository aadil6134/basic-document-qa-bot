from __future__ import annotations

from typing import Sequence

from google import genai
from google.genai import types

from src.config import require_api_key, settings


class GeminiEmbeddingFunction:
    """Chroma-compatible embedding function backed by Gemini embeddings."""

    def __init__(self, task_type: str = "RETRIEVAL_DOCUMENT") -> None:
        self.task_type = task_type
        self.client = genai.Client(api_key=require_api_key())

    def name(self) -> str:
        return "gemini"

    def __call__(self, input: Sequence[str]) -> list[list[float]]:
        return self._embed(input, task_type=self.task_type)

    def embed_documents(self, input: Sequence[str]) -> list[list[float]]:
        return self._embed(input, task_type="RETRIEVAL_DOCUMENT")

    def embed_query(self, input: Sequence[str]) -> list[list[float]]:
        return self._embed(input, task_type="RETRIEVAL_QUERY")

    def _embed(self, input: Sequence[str], task_type: str) -> list[list[float]]:
        embeddings: list[list[float]] = []

        for text in input:
            response = self.client.models.embed_content(
                model=settings.embedding_model,
                contents=text,
                config=types.EmbedContentConfig(task_type=task_type),
            )

            if not response.embeddings or not response.embeddings[0].values:
                raise RuntimeError("Gemini did not return an embedding vector.")

            embeddings.append(list(response.embeddings[0].values))

        return embeddings
