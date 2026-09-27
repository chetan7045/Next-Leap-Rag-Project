"""Local embedding service.

Loads ``sentence-transformers/all-MiniLM-L6-v2`` once per process and reuses it.
No external embedding API is ever called, and no model is instantiated per request.

The encoder runs on ONNX Runtime via ``fastembed`` rather than PyTorch. The weights,
tokenizer, pooling and L2 normalisation are the same ones ``sentence-transformers``
would load, so the vector space is unchanged and an existing Chroma index stays
valid. The reason for the swap is memory: importing PyTorch costs roughly 750 MB RSS
before any weights are read, which exceeds the 512 MB available on a Render Free (or
Starter) instance. The ONNX path peaks near 265 MB.
"""

from __future__ import annotations

import math
import threading
from typing import TYPE_CHECKING

from app.core.config import Settings
from app.core.errors import ConfigurationError
from app.core.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover
    from fastembed import TextEmbedding

logger = get_logger("app.services.embeddings")

# Expected output width for the configured default model.
_EXPECTED_DIMENSIONS = {"sentence-transformers/all-MiniLM-L6-v2": 384}


class EmbeddingService:
    """Thread-safe singleton wrapper around a local ONNX sentence-embedding model."""

    _instance: "EmbeddingService | None" = None
    _instance_lock = threading.Lock()

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model: TextEmbedding | None = None
        self._lock = threading.Lock()
        self._dimensions: int | None = None

    # --- Singleton -------------------------------------------------------
    @classmethod
    def get_instance(cls, settings: Settings) -> "EmbeddingService":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls(settings)
            return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Drop the cached model (used by tests)."""
        with cls._instance_lock:
            cls._instance = None

    # --- Model lifecycle -------------------------------------------------
    @property
    def model_name(self) -> str:
        return self._settings.embedding_model

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def dimensions(self) -> int | None:
        return self._dimensions

    def load(self) -> None:
        """Load the model once. Subsequent calls are no-ops."""
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            try:
                from fastembed import TextEmbedding
            except ImportError as exc:  # pragma: no cover - dependency guard
                raise ConfigurationError(
                    "fastembed is not installed; embeddings cannot run."
                ) from exc
            logger.info("Loading embedding model %s (ONNX)", self.model_name)
            try:
                model = TextEmbedding(
                    self.model_name,
                    threads=self._settings.embedding_threads,
                )
            except Exception as exc:  # noqa: BLE001 - surfaced as a configuration error
                raise ConfigurationError(
                    f"Could not load embedding model '{self.model_name}': {exc}"
                ) from exc
            self._model = model
            probe = self._encode(["dimension probe"])
            self._dimensions = len(probe[0])
            expected = _EXPECTED_DIMENSIONS.get(self.model_name)
            if expected and self._dimensions != expected:
                logger.warning(
                    "Embedding model %s produced %d dimensions (expected %d). "
                    "Rebuild the index if you changed models.",
                    self.model_name,
                    self._dimensions,
                    expected,
                )
            logger.info("Embedding model ready (%d dimensions)", self._dimensions)

    def _require_model(self) -> TextEmbedding:
        if self._model is None:
            self.load()
        assert self._model is not None
        return self._model

    # --- Encoding --------------------------------------------------------
    @staticmethod
    def _l2_normalise(vector: list[float]) -> list[float]:
        """Unit-normalise, guarding against a zero vector.

        ``fastembed`` already returns normalised vectors for this model; doing it here
        keeps the guarantee explicit and independent of upstream defaults.
        """
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            return list(vector)
        return [value / norm for value in vector]

    def _encode(self, texts: list[str]) -> list[list[float]]:
        model = self._require_model()
        vectors = model.embed(
            texts,
            batch_size=self._settings.embedding_batch_size,
        )
        return [self._l2_normalise([float(value) for value in row]) for row in vectors]

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query into a normalised vector."""
        return self._encode([text])[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of documents, returning normalised vectors."""
        if not texts:
            return []
        return self._encode(texts)
