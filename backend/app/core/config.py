"""Application configuration.

All environment-specific values are read here. Nothing else in the codebase reads
``os.environ`` directly, and no secret is ever logged or echoed.
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/app/core/config.py -> backend/
BACKEND_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_DIR.parent


class Settings(BaseSettings):
    """Runtime settings, populated from the environment / ``backend/.env``."""

    model_config = SettingsConfigDict(
        env_file=(BACKEND_DIR / ".env", BACKEND_DIR / ".env.local"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Service identity -------------------------------------------------
    app_name: str = "HDFC Fund Facts"
    environment: Literal["local", "development", "test", "production"] = "local"
    service_name: str = "niva-api"
    log_level: str = "INFO"

    # --- LLM (server-side only) ------------------------------------------
    llm_provider: Literal["gemini", "fake"] = "gemini"
    llm_model: str = "gemini-3.8-flash"
    llm_temperature: float = 0.1
    llm_max_output_tokens: int = 512
    llm_timeout_seconds: float = 30.0
    llm_max_attempts: int = 2
    # Excluded from every model_dump(): a settings dump ends up in logs and debug
    # payloads, and the key must never travel that way.
    gemini_api_key: str = Field(default="", exclude=True, repr=False)

    # --- Embeddings (local model, no external API) ------------------------
    # Runs on ONNX Runtime via fastembed. Keep 1 thread on small shared-CPU hosts
    # such as Render Free (0.1 CPU): extra ONNX threads only add scheduler overhead.
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_threads: int = Field(default=1, ge=1, le=32)

    # ONNX Runtime's CPU memory arena is the dominant term in this process's RSS.
    # It grows to the largest activation it has ever seen and never returns the
    # memory to the OS, so the cost is set by the biggest batch rather than by the
    # corpus (77 chunks, 384 dims) — the vectors themselves are under 1 MB.
    # Measured peak RSS of the full 77-chunk corpus, threads=1:
    #     arena on,  batch 32 -> 451 MB      arena off, batch 32 -> 353 MB
    #     arena on,  batch  8 -> 301 MB      arena off, batch  8 -> 275 MB
    #     arena on,  batch  1 -> 255 MB      arena off, batch  1 -> 254 MB
    # With the arena disabled the same work is also *faster* (9.2s vs 14.2s for the
    # full corpus), because the arena's grow-and-copy bookkeeping costs more than
    # plain malloc/free at this corpus size. Leaving it off keeps peak memory bounded
    # by a single forward pass instead of by the history of every batch ever run.
    embedding_enable_cpu_mem_arena: bool = False
    # Chunks per forward pass. Kept small so peak activation memory stays bounded on
    # a 512 MB host. Measured peak of the full real ingest run (arena off):
    #     batch 8 -> 399 MB      batch 4 -> 347 MB      batch 2 -> 339 MB
    # 4 is the measured sweet spot: below 4 the gain is marginal, and 4 was also
    # faster than 8 (19s vs 31s), so the smaller batch costs nothing here.
    embedding_batch_size: int = Field(default=4, ge=1, le=256)
    # Where fastembed stores the downloaded ONNX weights. Empty means "library
    # default". Previously render.yaml set FASTEMBED_CACHE_PATH but no field read it,
    # so the variable was silently dropped by ``extra="ignore"`` and the model was
    # cached wherever fastembed chose. On Render the weights land in the source tree
    # instead of a scratch dir.
    fastembed_cache_path: str = ""

    # --- Vector store -----------------------------------------------------
    # NOTE: "persistent" here means persistent-mode Chroma, *not* durable storage.
    # Chroma keeps its files on the local filesystem and only reloads them if that
    # filesystem outlives the process. On Render Free the filesystem is ephemeral, so
    # the index is empty on every cold start and must be rebuilt at boot. Point this
    # at a scratch dir (/tmp/chroma) rather than the source tree: /tmp is guaranteed
    # writable and is not part of the deployed artifact.
    chroma_mode: Literal["persistent", "ephemeral"] = "persistent"
    chroma_persist_directory: str = "./data/chroma"
    chroma_collection: str = "hdfc_mutual_fund_faq"

    # --- Retrieval --------------------------------------------------------
    top_k: int = Field(default=5, ge=1, le=20)
    similarity_threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    retrieve_candidates: int = Field(default=12, ge=1, le=50)
    max_context_chars: int = Field(default=9000, ge=500, le=100_000)
    max_message_length: int = Field(default=1000, ge=10, le=4000)

    # --- Ingestion --------------------------------------------------------
    sources_file: str = "./sources.json"
    ingestion_user_agent: str = "HDFCFundFactsBot/1.0 (+research prototype; contact: repo owner)"
    ingestion_timeout_seconds: float = 20.0
    ingestion_max_bytes: int = 4_000_000
    ingestion_max_attempts: int = 3
    ingestion_min_content_chars: int = 200
    auto_ingest_on_startup: bool = False

    # --- API / security ---------------------------------------------------
    # Comma-separated in the environment; exposed as a list via allowed_origin_list.
    allowed_origins: str = "http://localhost:3000"
    # Comma-separated hostname suffixes, e.g. ".onrender.com". Lets a deployment
    # accept its own frontend without knowing the hostname in advance, so the
    # default Render setup needs no manual CORS value. These are matched as
    # https-only subdomains; they are NOT credentials and grant no access on
    # their own, they only relax the browser's same-origin check.
    allowed_origin_suffixes: str = ""
    enable_docs: bool = True
    debug_rag: bool = False
    max_request_bytes: int = 16_384

    # --- Chunking ---------------------------------------------------------
    chunk_target_tokens: int = Field(default=700, ge=100, le=2000)
    chunk_overlap_tokens: int = Field(default=80, ge=0, le=600)
    chunk_min_tokens: int = Field(default=40, ge=10, le=1000)
    chunk_compact_token_limit: int = Field(default=140, ge=40, le=600)

    # --- Derived paths ----------------------------------------------------
    @property
    def backend_dir(self) -> Path:
        return BACKEND_DIR

    @property
    def repo_root(self) -> Path:
        return REPO_ROOT

    @property
    def data_dir(self) -> Path:
        return BACKEND_DIR / "data"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def sources_path(self) -> Path:
        """Resolve ``sources.json`` relative to the repo root, then the backend dir."""
        candidate = Path(self.sources_file)
        if candidate.is_absolute() and candidate.exists():
            return candidate
        for base in (BACKEND_DIR, REPO_ROOT, Path.cwd()):
            resolved = (base / candidate).resolve()
            if resolved.exists():
                return resolved
        return (BACKEND_DIR / candidate).resolve()

    @property
    def chroma_path(self) -> Path | None:
        """Absolute Chroma directory when persistence is enabled, else ``None``."""
        if self.chroma_mode == "ephemeral":
            return None
        path = Path(self.chroma_persist_directory).expanduser()
        if not path.is_absolute():
            path = (BACKEND_DIR / path).resolve()
        return path

    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    @property
    def allowed_origin_list(self) -> list[str]:
        """Configured origins as a clean list (accepts comma-separated or JSON input)."""
        text = (self.allowed_origins or "").strip()
        if not text:
            return ["http://localhost:3000"]
        if text.startswith("["):
            import json

            try:
                parsed = json.loads(text)
                if isinstance(parsed, list):
                    return [str(item).strip().rstrip("/") for item in parsed if str(item).strip()]
            except json.JSONDecodeError:
                pass
        return [origin.strip().rstrip("/") for origin in text.split(",") if origin.strip()]

    @property
    def allowed_origin_suffix_list(self) -> list[str]:
        """Configured hostname suffixes, normalised to a leading dot, no scheme."""
        text = (self.allowed_origin_suffixes or "").strip()
        if not text:
            return []
        suffixes: list[str] = []
        invalid: list[str] = []
        for raw in text.split(","):
            item = raw.strip().lower().rstrip("/")
            if not item:
                continue
            # Accept "onrender.com" and ".onrender.com"; reject anything with a
            # scheme, path, port or wildcard, since those would widen the match
            # beyond a plain subdomain rule.
            if "://" in item or "/" in item or ":" in item or "*" in item:
                invalid.append(item)
                continue
            suffixes.append(item if item.startswith(".") else f".{item}")
        for bad in invalid:
            print(
                f"WARNING: ignoring invalid ALLOWED_ORIGIN_SUFFIXES entry: {bad!r}",
                file=sys.stderr,
            )
        return suffixes

    # --- Validators -------------------------------------------------------
    @field_validator("log_level")
    @classmethod
    def _upper_log_level(cls, value: str) -> str:
        level = value.upper()
        if level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            return "INFO"
        return level

    @model_validator(mode="after")
    def _check_overlap(self) -> "Settings":
        if self.chunk_overlap_tokens >= self.chunk_target_tokens:
            object.__setattr__(self, "chunk_overlap_tokens", self.chunk_target_tokens // 4)
        return self

    # --- Startup validation ----------------------------------------------
    def validate_runtime(self) -> list[str]:
        """Return a list of human-readable configuration warnings.

        Deliberately never includes secret values.
        """
        warnings: list[str] = []
        origins = self.allowed_origin_list
        if not self.gemini_api_key:
            warnings.append(
                f"LLM provider '{self.llm_provider}' is configured but GEMINI_API_KEY is missing. "
                "Factual questions will be answered from retrieval only, or rejected."
            )
        if self.is_production:
            if self.enable_docs:
                warnings.append("ENVIRONMENT=production while enable_docs=true: /docs is publicly reachable.")
            if self.debug_rag:
                warnings.append("ENVIRONMENT=production with DEBUG_RAG=true: retrieval diagnostics are exposed.")
            if any(origin.strip() == "*" for origin in origins):
                warnings.append("ALLOWED_ORIGINS contains '*'. Wildcard CORS is not allowed in production.")
            elif origins == ["http://localhost:3000"]:
                warnings.append("ALLOWED_ORIGINS still points at localhost. Set the deployed frontend origin.")
            chroma_path = self.chroma_path
            if chroma_path is not None and BACKEND_DIR in chroma_path.parents:
                warnings.append(
                    f"CHROMA_PERSIST_DIRECTORY is inside the deployed source tree ({chroma_path}). "
                    "Use a scratch dir such as /tmp/chroma."
                )
            if self.embedding_batch_size > 32:
                warnings.append(
                    f"EMBEDDING_BATCH_SIZE={self.embedding_batch_size} raises peak RSS sharply on a "
                    "512 MB host; 4 is the measured sweet spot."
                )
        return warnings


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings singleton."""
    return Settings()


def reset_settings_cache() -> None:
    """Clear the cache (used by tests that mutate the environment)."""
    get_settings.cache_clear()
