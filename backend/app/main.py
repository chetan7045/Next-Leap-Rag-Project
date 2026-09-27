"""FastAPI application factory.

Startup is intentionally cheap: the embedding model is not loaded and no document is
downloaded unless ``AUTO_INGEST_ON_STARTUP=true`` and the index is empty.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import chat, health, schemes, sources
from app.core.config import get_settings
from app.core.errors import AppError
from app.core.logging import configure_logging, get_logger, log_event, register_secret
from app.core.security import (
    RequestSizeLimitMiddleware,
    SecurityHeadersMiddleware,
    build_cors_origins,
)

logger = get_logger("app.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    # Register the key before any log record is rendered so it is scrubbed from every
    # line and traceback, including anything a third-party SDK might emit.
    register_secret(settings.gemini_api_key)
    configure_logging(settings.log_level, json_output=settings.is_production)

    # Log that configuration loaded — never its values.
    logger.info("Starting %s v1.0.0", settings.app_name)
    logger.info("LLM configuration: provider=%s model=%s", settings.llm_provider, settings.llm_model)
    logger.info("LLM configuration: loaded")
    logger.info("Embedding model: %s", settings.embedding_model)
    logger.info(
        "Vector store: collection=%s mode=%s", settings.chroma_collection, settings.chroma_mode
    )
    for warning in settings.validate_runtime():
        logger.warning(warning)

    from app.dependencies.services import build_container

    container = build_container(settings)
    app.state.container = container
    app.state.started_at = time.time()

    # Load and run the encoder once. The first real request would otherwise block for
    # the better part of a minute while the model is initialised, which reads as a hang.
    if container.index_ready():
        warm_started = time.perf_counter()
        try:
            container.embeddings.load()
            container.embeddings.embed_query("warm up")
            logger.info(
                "Encoder warm in %dms", round((time.perf_counter() - warm_started) * 1000)
            )
        except Exception as exc:  # noqa: BLE001 - never block startup on warm-up
            logger.warning("Encoder warm-up failed (%s); first request will be slower.", exc)

    if settings.auto_ingest_on_startup:
        from app.services.ingestion.pipeline import IngestionPipeline

        report = IngestionPipeline(settings, container.registry, container.chroma, container.embeddings).ingest_if_empty()
        if report is not None:
            logger.info("Startup ingestion complete: %s", report.summary())

    logger.info("Service ready (environment=%s, index_ready=%s)", settings.environment, container.index_ready())
    try:
        yield
    finally:
        logger.info("Shutting down %s", settings.app_name)
        from app.dependencies.services import set_container

        set_container(None)


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level, json_output=settings.is_production)

    app = FastAPI(
        title=settings.app_name,
        version="1.0.0",
        description=(
            "Facts-only RAG assistant for HDFC Mutual Fund schemes. "
            "Retrieval-backed answers with backend-generated citations. "
            "Not investment advice."
        ),
        lifespan=lifespan,
        docs_url="/docs" if settings.enable_docs else None,
        redoc_url="/redoc" if settings.enable_docs else None,
        openapi_url="/openapi.json" if settings.enable_docs else None,
    )

    origins = build_cors_origins(settings)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "X-Request-Id"],
        expose_headers=["X-Request-Id"],
        max_age=600,
    )
    app.add_middleware(RequestSizeLimitMiddleware, max_bytes=settings.max_request_bytes)
    app.add_middleware(SecurityHeadersMiddleware)

    app.include_router(health.router)
    app.include_router(schemes.router)
    app.include_router(sources.router)
    app.include_router(chat.router)

    @app.exception_handler(AppError)
    async def _app_error_handler(request: Request, exc: AppError) -> JSONResponse:
        log_event(
            logger,
            30,
            "handled application error",
            request_id=getattr(request.state, "request_id", "-"),
            error_type=type(exc).__name__,
            status=exc.status_code,
        )
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.to_public()})

    @app.exception_handler(Exception)
    async def _unhandled_handler(request: Request, exc: Exception) -> JSONResponse:
        # Never leak a traceback to a client.
        log_event(
            logger,
            40,
            "unhandled error",
            request_id=getattr(request.state, "request_id", "-"),
            error_type=type(exc).__name__,
            path=request.url.path,
        )
        return JSONResponse(
            status_code=500,
            content={"detail": "An unexpected error occurred. Please try again."},
        )

    logger.info("CORS allowed origins: %s", origins if origins else "(none)")
    return app


app = create_app()
