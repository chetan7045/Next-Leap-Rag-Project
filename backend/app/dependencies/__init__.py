"""FastAPI dependency providers for routes."""

from __future__ import annotations

from fastapi import Request

from app.dependencies.services import ServiceContainer


def get_services(request: Request) -> ServiceContainer:
    """Return the process-wide service container attached during startup."""
    container = getattr(request.app.state, "container", None)
    if container is None:  # pragma: no cover - only if lifespan did not run
        from app.dependencies.services import get_container

        container = get_container()
        request.app.state.container = container
    return container


def get_app_settings(request: Request) -> "object":
    return get_services(request).settings
