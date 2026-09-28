"""FastAPI application.

The app owns one :class:`~app.runtime.Runtime`, built at startup and stored on
``app.state``. Nothing reaches for a global.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI

from app.api.routes import router
from app.config import Settings, get_settings
from app.observability import configure_logging
from app.orchestrator import ConversationOrchestrator
from app.runtime import Runtime, build_runtime


@asynccontextmanager
async def _lifespan(application: FastAPI) -> AsyncIterator[None]:
    """Release the model and decision connection pools on shutdown.

    Only two services hold an OS resource - the LLM service and the decision
    service each own an HTTP connection pool, and only when configured.
    Everything else in the runtime is in-memory. The check is ``hasattr``
    rather than an isinstance test because both services are Protocols: a
    stand-in that owns no socket simply has nothing to close.
    """
    yield
    runtime = getattr(application.state, "runtime", None)
    close = getattr(getattr(runtime, "llm", None), "close", None)
    if callable(close):
        close()
    aclose = getattr(getattr(runtime, "decisions", None), "aclose", None)
    if callable(aclose):
        await aclose()


def create_app(settings: Settings | None = None, runtime: Runtime | None = None) -> FastAPI:
    resolved = settings or get_settings()
    configure_logging(resolved.log_level)

    application = FastAPI(
        title="Multilingual Debt Voice Agent",
        version=resolved.policy_version,
        description=(
            "Foundation for a commercial-bank debt recovery voice agent. "
            "Deterministic policy engine plus interface boundaries; no telephony, "
            "STT, TTS or banking backend is connected. The model boundary can be "
            "pointed at an OpenAI-compatible endpoint by configuration, but no "
            "model is connected by default."
        ),
        lifespan=_lifespan,
    )
    resolved_runtime = runtime or build_runtime(resolved)
    application.state.runtime = resolved_runtime
    # Built once: the orchestrator reads the registry's tool specs at
    # construction, and that work should not repeat on every turn.
    application.state.orchestrator = ConversationOrchestrator(resolved_runtime)
    application.include_router(router)
    return application


app = create_app()
