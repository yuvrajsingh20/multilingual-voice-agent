"""FastAPI application.

The app owns one :class:`~app.runtime.Runtime`, built at startup and stored on
``app.state``. Nothing reaches for a global.
"""

from __future__ import annotations

from fastapi import FastAPI

from app.api.routes import router
from app.config import Settings, get_settings
from app.observability import configure_logging
from app.orchestrator import ConversationOrchestrator
from app.runtime import Runtime, build_runtime


def create_app(settings: Settings | None = None, runtime: Runtime | None = None) -> FastAPI:
    resolved = settings or get_settings()
    configure_logging(resolved.log_level)

    application = FastAPI(
        title="Multilingual Debt Voice Agent",
        version=resolved.policy_version,
        description=(
            "Foundation for a commercial-bank debt recovery voice agent. "
            "Deterministic policy engine plus interface boundaries; no telephony, "
            "STT, TTS, model or banking backend is connected."
        ),
    )
    resolved_runtime = runtime or build_runtime(resolved)
    application.state.runtime = resolved_runtime
    # Built once: the orchestrator reads the registry's tool specs at
    # construction, and that work should not repeat on every turn.
    application.state.orchestrator = ConversationOrchestrator(resolved_runtime)
    application.include_router(router)
    return application


app = create_app()
