from __future__ import annotations

from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes.completions import router as completions_router
from api.routes.metrics import router as metrics_router
from core.classifier_factory import get_classifier
from core.context_manager import ContextManager
from core.llm_client import LLMClient
from core.prometheus_metrics import RouterPrometheusMetrics
from core.router import ModelRouter
from core.settings import get_settings
from db.queries import MetricsRecorder


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        settings = get_settings()
        shared_http_client = httpx.AsyncClient(timeout=30.0)

        metrics = MetricsRecorder(settings.sqlite_path)
        await metrics.init()

        app.state.settings = settings
        app.state.classifier = get_classifier()
        app.state.model_router = ModelRouter(settings=settings)
        app.state.context_manager = ContextManager(
            settings=settings,
            http_client=shared_http_client,
        )
        app.state.llm_client = LLMClient(
            settings=settings,
            http_client=shared_http_client,
        )
        app.state.metrics = metrics
        app.state.prom_metrics = RouterPrometheusMetrics()
        app.state.shared_http_client = shared_http_client

        try:
            yield
        finally:
            await shared_http_client.aclose()
            await app.state.llm_client.aclose()

    app = FastAPI(
        title="Dynamic LLM Routing API",
        description="OpenAI-compatible smart router that optimizes LLM cost and latency.",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(completions_router, prefix="/v1")
    app.include_router(metrics_router)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
