from __future__ import annotations

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from api.routes.metrics import router as metrics_router
from core.prometheus_metrics import RouterPrometheusMetrics


@pytest.mark.asyncio
async def test_metrics_endpoint_exposes_prometheus_payload() -> None:
    app = FastAPI()
    app.include_router(metrics_router)
    app.state.prom_metrics = RouterPrometheusMetrics()
    app.state.prom_metrics.observe_completion(
        tier="small",
        cost_usd=0.0012,
        latency_ms=145.0,
    )

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/metrics")

    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    body = response.text
    assert "requests_total" in body
    assert 'tier="small"' in body
    assert "cost_usd_total" in body
    assert "latency_ms_histogram" in body
