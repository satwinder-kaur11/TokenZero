from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST

router = APIRouter(tags=["metrics"])


@router.get("/metrics")
async def metrics(request: Request) -> Response:
    prom_metrics = request.app.state.prom_metrics
    return Response(content=prom_metrics.render(), media_type=CONTENT_TYPE_LATEST)
