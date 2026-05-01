from fastapi import FastAPI

from api.routes.completions import router as completions_router


def create_app() -> FastAPI:
    app = FastAPI(
        title="Dynamic LLM Routing API",
        description="OpenAI-compatible smart router that optimizes LLM cost and latency.",
        version="0.1.0",
    )
    app.include_router(completions_router, prefix="/v1")

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
