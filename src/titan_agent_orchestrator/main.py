"""FastAPI application entry point."""

from fastapi import FastAPI

from titan_agent_orchestrator.adapters.titan import TitanCapabilityGateway
from titan_agent_orchestrator.api.routes import router
from titan_agent_orchestrator.orchestration.engine import OrchestrationEngine
from titan_agent_orchestrator.persistence.sqlite import SQLiteRepository
from titan_agent_orchestrator.settings import settings


def create_app(
    repository: SQLiteRepository | None = None,
    gateway: TitanCapabilityGateway | None = None,
) -> FastAPI:
    app = FastAPI(title="Titan Agent Orchestrator", version="0.1.0")
    app.state.engine = OrchestrationEngine(
        repository or SQLiteRepository(settings.database_url),
        gateway
        or TitanCapabilityGateway(
            settings.titan_control_plane_url,
            settings.titan_capability_gateway_token,
        ),
    )
    app.include_router(router)

    @app.get("/health", tags=["health"])
    def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()


def main() -> None:
    import uvicorn

    uvicorn.run("titan_agent_orchestrator.main:app", host="127.0.0.1", port=8010)
