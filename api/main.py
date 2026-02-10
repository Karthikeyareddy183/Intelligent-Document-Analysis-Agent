"""FastAPI application factory with lifespan management."""

import traceback
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from core.config import get_settings
from core.logger import setup_logger
from api.routes import upload, query, documents, health
from api.dependencies import init_services
from api.middleware import RequestLoggingMiddleware

logger = setup_logger("api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan — startup and shutdown events."""
    config = get_settings()
    logger.info("Starting application...")

    # Initialize all services
    app.state.config = config
    app.state.services = init_services(config)

    logger.info("Application ready")
    yield
    logger.info("Application shutting down...")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    config = get_settings()

    app = FastAPI(
        title="Intelligent Document Analysis API",
        description=(
            "Multi-agent RAG system with computer vision integration. "
            "Upload documents (PDFs, images) and ask questions about their content."
        ),
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Custom middleware
    app.add_middleware(RequestLoggingMiddleware)

    # Global exception handler for debugging
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        tb = traceback.format_exception(type(exc), exc, exc.__traceback__)
        logger.error("Unhandled exception", extra={"traceback": "".join(tb)})
        return JSONResponse(
            status_code=500,
            content={"error": str(exc), "type": type(exc).__name__},
        )

    # Register routes
    app.include_router(upload.router, prefix="/api/v1", tags=["Upload"])
    app.include_router(query.router, prefix="/api/v1", tags=["Query"])
    app.include_router(documents.router, prefix="/api/v1", tags=["Documents"])
    app.include_router(health.router, prefix="/api/v1", tags=["Health"])

    return app


app = create_app()
