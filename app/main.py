from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.routers import upload, query, health
from app.core.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    print(f"🚀 ResearchMind API starting — env: {settings.ENVIRONMENT}")
    yield
    print("ResearchMind API shutting down")


app = FastAPI(
    title="ResearchMind API",
    description="Multi-agent research assistant with RAG over uploaded documents.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(health.router, tags=["health"])
app.include_router(upload.router, prefix="/api/v1", tags=["documents"])
app.include_router(query.router, prefix="/api/v1", tags=["query"])