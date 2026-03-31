from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.routers import upload, query, health
from app.routers import graph, auth, alerts
from app.core.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"FinanceIQ API starting — env: {settings.ENVIRONMENT}")
    yield
    print("FinanceIQ API shutting down")


app = FastAPI(
    title="FinanceIQ API",
    description=(
        "AI-powered financial document intelligence platform. "
        "Upload 10-K filings, earnings transcripts, and analyst reports. "
        "Ask questions, detect contradictions, traverse document relationships, "
        "get role-appropriate answers, and receive proactive alerts — "
        "all running locally with no API keys required."
    ),
    version="0.4.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Core
app.include_router(health.router,  tags=["health"])
app.include_router(upload.router,  prefix="/api/v1", tags=["documents"])
app.include_router(query.router,   prefix="/api/v1", tags=["query + audit"])

# Week 2: Knowledge graph
app.include_router(graph.router,   prefix="/api/v1", tags=["knowledge graph"])

# Week 3: Auth + roles
app.include_router(auth.router,    prefix="/api/v1", tags=["authentication"])

# Week 4: Proactive alerts
app.include_router(alerts.router,  prefix="/api/v1", tags=["proactive alerts"])