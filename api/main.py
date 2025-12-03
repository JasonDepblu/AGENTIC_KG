"""
FastAPI application for Agentic KG Web Interface.

This module provides the main FastAPI app with CORS, routes, and startup configuration.
"""

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from .routes import files_router, chat_router, sessions_router, graph_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    print("Starting Agentic KG API...")

    # Verify import directory exists
    import_dir = os.getenv("NEO4J_IMPORT_DIR", "./data")
    if not Path(import_dir).exists():
        print(f"Warning: Import directory not found: {import_dir}")
        print("Creating directory...")
        Path(import_dir).mkdir(parents=True, exist_ok=True)

    yield

    # Shutdown
    print("Shutting down Agentic KG API...")


# Create FastAPI app
app = FastAPI(
    title="Agentic KG API",
    description="API for Knowledge Graph Construction with Multi-Agent System",
    version="0.1.0",
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite dev server
        "http://localhost:5174",  # Alternative Vite port
        "http://localhost:3000",  # Alternative dev port
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(files_router, prefix="/api")
app.include_router(sessions_router, prefix="/api")
app.include_router(chat_router, prefix="/api")
app.include_router(graph_router)  # Already has /api/graph prefix


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "Agentic KG API",
        "version": "0.1.0",
        "docs": "/docs",
    }


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


# Mount static files for frontend (production)
frontend_dist = Path(__file__).parent.parent / "frontend" / "dist"
if frontend_dist.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
