"""FastAPI Main Application Entrypoint.

Initializes database, CORS, route handlers, and startup lifecycle hooks.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os

from .config import BASE_DIR
from .db.database import init_db
from .data.catalog import populate_catalog_db
from .api.routes import router

app = FastAPI(
    title="AI/ML Heavy Rainfall Early Warning & Inundation System",
    description="Research prototype integrating Satellite, Radar, Observational, and NWP data.",
    version="1.0.0"
)

# Enable CORS for local React/Vite development
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "*"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def on_startup():
    """Run database migration and initialize official metadata."""
    init_db()
    populate_catalog_db()

# Mount API routes
app.include_router(router)

@app.get("/")
def root():
    return {
        "message": "RainShield Heavy Rainfall & Inundation Prediction Engine is running.",
        "documentation": "/docs",
        "service": "RainShield API",
        "health": "/api/health"
    }

@app.get("/health")
def root_health():
    return {"status": "healthy", "service": "RainShield API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
