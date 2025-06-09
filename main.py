# The module is a FastAPI application that serves as the main entry point for the MOF-Advisor-API.
# Author: shiboli
# Date: 2025-06-09
# Version: 0.1.0

from fastapi import FastAPI
from app.api.endpoints import router as api_router
from app.core.logger import console

# Create the FastAPI app instance
app = FastAPI(
    title="MOF-Advisor-API",
    version="0.1.0",
    description="An intelligent API to suggest MOF synthesis protocols based on literature."
)

@app.on_event("startup")
async def startup_event():
    """Log a startup message."""
    console.info("Application startup successful.")
    console.rule("[bold green]MOF-Advisor-API is running[/bold green]")

# Include the API router
# All routes defined in endpoints.py will be available under the / prefix
app.include_router(api_router, prefix="/api/v1")

@app.get("/", tags=["Root"])
async def read_root():
    """A simple root endpoint to check if the API is running."""
    return {"message": "Welcome to the MOF-Advisor-API. Go to /docs for the API documentation."}