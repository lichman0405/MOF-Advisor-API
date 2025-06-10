# -*- coding: utf-8 -*-
# API endpoints for the MOF-Advisor-API.
# This version dispatches long-running ingestion tasks to a Celery worker queue.
# Author: shiboli & Gemini
# Date: 2025-06-10
# Version: 3.0.0

import os
from fastapi import APIRouter, HTTPException, status, UploadFile, File
from pydantic import BaseModel
from typing import List
from app.models.schemas import SynthesisRequest, SynthesisResponse, ErrorResponse, IngestionResponse
from app.core.rag_service import rag_service
from celery_worker import process_paper_task
from app.core.logger import console
from app.config import settings

# Create a new router instance
router = APIRouter()

PROCESSED_LOG_FILE = os.path.join(os.path.dirname(__file__), '..', '..', 'scripts', 'processed_files.log')

def get_processed_files() -> set:
    """Reads the log of processed filenames to avoid duplicates."""
    if not os.path.exists(PROCESSED_LOG_FILE):
        return set()
    try:
        with open(PROCESSED_LOG_FILE, 'r', encoding='utf-8') as f:
            return set(line.strip() for line in f)
    except IOError:
        console.warning(f"Could not read processed files log at {PROCESSED_LOG_FILE}")
        return set()
    
@router.post(
    "/suggest",
    response_model=SynthesisResponse,
    responses={
        status.HTTP_400_BAD_REQUEST: {"model": ErrorResponse, "description": "Chemically implausible request"},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": ErrorResponse, "description": "Internal server error"}
    },
    summary="Suggest a MOF Synthesis Protocol"
)

async def suggest_synthesis_protocol(request: SynthesisRequest):
    """
    Receives a request with a metal site and organic linker, and returns
    a suggested synthesis protocol.
    """
    try:
        console.info(f"Handling /suggest request for {request.model_dump()}")
        result = rag_service.query_synthesis_method(
            metal=request.metal_site,
            organic_linker=request.organic_linker
        )
        return SynthesisResponse(**result)
    except ValueError as e:
        console.warning(f"Bad request due to value error: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception:
        console.exception("An unexpected error occurred in /suggest endpoint.")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An internal server error occurred.")
class IngestionStatusResponse(BaseModel):
    total_papers_in_directory: int
    total_papers_processed: int
    processed_files: List[str]

@router.get(
    "/ingest/status", 
    response_model=IngestionStatusResponse, 
    summary="Get Ingestion Status"
)
async def get_ingestion_status():
    """Returns a summary of the ingestion process based on the log file."""
    try:
        all_files = [f for f in os.listdir(settings.PAPERS_DIR) if f.endswith(".md")]
        processed = sorted(list(get_processed_files()))
        return IngestionStatusResponse(
            total_papers_in_directory=len(all_files),
            total_papers_processed=len(processed),
            processed_files=processed
        )
    except Exception as e:
        console.exception("Error getting ingestion status.")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


# --- Ingestion Endpoints (Refactored to use Celery) ---

@router.post(
    "/ingest/file",
    response_model=IngestionResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Ingest a single paper via Celery"
)
async def ingest_single_file(file: UploadFile = File(...)):
    """
    Accepts a single .md file, saves it, and dispatches an ingestion task
    to the Celery queue. Checks for duplicates before dispatching.
    """
    processed_files = get_processed_files()
    filename = file.filename
    
    if filename in processed_files:
        console.warning(f"API received a duplicate file, skipping: {filename}")
        return IngestionResponse(
            message=f"File '{filename}' has already been processed and was skipped.",
            files_accepted=0,
            filenames=[]
        )

    # Save the file to the shared papers directory
    save_path = os.path.join(str(settings.PAPERS_DIR), str(filename))
    file_content_bytes = await file.read()
    with open(save_path, "wb") as buffer:
        buffer.write(file_content_bytes)
    
    # --- CORE CHANGE: Dispatch the task to Celery ---
    # .delay() is the shortcut to send a task message to the broker (Redis).
    console.info(f"Dispatching task to Celery for file: {filename}")
    process_paper_task.delay(filename=filename, file_content=file_content_bytes.decode('utf-8'))
    
    return IngestionResponse(
        message=f"File '{filename}' was accepted and dispatched for processing.",
        files_accepted=1,
        filenames=[filename] if filename is not None else []
    )


@router.post(
    "/ingest/files",
    response_model=IngestionResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Ingest a batch of papers via Celery"
)
async def ingest_batch_files(files: List[UploadFile] = File(...)):
    """
    Accepts multiple .md files, saves new ones, and dispatches ingestion
    tasks to the Celery queue for each new file.
    """
    processed_files = get_processed_files()
    accepted_files_for_processing = []

    for file in files:
        filename = file.filename
        if filename in processed_files:
            console.warning(f"API received a duplicate file in batch, skipping: {filename}")
            continue

        save_path = os.path.join(str(settings.PAPERS_DIR), str(filename))
        file_content_bytes = await file.read()
        with open(save_path, "wb") as buffer:
            buffer.write(file_content_bytes)
            
        # Dispatch a task for each valid new file
        console.info(f"Dispatching task to Celery for file: {filename}")
        process_paper_task.delay(filename=filename, file_content=file_content_bytes.decode('utf-8'))
        accepted_files_for_processing.append(filename)
    
    if not accepted_files_for_processing:
        return IngestionResponse(
            message="All submitted files were duplicates and were skipped.",
            files_accepted=0,
            filenames=[]
        )

    return IngestionResponse(
        message=f"Accepted {len(accepted_files_for_processing)} new file(s). They have been dispatched for background processing.",
        files_accepted=len(accepted_files_for_processing),
        filenames=accepted_files_for_processing
    )