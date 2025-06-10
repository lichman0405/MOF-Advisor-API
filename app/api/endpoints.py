# This module defines the API endpoints for suggesting MOF synthesis protocols.
# Author: shiboli
# Date: 2025-06-09
# Version: 0.1.0

import os
from fastapi import APIRouter, HTTPException, status, UploadFile, File, BackgroundTasks
from typing import List

from app.models.schemas import SynthesisRequest, SynthesisResponse, ErrorResponse, IngestionResponse
from app.core.rag_service import rag_service
from app.core.ingestion_service import ingestion_service # New import
from app.core.logger import console
from app.config import settings

router = APIRouter()

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
    Receives a request with a metal site and organic linker.
    First, it validates the chemical feasibility.
    Then, it returns a suggested synthesis protocol based on the knowledge base.
    """
    try:
        console.info(f"Handling /suggest request for {request.model_dump()}")
        result = rag_service.query_synthesis_method(
            metal=request.metal_site,
            organic_linker=request.organic_linker
        )
        return SynthesisResponse(**result)
    except ValueError as e:
        # This now specifically catches our feasibility check failure
        console.warning(f"Bad request due to value error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail=str(e)
        )
    except Exception as e:
        console.exception("An unexpected error occurred in /suggest endpoint.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal server error occurred."
        )
    
# Helper function to check for duplicates
def get_processed_files() -> set:
    log_file = os.path.join(os.path.dirname(__file__), '..', '..', 'scripts', 'processed_files.log')
    if not os.path.exists(log_file):
        return set()
    with open(log_file, 'r') as f:
        return set(line.strip() for line in f)

# Background task function
def background_ingest_file(filename: str, file_content: str):
    # This function runs in the background
    # We get the collection from the already initialized rag_service to avoid re-creating clients
    collection = rag_service.collection
    ingestion_service.process_and_store_paper(filename, file_content, collection)
    # Append to log file after processing
    log_file = os.path.join(os.path.dirname(__file__), '..', '..', 'scripts', 'processed_files.log')
    with open(log_file, 'a') as f:
        f.write(filename + '\n')

@router.post(
    "/ingest/file",
    response_model=IngestionResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Ingest a single paper"
)
async def ingest_single_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    Accepts a single .md file, saves it, and schedules it for background ingestion.
    Checks for duplicates by filename.
    """
    processed_files = get_processed_files()
    filename = file.filename
    
    if filename in processed_files:
        console.warning(f"Skipping duplicate file: {filename}")
        return IngestionResponse(
            message=f"File '{filename}' has already been processed and was skipped.",
            files_accepted=0,
            filenames=[]
        )

    # Save the file to the papers directory
    save_path = os.path.join(settings.PAPERS_DIR, filename)
    file_content_bytes = await file.read()
    with open(save_path, "wb") as buffer:
        buffer.write(file_content_bytes)
    
    # Add the ingestion task to run in the background
    background_tasks.add_task(background_ingest_file, filename, file_content_bytes.decode('utf-8'))
    
    return IngestionResponse(
        message=f"File '{filename}' was accepted and is being processed in the background.",
        files_accepted=1,
        filenames=[filename]
    )

@router.post(
    "/ingest/files",
    response_model=IngestionResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Ingest a batch of papers"
)
async def ingest_batch_files(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...)
):
    """
    Accepts multiple .md files, saves them, and schedules them for background ingestion.
    Checks for duplicates by filename and only processes new files.
    """
    processed_files = get_processed_files()
    accepted_files = []

    for file in files:
        filename = file.filename
        if filename in processed_files:
            console.warning(f"Skipping duplicate file in batch: {filename}")
            continue

        save_path = os.path.join(settings.PAPERS_DIR, filename)
        file_content_bytes = await file.read()
        with open(save_path, "wb") as buffer:
            buffer.write(file_content_bytes)
            
        background_tasks.add_task(background_ingest_file, filename, file_content_bytes.decode('utf-8'))
        accepted_files.append(filename)
    
    if not accepted_files:
        return IngestionResponse(
            message="All submitted files were duplicates and were skipped.",
            files_accepted=0,
            filenames=[]
        )

    return IngestionResponse(
        message=f"Accepted {len(accepted_files)} new file(s) for background processing. See server logs for progress.",
        files_accepted=len(accepted_files),
        filenames=accepted_files
    )