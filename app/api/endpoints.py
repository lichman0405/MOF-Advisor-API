# This module defines the API endpoints for suggesting MOF synthesis protocols.
# Author: shiboli
# Date: 2025-06-09
# Version: 0.1.0


from fastapi import APIRouter, HTTPException
from app.models.schemas import SynthesisRequest, SynthesisResponse, ErrorResponse
from app.core.rag_service import rag_service
from app.core.logger import console

# Create a new router
router = APIRouter()

@router.post(
    "/suggest",
    response_model=SynthesisResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Information not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    },
    summary="Suggest a MOF Synthesis Protocol"
)
async def suggest_synthesis_protocol(request: SynthesisRequest):
    """
    Receives a request with a metal site and organic linker, and returns
    a suggested synthesis protocol based on the knowledge base.
    """
    try:
        console.info(f"Handling /suggest request for {request.dict()}")
        result = rag_service.query_synthesis_method(
            metal=request.metal_site,
            organic_linker=request.organic_linker
        )
        return SynthesisResponse(**result)
    except ValueError as e:
        console.warning(f"Value error during suggestion: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        console.exception("An unexpected error occurred in /suggest endpoint.")
        raise HTTPException(status_code=500, detail="An internal server error occurred.")