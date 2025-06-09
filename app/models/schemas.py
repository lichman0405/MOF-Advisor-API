# The module provides Pydantic schemas for validating API requests and responses related to MOF synthesis suggestions.
# It includes models for synthesis requests, successful responses, and error responses.
# Author: shiboli
# Date: 2025-06-09


# -*- coding: utf-8 -*-
"""
Pydantic Schemas for API request and response validation.
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Any

class SynthesisRequest(BaseModel):
    """The request model for the /suggest endpoint."""
    metal_site: str = Field(..., example="Copper")
    organic_linker: str = Field(..., example="BTC")

class SynthesisResponse(BaseModel):
    """The response model for a successful synthesis suggestion."""
    status: str = "success"
    
    # NEW FIELD: To indicate how the suggestion was generated.
    generation_mode: str = Field(
        ...,
        example="RAG_BASED",
        description="Indicates the generation mode: 'RAG_BASED' (from knowledge base) or 'LLM_FALLBACK' (from LLM's general knowledge)."
    )
    
    suggestion: Dict[str, Any] = Field(
        ...,
        description="A structured dictionary containing the suggested synthesis protocol."
    )
    sources: List[Dict[str, Any]] = Field(
        [], # Default to an empty list
        description="A list of source documents from the knowledge base. Empty in fallback mode."
    )

class ErrorResponse(BaseModel):
    """The response model for an error."""
    status: str = "error"
    message: str


class IngestionResponse(BaseModel):
    """
    Response model for the file ingestion endpoints.
    """
    message: str
    files_accepted: int
    filenames: List[str]

if __name__ == "__main__":
    # This block is for testing the schemas directly
    example_request = SynthesisRequest(metal_site="Copper", organic_linker="BTC")
    print(example_request.model_dump_json(indent=2))

    example_response = SynthesisResponse(
        suggestion={"protocol": "Mix and heat at 200C for 12 hours."},
        sources=[{"title": "Example Paper", "url": "http://example.com"}],
        generation_mode="RAG_BASED"
    )
    print(example_response.model_dump_json(indent=2))

    example_error = ErrorResponse(message="Invalid input parameters.")
    print(example_error.model_dump_json(indent=2))