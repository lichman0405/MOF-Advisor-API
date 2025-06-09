# The module provides Pydantic schemas for validating API requests and responses related to MOF synthesis suggestions.
# It includes models for synthesis requests, successful responses, and error responses.
# Author: shiboli
# Date: 2025-06-09


"""
Pydantic Schemas for API request and response validation.
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Any

class SynthesisRequest(BaseModel):
    """
    The request model for the /suggest endpoint.
    """
    metal_site: str = Field(
        ...,
        example="Copper",
        description="The desired metal site for the MOF synthesis."
    )
    organic_linker: str = Field(
        ...,
        example="BTC",
        description="The desired organic linker for the MOF synthesis."
    )

class SynthesisResponse(BaseModel):
    """
    The response model for a successful synthesis suggestion.
    """
    status: str = "success"
    suggestion: Dict[str, Any] = Field(
        ...,
        description="A structured dictionary containing the suggested synthesis protocol."
    )
    sources: List[Dict[str, Any]] = Field(
        ...,
        description="A list of source documents from the knowledge base that informed the suggestion."
    )

class ErrorResponse(BaseModel):
    """
    The response model for an error.
    """
    status: str = "error"
    message: str

if __name__ == "__main__":
    # Example usage of the schemas
    request_example = SynthesisRequest(
        metal_site="Copper",
        organic_linker="BTC"
    )
    
    response_example = SynthesisResponse(
        suggestion={
            "synthesis_method": "Hydrothermal",
            "temperature_celsius": 150,
            "time_hours": 24,
            "solvent": ["DMF", "Water"],
            "notes": "Ensure to use high purity reagents."
        },
        sources=[
            {"title": "MOF Synthesis Guide", "url": "https://example.com/mof-guide"}
        ]
    )
    
    error_example = ErrorResponse(
        message="Invalid input parameters."
    )
    
    print(request_example.model_dump_json(indent=2))
    print(response_example.model_dump_json(indent=2))
    print(error_example.model_dump_json(indent=2))