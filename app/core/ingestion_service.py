# The module is for processing and storing documents using LLMs and embeddings.
# It is designed to be used within the context of a larger application, reusing initialized clients.
# Author: shiboli
# Date: 2025-06-09
# Version: 0.1.0

import json
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from typing import Dict, Any

from app.core.logger import console
from app.config import settings

def flatten_metadata(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Flattens a nested dictionary and sanitizes values to make it compatible
    with ChromaDB metadata rules.
    """
    flat_meta = {}
    for key, value in data.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                new_key = f"{key}_{sub_key}"
                flat_meta[new_key] = sub_value
        elif isinstance(value, list):
            flat_meta[key] = ", ".join(map(str, value or []))
        else:
            flat_meta[key] = value
            
    # Final check to ensure all values are of a permissible type for ChromaDB
    sanitized_meta = {}
    for key, value in flat_meta.items():
        if value is None:
            sanitized_meta[key] = "None"
        elif isinstance(value, bool):
            sanitized_meta[key] = str(value)
        elif isinstance(value, (str, int, float)):
            sanitized_meta[key] = value
        else:
            sanitized_meta[key] = str(value)
            
    return sanitized_meta

class IngestionService:
    """
    A service class that handles the processing of a single document.
    It initializes its own clients based on the global settings.
    """
    def __init__(self):
        """Initializes the service with necessary clients and models."""
        try:
            active_llm = settings.active_llm_config
            self.llm_client = OpenAI(api_key=active_llm.api_key, base_url=active_llm.base_url)
            self.embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL, device='cpu')
        except Exception as e:
            console.exception("Failed to initialize clients in IngestionService.")
            # Propagate the exception to notify the application startup process
            raise e

    def _create_extraction_prompt(self, full_text: str) -> tuple[str, str]:
        """Creates the system and user prompts for LLM-based data extraction."""
        system_prompt = """
        You are an expert chemist and data scientist specializing in Metal-Organic Frameworks (MOFs).
        Your task is to meticulously read the entire scientific paper provided below and extract the key synthesis parameters for the main MOF synthesized in this work.

        You MUST return the information in a single, valid JSON object. Do not add any explanation or conversational text before or after the JSON object.
        The JSON structure must strictly follow this schema:

        {
          "mof_name": "string or null",
          "metal_source": { "formula": "string or null", "molar_amount": "string or null" },
          "organic_linker": { "name": "string or null", "molar_amount": "string or null" },
          "synthesis_method": "string, e.g., 'Solvothermal', 'Hydrothermal'",
          "solvent": "list of strings or null",
          "temperature_celsius": "integer or null",
          "time_hours": "integer or null",
          "modulator": "string or null",
          "yield": "string or null",
          "notes": "any other critical synthesis details or observations, like pH, activation procedure etc."
        }

        If a piece of information is not found anywhere in the paper, use null for its value.
        Prioritize information from the experimental or synthesis section, but use the entire paper's context if needed.
        """
        user_prompt = f"Now, please process the following full paper text and return the JSON object:\n\n---\n\n{full_text}"
        return system_prompt, user_prompt

    def process_and_store(self, filename: str, file_content: str, collection):
        """
        Takes file content, processes it, and stores it in the given ChromaDB collection.
        This is the core logic for ingesting a single document.
        """
        try:
            console.info(f"Background Task: Starting processing for '{filename}'")
            
            # Step 1: LLM Extraction
            system_prompt, user_prompt = self._create_extraction_prompt(file_content)
            response = self.llm_client.chat.completions.create(
                model=settings.active_llm_config.model,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                response_format={"type": "json_object"}
            )
            structured_data = json.loads(response.choices[0].message.content)
            console.display_data_as_table(structured_data, f"BG Task: {filename}")

            # Step 2: Document Chunking
            document_chunk = f"""
            MOF Name: {structured_data.get('mof_name', 'N/A')}
            Synthesis Method: {structured_data.get('synthesis_method', 'N/A')}
            Metal Source: {structured_data.get('metal_source', {}).get('formula', 'N/A')}
            Organic Linker: {structured_data.get('organic_linker', {}).get('name', 'N/A')}
            Solvent: {', '.join(structured_data.get('solvent') or [])}
            Temperature: {structured_data.get('temperature_celsius', 'N/A')} C
            Time: {structured_data.get('time_hours', 'N/A')} hours
            Notes: {structured_data.get('notes', 'N/A')}
            """
            
            # Step 3: Embedding
            document_embedding = self.embedding_model.encode(document_chunk, normalize_embeddings=True)
            
            # Step 4: Storage with flattened metadata
            metadata_for_db = flatten_metadata(structured_data)
            
            collection.add(
                embeddings=[document_embedding.tolist()],
                documents=[document_chunk],
                metadatas=[metadata_for_db],
                ids=[filename]
            )
            console.success(f"Background Task: Successfully processed and stored '{filename}'.")

        except Exception as e:
            console.display_error_panel(f"BG Task: {filename}", str(e))
            console.exception("Background ingestion task failed:")


# Create a single instance for the application to use, following the singleton pattern.
ingestion_service = IngestionService()