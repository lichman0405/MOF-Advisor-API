# The module is for processing and storing documents using LLMs and embeddings.
# It now implements the advanced two-stage pipeline to extract multiple syntheses
# from a single document, ensuring data integrity and consistency with the offline script.
# Author: shiboli & Gemini
# Date: 2025-06-10
# Version: 2.0.0

import os
import json
import time
import codecs
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from typing import Dict, Any, Optional, List

# Import our custom console manager and centralized settings
from app.core.logger import console
from app.config import settings

def robust_json_loads(json_string: str) -> Optional[Dict[str, Any]]:
    """
    A robust JSON parser that attempts to fix common LLM-generated errors,
    especially invalid backslash escape sequences.
    """
    try:
        return json.loads(json_string)
    except json.JSONDecodeError as e:
        console.warning(f"Standard JSON parsing failed: {e}. Attempting to fix...")
        try:
            repaired_string = codecs.decode(json_string, 'unicode_escape')
            console.info("Successfully repaired string with 'unicode_escape'. Trying to parse again...")
            return json.loads(repaired_string)
        except Exception as repair_e:
            console.error(f"Failed to repair and parse JSON string. Error: {repair_e}")
            console.error(f"Original faulty string was: {repr(json_string)}")
            return None

def flatten_metadata(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Flattens a nested dictionary and robustly sanitizes all values to be of a type
    that ChromaDB accepts (str, int, float, or bool).
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
    A service class that handles the processing of a single document via API.
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
            raise e

    def _create_identification_prompt(self, full_text: str) -> List[Dict[str, str]]:
        """Creates the messages for Stage 1: identifying all syntheses."""
        system_prompt = """You are an expert chemist and a data parsing specialist. Your task is to read the provided scientific paper text and identify every distinct synthesis procedure for a Metal-Organic Framework (MOF). You MUST return a single JSON object with one key, "syntheses", which contains a list of objects. Each object in the list must have two keys: "mof_name" (the name of the MOF, or a descriptive name if not specified) and "experimental_text" (the specific, verbatim text snippet from the paper that describes its synthesis). If no synthesis procedures are found, return an empty list: {"syntheses": []}."""
        user_prompt = f"Please identify all distinct MOF synthesis procedures from the following text:\n\n---\n\n{full_text}"
        return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

    def _create_granular_extraction_prompt(self, experimental_text: str) -> List[Dict[str, str]]:
        """Creates the messages for Stage 2: extracting detailed parameters from a snippet."""
        system_prompt = """You are an expert chemist. Your task is to extract key synthesis parameters from the provided text snippet. You MUST return the information in a single, valid JSON object. The JSON structure must strictly follow this schema: {"mof_name": "string or null", "metal_source": { "formula": "string or null", "molar_amount": "string or null" }, "organic_linker": { "name": "string or null", "molar_amount": "string or null" }, "synthesis_method": "string, e.g., 'Solvothermal', 'Hydrothermal'", "solvent": "list of strings or null", "temperature_celsius": "integer or null", "time_hours": "integer or null", "modulator": "string or null", "yield": "string or null", "notes": "any other critical synthesis details or observations, like pH, activation procedure etc."}"""
        user_prompt = f"Please extract the synthesis parameters from this text snippet:\n\n---\n\n{experimental_text}"
        return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

    def process_and_store_paper(self, filename: str, file_content: str, collection) -> int:
        """
        The main method for API ingestion. Implements the two-stage pipeline.
        Returns the number of new entries successfully added to the database.
        """
        successful_entries = 0
        try:
            # --- Stage 1: Identify all synthesis snippets ---
            console.info(f"API Ingestion - Stage 1: Identifying syntheses in '{filename}'...")
            id_messages = self._create_identification_prompt(file_content)
            
            id_response = self.llm_client.chat.completions.create(
                model=settings.active_llm_config.model,
                messages=id_messages,
                response_format={"type": "json_object"}
            )
            
            parsed_id_response = robust_json_loads(id_response.choices[0].message.content)
            if not parsed_id_response:
                raise ValueError("Failed to parse the Stage 1 (identification) response from LLM.")

            synthesis_list = parsed_id_response.get("syntheses", [])
            if not synthesis_list:
                console.warning(f"API Ingestion: No parsable synthesis procedures found in '{filename}'.")
                return 0
            
            console.success(f"API Ingestion: Found {len(synthesis_list)} potential procedure(s) in '{filename}'.")

            # --- Stage 2: Process each snippet individually ---
            for i, synthesis_info in enumerate(synthesis_list):
                mof_name = synthesis_info.get("mof_name", f"synthesis_{i+1}")
                exp_text = synthesis_info.get("experimental_text")
                if not exp_text:
                    console.warning(f"Skipping synthesis '{mof_name}' in '{filename}' due to empty experimental text.")
                    continue

                unique_id = f"{filename}_{mof_name.replace(' ', '_')}"
                
                console.info(f"API Ingestion - Stage 2: Extracting details for '{mof_name}'...")
                ext_messages = self._create_granular_extraction_prompt(exp_text)
                
                detail_response = self.llm_client.chat.completions.create(
                    model=settings.active_llm_config.model,
                    messages=ext_messages,
                    response_format={"type": "json_object"}
                )
                
                structured_data = robust_json_loads(detail_response.choices[0].message.content)
                if not structured_data:
                    console.error(f"Skipping entry '{unique_id}' due to JSON parsing failure in Stage 2.")
                    continue

                structured_data['mof_name'] = mof_name
                console.display_data_as_table(structured_data, f"API Ingest: {filename} -> {mof_name}")

                document_chunk = f"""MOF Name: {structured_data.get('mof_name', 'N/A')}\nSynthesis Method: {structured_data.get('synthesis_method', 'N/A')}\nMetal Source: {structured_data.get('metal_source', {}).get('formula', 'N/A')}\nOrganic Linker: {structured_data.get('organic_linker', {}).get('name', 'N/A')}\nSolvent: {', '.join(structured_data.get('solvent') or [])}\nTemperature: {structured_data.get('temperature_celsius', 'N/A')} C\nTime: {structured_data.get('time_hours', 'N/A')} hours\nNotes: {structured_data.get('notes', 'N/A')}"""
                document_embedding = self.embedding_model.encode(document_chunk, normalize_embeddings=True)
                metadata_for_db = flatten_metadata(structured_data)
                
                collection.add(
                    ids=[unique_id],
                    embeddings=[document_embedding.tolist()],
                    documents=[document_chunk],
                    metadatas=[metadata_for_db]
                )
                console.success(f"API Ingestion: Entry '{unique_id}' successfully stored.")
                successful_entries += 1
                time.sleep(1) # Simple rate limiting

        except Exception as e:
            console.display_error_panel(f"API Ingestion for {filename}", str(e))
            console.exception("API background ingestion task failed:")
        
        return successful_entries

# Create a single instance for the application to use
ingestion_service = IngestionService()