# The module defines the Celery application and the main task for paper ingestion.
# It encapsulates the "one-synthesis-one-entry" logic to be run asynchronously.
# Author: shiboli & Gemini
# Date: 2025-06-10
# Version: 3.0.0

import os
import json
import time
import codecs
from celery import Celery
from openai import OpenAI
import chromadb
from sentence_transformers import SentenceTransformer
from typing import Dict, Any, Optional, List

# Import our centralized settings and logger
# The worker needs to know about the project's configuration
from app.config import settings
from app.core.logger import console

# --- 1. Celery Application Initialization ---
# Initialize the Celery app, connecting it to our Redis message broker.
celery_app = Celery(
    "tasks",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL # Use Redis as the result backend as well
)

# Optional: Configure Celery for better robustness
celery_app.conf.update(
    task_track_started=True,
    # You can add more configurations here, like retry policies
)

# --- 2. Reusable Helper Functions ---
# These functions support the main task logic.

def robust_json_loads(json_string: str) -> Optional[Dict[str, Any]]:
    """A robust JSON parser to handle LLM quirks."""
    try:
        return json.loads(json_string)
    except json.JSONDecodeError as e:
        console.warning(f"Celery Task: Standard JSON parsing failed: {e}. Retrying with repair...")
        try:
            repaired_string = codecs.decode(json_string, 'unicode_escape')
            return json.loads(repaired_string)
        except Exception as repair_e:
            console.error(f"Celery Task: Repair failed: {repair_e}")
            return None

def flatten_metadata(data: Dict[str, Any]) -> Dict[str, Any]:
    """Flattens and sanitizes metadata for ChromaDB."""
    flat_meta = {}
    for key, value in data.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                flat_meta[f"{key}_{sub_key}"] = sub_value
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

def create_identification_prompt(full_text: str) -> List[Dict[str, str]]:
    """Creates the messages for Stage 1: identifying all syntheses."""
    system_prompt =  """
    You are an expert chemist and a data parsing specialist. 
    Your task is to read the provided scientific paper text 
    and identify every distinct synthesis procedure for a Metal-Organic Framework (MOF). 
    You MUST return a single JSON object with one key, "syntheses", which contains a list of objects. 
    Each object in the list must have two keys: "mof_name" (the name of the MOF, 
    or a descriptive name if not specified) and "experimental_text" 
    (the specific, verbatim text snippet from the paper that describes its synthesis). 
    If no synthesis procedures are found, return an empty list: {"syntheses": []}.
    """
    user_prompt = f"Please identify all distinct MOF synthesis procedures from the following text:\n\n---\n\n{full_text}"
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

def create_granular_extraction_prompt(experimental_text: str) -> List[Dict[str, str]]:
    """Creates the messages for Stage 2: extracting detailed parameters."""
    system_prompt = """
    You are an expert chemist. 
    Your task is to extract key synthesis parameters from the provided text snippet. 
    You MUST return the information in a single, valid JSON object. 
    The JSON structure must strictly follow this schema: 
    {"mof_name": "string or null", "metal_source": 
        { "formula": "string or null", "molar_amount": "string or null" }, 
     "organic_linker": 
        { "name": "string or null", "molar_amount": "string or null" }, 
     "synthesis_method": "string, e.g., 'Solvothermal', 'Hydrothermal'", 
     "solvent": "list of strings or null", 
     "temperature_celsius": "integer or null", 
     "time_hours": "integer or null", 
     "modulator": "string or null", 
     "yield": "string or null", 
     "notes": "any other critical synthesis details or observations, like pH, activation procedure etc."}
    """

    user_prompt = f"Please extract the synthesis parameters from this text snippet:\n\n---\n\n{experimental_text}"
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

# --- 3. The Celery Task Definition ---
@celery_app.task(name="process_paper_task", bind=True)
def process_paper_task(self, filename: str, file_content: str) -> str:
    """
    The main Celery task to process a single paper using the two-stage pipeline.
    `bind=True` makes `self` available, allowing access to task metadata.
    """
    console.rule(f"[bold blue]Celery Task Received[/bold blue]: Processing '{filename}' | Task ID: {self.request.id}")
    
    # --- IMPORTANT ---
    # Each Celery task runs in a separate process. Therefore, it must
    # initialize its own clients. It cannot share them with the FastAPI app.
    try:
        active_llm = settings.active_llm_config
        llm_client = OpenAI(api_key=active_llm.api_key, base_url=active_llm.base_url)
        embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL, device='cpu')
        db_client = chromadb.PersistentClient(path=settings.DB_PATH)
        collection = db_client.get_or_create_collection(name=settings.COLLECTION_NAME)
    except Exception as e:
        console.exception("Celery Task: Client initialization failed.")
        # Celery can retry tasks, here we raise the exception to trigger that if configured
        raise e

    successful_entries = 0
    try:
        # Stage 1: Identify all synthesis snippets
        console.info(f"Task {self.request.id} - Stage 1: Identifying syntheses...")
        id_messages = create_identification_prompt(file_content)
        id_response = llm_client.chat.completions.create(
            model=active_llm.model, 
            messages=id_messages, 
            response_format={"type": "json_object"},
            temperature=0.0,
            )
        synthesis_list = (robust_json_loads(id_response.choices[0].message.content) or {}).get("syntheses", [])
        
        if not synthesis_list:
            message = f"No syntheses found in '{filename}'."
            console.warning(f"Celery Task: {message}")
            return message

        console.success(f"Task {self.request.id}: Found {len(synthesis_list)} procedures.")

        # Stage 2: Process each snippet individually
        for i, s_info in enumerate(synthesis_list):
            mof_name, exp_text = s_info.get("mof_name", f"s_{i+1}"), s_info.get("experimental_text")
            if not exp_text: continue
            
            unique_id = f"{filename}_{mof_name.replace(' ', '_').replace('/', '_')}"
            console.info(f"Task {self.request.id} - Stage 2: Extracting for '{mof_name}'...")
            ext_messages = create_granular_extraction_prompt(exp_text)
            detail_response = llm_client.chat.completions.create(
                model=active_llm.model, 
                messages=ext_messages, 
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            structured_data = robust_json_loads(detail_response.choices[0].message.content)
            
            if not structured_data:
                console.error(f"Skipping entry '{unique_id}' due to parsing error."); continue
            
            structured_data['mof_name'] = mof_name

            console.display_data_as_table(structured_data, f"API Ingest: {filename} -> {mof_name}")

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

            document_embedding = embedding_model.encode(document_chunk, normalize_embeddings=True)
            metadata_for_db = flatten_metadata(structured_data)
            
            collection.add(ids=[unique_id], embeddings=[document_embedding.tolist()], documents=[document_chunk], metadatas=[metadata_for_db])
            console.success(f"Task {self.request.id}: Stored '{unique_id}'.")
            successful_entries += 1
            time.sleep(1)
        
        # Update the processed log file only on full success
        log_file = os.path.join(os.path.dirname(__file__), 'scripts', 'processed_files.log')
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        with open(log_file, 'a') as f: f.write(filename + '\n')
        
        final_message = f"Successfully processed '{filename}', added {successful_entries} new entries."
        console.success(final_message)
        return final_message

    except Exception as e:
        console.display_error_panel(f"Celery Task Error for {filename}", str(e))
        console.exception(f"Celery task {self.request.id} failed:")
        raise e