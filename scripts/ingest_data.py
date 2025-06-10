# Ingest and enrich scientific papers on MOF synthesis using LLMs and vector databases.
# This version implements a two-stage process:
# 1. Identify all distinct synthesis procedures in a single paper.
# 2. Extract detailed parameters for each identified procedure individually.
# This "one-synthesis-one-entry" approach ensures data integrity and granularity.
# Author: shiboli
# Date: 2025-06-10
# Version: 0.1.0
import os
import json
import time
import argparse
from openai import OpenAI
import chromadb
from sentence_transformers import SentenceTransformer
import codecs
from typing import Dict, Any, Optional

from app.core.logger import console
from app.config import settings
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn

PROCESSED_LOG_FILE = os.path.join(os.path.dirname(__file__), "processed_files.log")


def robust_json_loads(json_string: str) -> Optional[Dict[str, Any]]:
    """
    A robust JSON parser that attempts to fix common LLM-generated errors,
    especially invalid backslash escape sequences.

    Args:
        json_string (str): The potentially malformed JSON string from the LLM.

    Returns:
        Optional[Dict[str, Any]]: The parsed dictionary, or None if parsing fails.
    """
    try:
        # First, try the standard parser
        return json.loads(json_string)
    except json.JSONDecodeError as e:
        console.warning(f"Standard JSON parsing failed: {e}. Attempting to fix escape sequences...")
        # Attempt to fix common backslash escape issues
        try:
            # This will interpret valid escape sequences like \n, \t
            # and also handle malformed ones by treating the backslash literally.
            repaired_string = codecs.decode(json_string, 'unicode_escape')
            console.info("Successfully repaired string with 'unicode_escape'. Trying to parse again...")
            return json.loads(repaired_string)
        except Exception as repair_e:
            console.error(f"Failed to repair and parse JSON string. Error: {repair_e}")
            # Print the faulty string for debugging purposes, using repr() to show special chars
            console.error(f"Original faulty string was: {repr(json_string)}")
            return None
        

def create_identification_prompt(full_text: str) -> tuple[str, str]:
    """
    Creates the prompt for the first stage: identifying all syntheses in a paper.
    """
    system_prompt = """
    You are an expert chemist and a data parsing specialist. Your task is to read the provided scientific paper text
    and identify every distinct synthesis procedure for a Metal-Organic Framework (MOF).
    
    You MUST return a single JSON object with one key, "syntheses", which contains a list of objects.
    Each object in the list must have two keys: "mof_name" (the name of the MOF, or a descriptive name if not specified)
    and "experimental_text" (the specific, verbatim text snippet from the paper that describes its synthesis).

    Example Response:
    {
      "syntheses": [
        {
          "mof_name": "UiO-66",
          "experimental_text": "A mixture of ZrCl4 (1.0 mmol) and H2BDC (1.0 mmol) in 20 mL of DMF was sonicated..."
        },
        {
          "mof_name": "UiO-67",
          "experimental_text": "For UiO-67, H2BPDC (1.0 mmol) was used instead of H2BDC under similar conditions..."
        }
      ]
    }

    If no synthesis procedures are found, return an empty list: {"syntheses": []}.
    """
    user_prompt = f"Please identify all distinct MOF synthesis procedures from the following text:\n\n---\n\n{full_text}"
    return system_prompt, user_prompt

def create_granular_extraction_prompt(experimental_text: str) -> tuple[str, str]:
    """
    Creates the prompt for the second stage: extracting detailed parameters from a specific text snippet.
    """
    system_prompt = """
    You are an expert chemist. Your task is to extract key synthesis parameters from the provided text snippet.
    You MUST return the information in a single, valid JSON object.
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
    """
    user_prompt = f"Please extract the synthesis parameters from this text snippet:\n\n---\n\n{experimental_text}"
    return system_prompt, user_prompt

def flatten_metadata(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Flattens a nested dictionary and robustly sanitizes all values to be of a type
    that ChromaDB accepts (str, int, float, bool).
    Crucially, it converts all Python `None` types to the string "None".
    """
    flat_meta = {}
    
    # First, flatten the nested dictionaries
    for key, value in data.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                new_key = f"{key}_{sub_key}"
                flat_meta[new_key] = sub_value
        elif isinstance(value, list):
            # Convert list to a comma-separated string
            flat_meta[key] = ", ".join(map(str, value or []))
        else:
            flat_meta[key] = value
            
    # Now, robustly sanitize every value
    sanitized_meta = {}
    for key, value in flat_meta.items():
        if value is None:
            # THE CORE FIX: Explicitly convert Python None to a string "None"
            sanitized_meta[key] = "None"
        elif isinstance(value, (str, int, float, bool)):
            # Keep allowed types as they are
            sanitized_meta[key] = value
        else:
            # As a fallback, convert any other unexpected type to its string representation
            sanitized_meta[key] = str(value)
            
    return sanitized_meta

def process_paper_for_all_syntheses(filename: str, filepath: str, client: OpenAI, embedding_model, collection) -> int:
    """
    The main processing function for a single paper, implementing the two-stage pipeline.
    Returns the number of successfully ingested entries for this paper.
    Args:
        filename (str): The name of the paper file.
        filepath (str): The full path to the paper file.
        client (OpenAI): The OpenAI client for LLM interactions.
        embedding_model: The embedding model for vectorization.
        collection: The ChromaDB collection to store the results.
    Returns:
        int: The number of successful entries ingested from this paper. 
    """
    successful_entries = 0
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            paper_text = f.read()

        # --- Stage 1: Identify all synthesis snippets ---
        console.info(f"Stage 1: Identifying all syntheses in '{filename}'...")
        id_system_prompt, id_user_prompt = create_identification_prompt(paper_text)
        response = client.chat.completions.create(
            model=settings.active_llm_config.model,
            messages=[{"role": "system", "content": id_system_prompt}, {"role": "user", "content": id_user_prompt}],
            response_format={"type": "json_object"},
            temperature=0.0,  # Ensure deterministic output
        )
        parsed_response = robust_json_loads(response.choices[0].message.content)
        if not parsed_response:
            # If parsing fails, we cannot continue with this paper.
            raise ValueError("Failed to parse the initial synthesis identification response from LLM.")

        synthesis_list = parsed_response.get("syntheses", [])

        if not synthesis_list:
            console.warning(f"No synthesis procedures found in '{filename}'. Skipping.")
            return 0
        
        console.success(f"Found {len(synthesis_list)} potential synthesis procedure(s) in '{filename}'.")

        # --- Stage 2: Process each snippet individually ---
        for i, synthesis_info in enumerate(synthesis_list):
            mof_name = synthesis_info.get("mof_name", f"synthesis_{i+1}")
            exp_text = synthesis_info.get("experimental_text")

            # Generate a unique ID for this specific entry
            unique_id = f"{filename}_{mof_name}".replace(" ", "_")
            
            console.info(f"Stage 2: Extracting details for '{mof_name}' from '{filename}'...")
            ext_system_prompt, ext_user_prompt = create_granular_extraction_prompt(exp_text)
            
            detail_response = client.chat.completions.create(
                model=settings.active_llm_config.model,
                messages=[{"role": "system", "content": ext_system_prompt}, {"role": "user", "content": ext_user_prompt}],
                response_format={"type": "json_object"},
                temperature=0.0,  # Ensure deterministic output
            )
            structured_data = robust_json_loads(detail_response.choices[0].message.content)
            if not structured_data:
                # If this specific snippet fails, log it and skip to the next one
                console.error(f"Skipping entry for '{mof_name}' due to a JSON parsing failure in Stage 2.")
                continue # This 'continue' will skip the rest of the current loop iteration
            
            # Ensure the mof_name from this extraction is consistent
            structured_data['mof_name'] = mof_name
            console.display_data_as_table(structured_data, f"{filename} -> {mof_name}")

            # Create document chunk and store in DB
            document_chunk = f"MOF Name: {mof_name}\nSynthesis Method: {structured_data.get('synthesis_method', 'N/A')}..."
            document_embedding = embedding_model.encode(document_chunk, normalize_embeddings=True)
            metadata_for_db = flatten_metadata(structured_data)
            
            collection.add(
                ids=[unique_id],
                embeddings=[document_embedding.tolist()],
                documents=[document_chunk],
                metadatas=[metadata_for_db]
            )
            console.success(f"Entry '{unique_id}' vectorized and stored.")
            successful_entries += 1
            time.sleep(1) # Rate limiting

    except Exception as e:
        console.display_error_panel(filename, str(e))
        console.exception("An error occurred during the two-stage processing:")

    return successful_entries

def main(force_rerun: bool):
    """Main function to orchestrate the ingestion."""
    console.rule("MOF Data Ingestion - Granular Mode V3.0.0")
    console.info("Initializing clients and models...")

    active_llm_config = settings.active_llm_config
    openai_client = OpenAI(api_key=active_llm_config.api_key, base_url=active_llm_config.base_url)
    console.info(f"LLM client configured for provider '{settings.LLM_PROVIDER}' at '{openai_client.base_url}'")

    try:
        console.info(f"Loading local sentence transformer model ({settings.EMBEDDING_MODEL})..."); embed_model = SentenceTransformer(settings.EMBEDDING_MODEL, device='cpu'); console.success("All models and clients initialized successfully.")
    except Exception as e:
        console.exception("Failed to initialize embedding model. Exiting."); exit(1)
    db = chromadb.PersistentClient(path=settings.DB_PATH)
    if force_rerun:
        console.warning("FORCE mode enabled. Re-processing all files.")
        try:
            console.info(f"Deleting existing collection '{settings.COLLECTION_NAME}'..."); db.delete_collection(name=settings.COLLECTION_NAME); console.success("Collection deleted.")
        except Exception:
            console.warning(f"Collection '{settings.COLLECTION_NAME}' did not exist or could not be deleted.")
        if os.path.exists(PROCESSED_LOG_FILE):
            os.remove(PROCESSED_LOG_FILE); console.success("Processed files log deleted.")
    chroma_collection = db.get_or_create_collection(name=settings.COLLECTION_NAME)
    processed_files = set();
    try:
        if os.path.exists(PROCESSED_LOG_FILE):
            with open(PROCESSED_LOG_FILE, 'r') as f: processed_files = set(f.read().splitlines())
    except IOError as e: console.warning(f"Could not read processed files log: {e}")
    if not settings.PAPERS_DIR or not os.path.isdir(settings.PAPERS_DIR): console.error(f"Papers directory not found: {settings.PAPERS_DIR}"); exit(1)
    all_files = [f for f in os.listdir(settings.PAPERS_DIR) if f.endswith(".md")]; files_to_process = sorted(list(set(all_files) - processed_files))
    if not files_to_process: console.info("No new papers to process.")
    else:
        console.info(f"Found {len(files_to_process)} new paper(s) to process."); total_new_entries = 0
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(), TextColumn("[progress.percentage]{task.percentage:>3.0f}%"), TimeElapsedColumn(), console=console._console) as progress:
            task = progress.add_task("[cyan]Processing papers...", total=len(files_to_process))
            for filename in files_to_process:
                progress.update(task, description=f"[cyan]Processing [bold]{filename}[/bold]")
                filepath = os.path.join(settings.PAPERS_DIR, filename)
                new_entries = process_paper_for_all_syntheses(filename, filepath, openai_client, embed_model, chroma_collection)
                if new_entries > 0:
                    with open(PROCESSED_LOG_FILE, 'a') as f: f.write(filename + '\n')
                    total_new_entries += new_entries
                progress.update(task, advance=1)
        if 'total_new_entries' in locals(): console.success(f"Ingestion summary: Added {total_new_entries} new entries from {len(files_to_process)} paper(s).")
    total_items = chroma_collection.count(); console.info(f"Total items in the knowledge base: [bold]{total_items}[/bold]")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest MOF synthesis papers into ChromaDB.", formatter_class=argparse.RawTextHelpFormatter); parser.add_argument("-f", "--force", action="store_true", help="Force re-processing of all files."); args = parser.parse_args()
    console.rule("Configuration Check"); console.display_data_as_table(settings.model_dump(), "Loaded Application Settings")
    main(force_rerun=args.force)