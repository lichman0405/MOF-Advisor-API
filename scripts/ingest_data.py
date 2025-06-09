# Ingest and enrich scientific papers on MOF synthesis using LLMs and vector databases
# Author: shiboli
# Date: 2025-06-09
# Version: 0.1.0

import os
import json
import time
import argparse
from openai import OpenAI
import chromadb
from sentence_transformers import SentenceTransformer
from typing import Dict, Any
from app.core.logger import console
from app.config import settings
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn

PROCESSED_LOG_FILE = os.path.join(os.path.dirname(__file__), "processed_files.log")

def create_extraction_prompt(full_text: str) -> tuple[str, str]:
    """
    Creates the system and user prompts for LLM-based data extraction.
    Args:
        full_text (str): The full text of the scientific paper to be processed.
    Returns:
        tuple[str, str]: The system prompt and user prompt for the LLM.
    """
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


def flatten_metadata(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Flattens a nested dictionary and sanitizes all values to be of a type
    that ChromaDB robustly accepts (str, int, float, or bool).
    Converts None and other types to their string representation.
    Args:
        data (Dict[str, Any]): The nested dictionary to flatten and sanitize.
    Returns:
        Dict[str, Any]: A flattened and sanitized dictionary.
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
            
    # Now, sanitize every value to ensure it's a ChromaDB-compatible type
    sanitized_meta = {}
    for key, value in flat_meta.items():
        if value is None:
            # Convert None to the string "None" to avoid type errors
            sanitized_meta[key] = "None"
        elif isinstance(value, bool):
            # Convert boolean to its string representation "True" or "False"
            sanitized_meta[key] = str(value)
        elif isinstance(value, (str, int, float)):
            # Keep allowed types as they are
            sanitized_meta[key] = value
        else:
            # Convert any other type to its string representation as a fallback
            sanitized_meta[key] = str(value)
            
    return sanitized_meta


def process_single_paper(filename: str, filepath: str, client: OpenAI, embedding_model, collection) -> bool:
    """
    Processes a single paper file: extracts, embeds, and stores data.
    Args:
        filename (str): The name of the file being processed.
        filepath (str): The full path to the file.
        client (OpenAI): The OpenAI client for LLM interactions.
        embedding_model: The SentenceTransformer model for embeddings.
        collection: The ChromaDB collection to store the data.
    Returns:
        bool: True if processing was successful, False otherwise.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            paper_text = f.read()

        # Step 1: LLM Extraction
        system_prompt, user_prompt = create_extraction_prompt(paper_text)
        
        response = client.chat.completions.create(
            model=settings.active_llm_config.model,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            response_format={"type": "json_object"}
        )
        if response.choices[0].message.content is None:
            raise ValueError("LLM response content is None, cannot parse JSON.")
        structured_data = json.loads(response.choices[0].message.content)
        console.display_data_as_table(structured_data, filename)

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
        document_embedding = embedding_model.encode(document_chunk, normalize_embeddings=True)
        
        # Step 4: Storage with flattened and sanitized metadata
        metadata_for_db = flatten_metadata(structured_data)
        
        collection.add(
            embeddings=[document_embedding.tolist()],
            documents=[document_chunk],
            metadatas=[metadata_for_db],
            ids=[filename]
        )
        console.success(f"'{filename}' has been vectorized and stored in ChromaDB.")
        return True

    except Exception as e:
        console.display_error_panel(filename, str(e))
        console.exception("Detailed traceback:")
        return False


def main(force_rerun: bool):
    """
    Main function to orchestrate the ingestion and enrichment of MOF synthesis data.
    """
    console.rule("MOF Data Ingestion & Enrichment V2.0.0")

    # --- Initialization using centralized settings ---
    console.info("Initializing clients and models...")
    
    active_llm_config = settings.active_llm_config
    openai_client = OpenAI(api_key=active_llm_config.api_key, base_url=active_llm_config.base_url)
    console.info(f"LLM client configured for provider '{settings.LLM_PROVIDER}' at '{openai_client.base_url}'")

    try:
        console.info(f"Loading local embedding model ({settings.EMBEDDING_MODEL})...")
        embed_model = SentenceTransformer(settings.EMBEDDING_MODEL, device='cpu')
        console.success("All models and clients initialized successfully.")
    except Exception as e:
        console.exception("Failed to initialize embedding model. Exiting.")
        exit(1)

    db = chromadb.PersistentClient(path=settings.DB_PATH)

    if force_rerun:
        console.warning("FORCE mode enabled. Re-processing all files.")
        try:
            console.info(f"Deleting existing collection '{settings.COLLECTION_NAME}'...")
            db.delete_collection(name=settings.COLLECTION_NAME)
            console.success("Collection deleted.")
        except Exception:
            console.warning(f"Collection '{settings.COLLECTION_NAME}' did not exist or could not be deleted.")
            
        if os.path.exists(PROCESSED_LOG_FILE):
            os.remove(PROCESSED_LOG_FILE)
            console.success("Processed files log deleted.")

    chroma_collection = db.get_or_create_collection(name=settings.COLLECTION_NAME)
    
    processed_files = set()
    try:
        if os.path.exists(PROCESSED_LOG_FILE):
            with open(PROCESSED_LOG_FILE, 'r') as f:
                processed_files = set(f.read().splitlines())
    except IOError as e:
        console.warning(f"Could not read processed files log: {e}")

    if not settings.PAPERS_DIR or not os.path.isdir(settings.PAPERS_DIR):
        console.error(f"Papers directory not found or not configured: {settings.PAPERS_DIR}")
        exit(1)
        
    all_files = [f for f in os.listdir(settings.PAPERS_DIR) if f.endswith(".md")]
    files_to_process = sorted(list(set(all_files) - processed_files))

    if not files_to_process:
        console.info("No new papers to process. All files are up to date.")
    else:
        console.info(f"Found {len(files_to_process)} new paper(s) to process.")
        
        successful_ingestions = 0
        with Progress(
            SpinnerColumn(), TextColumn("[progress.description]{task.description}", justify="left"),
            BarColumn(), TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(), console=console._console
        ) as progress:
            task = progress.add_task("[cyan]Ingesting papers...", total=len(files_to_process))
            for filename in files_to_process:
                progress.update(task, description=f"[cyan]Processing [bold]{filename}[/bold]")
                filepath = os.path.join(settings.PAPERS_DIR, filename)
                
                if process_single_paper(filename, filepath, openai_client, embed_model, chroma_collection):
                    successful_ingestions += 1
                    with open(PROCESSED_LOG_FILE, 'a') as f:
                        f.write(filename + '\n')
                
                progress.update(task, advance=1)
                time.sleep(1)

    console.rule("Process Finished", style="green")
    if 'successful_ingestions' in locals() and files_to_process:
        console.success(f"Ingestion summary: {successful_ingestions}/{len(files_to_process)} new papers processed successfully.")
    
    total_items = chroma_collection.count()
    console.info(f"Total items in the knowledge base: [bold]{total_items}[/bold]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingest MOF synthesis papers into ChromaDB.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "-f", "--force",
        action="store_true",
        help="Force re-processing of all files by deleting the existing collection and logs."
    )
    args = parser.parse_args()
    
    main(force_rerun=args.force)