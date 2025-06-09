# Ingest and enrich scientific papers on MOF synthesis using LLMs and vector databases
# Author: shiboli & Gemini
# Date: 2025-06-09
# Version: 1.4.0

import os
import json
import time
from openai import OpenAI
import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from typing import Dict, Any

# Import our custom console manager and necessary Rich components
from app.core.logger import console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn

# Load environment variables from .env file once at the top
load_dotenv()

PAPERS_DIR = os.getenv("PAPERS_DIR")
DB_PATH = os.getenv("DB_PATH")
LLM_MODEL = os.getenv("DEEPSEEK_REASONER_MODEL")
LLM_BASE_URL = os.getenv("DEEPSEEK_BASE_URL")
API_KEY = os.getenv("DEEPSEEK_API_KEY") 
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")


def create_extraction_prompt(full_text: str) -> tuple[str, str]:
    """
    Creates the system and user prompts for LLM-based data extraction.
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


# --- FINAL FIX to the flatten_metadata function ---
def flatten_metadata(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Flattens a nested dictionary and sanitizes all values to be of a type
    that ChromaDB robustly accepts (str, int, float).
    Converts None, bool, and other types to their string representation.
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
            
    # Now, sanitize every value to ensure it's a str, int, or float
    sanitized_meta = {}
    for key, value in flat_meta.items():
        if value is None:
            # Convert None to the string "None"
            sanitized_meta[key] = "None"
        elif isinstance(value, bool):
            # Convert boolean to its string representation
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
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            paper_text = f.read()

        # Step 1: LLM Extraction
        console.info(f"Sending request to LLM for '{filename}'...")
        system_prompt, user_prompt = create_extraction_prompt(paper_text)
        
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            response_format={"type": "json_object"},
            temperature=0
        )
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
        
        # Step 4: Storage with sanitized and flattened metadata
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


def main():
    """
    Main function to orchestrate the ingestion and enrichment of MOF synthesis data.
    """
    console.rule("MOF Data Ingestion & Enrichment V1.4.0")
    console.info("Initializing clients and models...")
    
    if not API_KEY:
        console.error("DEEPSEEK_API_KEY not found in environment variables. Please check your .env file.")
        exit(1)
        
    openai_client = OpenAI(
        api_key=API_KEY,
        base_url=LLM_BASE_URL
    )
    console.info(f"LLM client configured for model '{LLM_MODEL}' at '{openai_client.base_url}'")

    try:
        console.info(f"Loading local sentence transformer model ({EMBEDDING_MODEL})...")
        embed_model = SentenceTransformer(EMBEDDING_MODEL, device='cpu')
        console.success("All models and clients initialized successfully.")
    except Exception as e:
        console.exception("Failed to initialize embedding model. Exiting.")
        exit(1)

    db = chromadb.PersistentClient(path=DB_PATH)
    chroma_collection = db.get_or_create_collection(name=COLLECTION_NAME)
    
    # --- State Management & File Discovery ---
    PROCESSED_LOG_FILE = os.path.join(os.path.dirname(__file__), "processed_files.log")
    processed_files = set()
    try:
        if os.path.exists(PROCESSED_LOG_FILE):
            with open(PROCESSED_LOG_FILE, 'r') as f:
                processed_files = set(f.read().splitlines())
    except IOError as e:
        console.warning(f"Could not read processed files log: {e}")

    if not PAPERS_DIR or not os.path.isdir(PAPERS_DIR):
        console.error(f"Papers directory not found or not configured: {PAPERS_DIR}")
        exit(1)
        
    all_files = [f for f in os.listdir(PAPERS_DIR) if f.endswith(".md")]
    files_to_process = sorted(list(set(all_files) - processed_files))

    if not files_to_process:
        console.info("No new papers to process. All files are up to date.")
    else:
        console.info(f"Found {len(files_to_process)} new paper(s) to process.")
        
        successful_ingestions = 0
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}", justify="left"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console._console 
        ) as progress:
            task = progress.add_task("[cyan]Ingesting papers...", total=len(files_to_process))

            for filename in files_to_process:
                progress.update(task, description=f"[cyan]Processing [bold]{filename}[/bold]")
                filepath = os.path.join(PAPERS_DIR, filename)
                
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
    console.rule("Configuration Check")
    console.info(f"Using LLM model: {LLM_MODEL}")
    console.info(f"Using embedding model: {EMBEDDING_MODEL}")
    console.info(f"Using ChromaDB path: {DB_PATH}")
    console.info(f"Using papers directory: {PAPERS_DIR}")
    console.info(f"Using collection name: {COLLECTION_NAME}")
    console.info(f"Using LLM base URL: {LLM_BASE_URL if LLM_BASE_URL else 'Default OpenAI URL'}")
    console.info(f"Using API key: {'Set' if API_KEY else 'Not Set'}")
    
    main()