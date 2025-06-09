# -*- coding: utf-8 -*-
"""
A diagnostic script to inspect the contents of the ChromaDB knowledge base.

This script connects to the existing database and fetches a specified number
of entries, displaying their ID, document text, and metadata in a readable format.

Usage:
    python scripts/inspect_db.py
    python scripts/inspect_db.py --limit 3
"""

import os
import chromadb
import argparse
from dotenv import load_dotenv

# Import our custom console manager for beautiful output
from app.core.logger import console

# Load environment variables to get DB path and collection name
load_dotenv()

# --- Constants from Environment ---
DB_PATH = os.getenv("DB_PATH", "./chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "mof_synthesis_papers")

def inspect_database(limit: int):
    """
    Connects to ChromaDB, fetches records, and prints them.
    
    Args:
        limit (int): The maximum number of records to fetch and display.
    """
    console.rule(f"Inspecting ChromaDB Collection: '{COLLECTION_NAME}'")

    try:
        # Connect to the persistent database
        client = chromadb.PersistentClient(path=DB_PATH)
        console.info(f"Successfully connected to ChromaDB at: {DB_PATH}")
        
        # Get the existing collection
        collection = client.get_collection(name=COLLECTION_NAME)
        console.success(f"Successfully retrieved collection '{COLLECTION_NAME}'.")
        
    except Exception as e:
        console.exception(f"Failed to connect to or retrieve collection: {e}")
        return

    # Get the total number of items
    total_items = collection.count()
    if total_items == 0:
        console.warning("The knowledge base is empty. No items to display.")
        return
        
    console.info(f"Total items in collection: [bold]{total_items}[/bold]. Showing up to {limit} items.")

    # Fetch the actual records from the database
    try:
        records = collection.get(
            limit=limit,
            include=["metadatas", "documents"]  # We need metadata and the document text
        )
    except Exception as e:
        console.exception(f"Failed to fetch records from the collection: {e}")
        return

    # Display each record in a formatted way
    for i, item_id in enumerate(records['ids']):
        item_metadata = records['metadatas'][i]
        item_document = records['documents'][i]

        console.rule(f"Record {i + 1}/{len(records['ids'])} | ID: {item_id}", style="cyan")
        
        # Use our new console manager method to display the document
        console.display_text_in_panel(item_document, "Embedded Document")
        
        # Use the existing method to display the metadata table
        console.display_data_as_table(item_metadata, "Stored Metadata")

    console.rule("Inspection Complete", style="green")

if __name__ == "__main__":
    # Setup command-line argument parsing
    parser = argparse.ArgumentParser(
        description="Inspect the contents of the MOF synthesis ChromaDB."
    )
    parser.add_argument(
        "-l", "--limit",
        type=int,
        default=5,
        help="The maximum number of records to display."
    )
    args = parser.parse_args()
    
    inspect_database(limit=args.limit)