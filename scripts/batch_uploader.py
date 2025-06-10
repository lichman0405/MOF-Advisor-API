# -*- coding: utf-8 -*-
"""
A robust client-side script for batch uploading papers to the MOF-Advisor-API.

This script scans a directory for markdown files, uploads them to the API in parallel,
and then monitors the asynchronous ingestion process until completion.
"""
import os
import requests
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Tuple

# Import our custom console manager and centralized settings
from app.core.logger import console
from app.config import settings

# --- Configuration ---
# The base URL for the API, constructed for local deployment.
# Assumes the API is running on port 8088 as per our docker-compose.yml
API_BASE_URL = "http://192.168.100.208:8088/api/v1"

def upload_single_file(filepath: str) -> Tuple[str, bool, str]:
    """
    Uploads a single file to the /ingest/file endpoint.

    Args:
        filepath (str): The absolute or relative path to the file.

    Returns:
        A tuple containing (filename, success_boolean, message).
    """
    filename = os.path.basename(filepath)
    url = f"{API_BASE_URL}/ingest/file"
    try:
        with open(filepath, 'rb') as f:
            files_payload = {'file': (filename, f, 'text/markdown')}
            # Set a reasonable timeout for the request
            response = requests.post(url, files=files_payload, timeout=60)
        
        # HTTP 202 Accepted means the server received it and will process it.
        if response.status_code == 202:
            return filename, True, "Accepted for background processing."
        # If the file is a duplicate, the server might respond differently (e.g. 200 OK with a skip message)
        elif response.status_code == 200 and "skipped" in response.json().get("message", ""):
            return filename, True, "Skipped (already processed)."
        else:
            return filename, False, f"Failed with status {response.status_code}: {response.text}"
            
    except requests.exceptions.RequestException as e:
        return filename, False, f"Request failed due to a network error: {e}"
    except FileNotFoundError:
        return filename, False, "File not found on the client side."


def monitor_progress():
    """Periodically polls the /ingest/status endpoint to show progress."""
    status_url = f"{API_BASE_URL}/ingest/status"
    try:
        # Get the initial state to know the total number of files
        initial_status = requests.get(status_url, timeout=10).json()
        total_in_dir = initial_status.get('total_papers_in_directory', 0)
        
        if total_in_dir == 0:
            console.warning("No papers found in the server's data directory to monitor.")
            return

        # Use rich's status indicator for a nice loading animation
        with console._console.status("[bold green]Monitoring background ingestion...") as status:
            while True:
                current_status = requests.get(status_url, timeout=10).json()
                processed_count = current_status.get('total_papers_processed', 0)
                
                status.update(f"Processing... {processed_count} / {total_in_dir} files complete.")
                
                if processed_count >= total_in_dir:
                    break # Exit the loop when all files are processed
                
                time.sleep(5) # Poll the status every 5 seconds
                
        console.success("All background ingestion tasks appear to be complete!")

    except requests.exceptions.RequestException as e:
        console.error(f"Could not monitor progress due to a network error: {e}")


def main(directory: str, max_workers: int):
    """Finds all .md files in a directory and uploads them in parallel."""
    console.rule(f"Starting Batch Upload from Directory: '{directory}'")
    
    if not os.path.isdir(directory):
        console.error(f"Directory not found: {directory}"); return

    filepaths = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".md")]
    if not filepaths:
        console.warning("No .md files found in the specified directory."); return

    console.info(f"Found {len(filepaths)} files. Starting parallel upload with {max_workers} workers...")

    successful_uploads = []
    failed_uploads = []

    # Use ThreadPoolExecutor to upload files in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Create a future for each file upload
        future_to_filepath = {executor.submit(upload_single_file, fp): fp for fp in filepaths}
        
        # Use rich's progress tracker to show upload progress
        for future in console.get_progress_tracker(as_completed(future_to_filepath), description="Uploading...", total=len(filepaths)):
            try:
                filename, success, message = future.result()
                if success:
                    successful_uploads.append(f"{filename} - {message}")
                else:
                    failed_uploads.append((filename, message))
            except Exception as exc:
                failed_uploads.append((future_to_filepath[future], str(exc)))

    console.rule("Upload Summary")
    console.success(f"Successfully submitted/skipped {len(successful_uploads)} files.")
    if failed_uploads:
        console.error(f"Failed to submit {len(failed_uploads)} files:")
        for filename, reason in failed_uploads:
            console.info(f"  - {filename}: {reason}")
    
    # After uploading, start monitoring the server-side processing
    if successful_uploads:
        console.info("Uploads complete. Now monitoring background processing on the server...")
        monitor_progress()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch upload client for MOF-Advisor-API.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "-d", "--directory",
        type=str,
        default=settings.PAPERS_DIR,
        help=f"Directory to scan for .md files. Defaults to '{settings.PAPERS_DIR}' from your config."
    )
    parser.add_argument(
        "-w", "--workers",
        type=int,
        default=5,
        help="Number of parallel upload workers (threads)."
    )
    args = parser.parse_args()
    
    main(directory=args.directory, max_workers=args.workers)