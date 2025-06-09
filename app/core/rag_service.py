# The module provides a service for retrieving synthesis methods for
# Metal-Organic Frameworks (MOFs) using a Retrieval-Augmented Generation (RAG) approach.
# Author: shiboli
# Date: 2025-06-09
# Version: 1.3.0

import json
import os
import chromadb
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from app.core.logger import console
from dotenv import load_dotenv
from typing import List, Dict, Any

# Load environment variables from .env file
load_dotenv()

# --- Constants ---
# All configurations are now loaded from environment variables for flexibility.
PAPERS_DIR = os.getenv("PAPERS_DIR")
DB_PATH = os.getenv("DB_PATH")
LLM_MODEL = os.getenv("DEEPSEEK_MODEL")
LLM_BASE_URL = os.getenv("DEEPSEEK_BASE_URL") # Use None as default if not set
OPENAI_API_KEY = os.getenv("DEEPSEEK_API_KEY") # Explicitly load the API key
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")


class RAGService:
    """
    Encapsulates the entire RAG pipeline from query to generation.
    """
    def __init__(self):
        """
        Initializes the service by loading models and connecting to the database.
        """
        try:
            console.info("Initializing RAGService...")

            # --- CORE FIX: Explicitly pass api_key and base_url to the client ---
            if not OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY environment variable not set.")

            self.llm_client = OpenAI(
                api_key=OPENAI_API_KEY,
                base_url=LLM_BASE_URL
            )
            console.info(f"LLM client configured for model '{LLM_MODEL}' at '{self.llm_client.base_url}'")
            # --------------------------------------------------------------------

            # Initialize local embedding model
            console.info(f"Loading local embedding model: '{EMBEDDING_MODEL}'...")
            self.embedding_model = SentenceTransformer(EMBEDDING_MODEL, device='cpu')

            # Initialize ChromaDB client and get the collection
            console.info(f"Connecting to vector database at: '{DB_PATH}'")
            self.db_client = chromadb.PersistentClient(path=DB_PATH)
            self.collection = self.db_client.get_or_create_collection(name=COLLECTION_NAME)
            console.info(f"Successfully connected to collection: '{COLLECTION_NAME}'")

            console.success("RAGService initialized successfully.")
        except Exception as e:
            console.exception("Failed to initialize RAGService.")
            raise e

    def _create_llm_prompt(self, query: str, contexts: List[str]) -> tuple[str, str]:
        """Creates the system and user prompts for the LLM."""
        context_str = "\n\n---\n\n".join(contexts)
        
        system_prompt = """
        You are a world-class chemist specializing in MOF synthesis.
        Your task is to devise a reasonable synthesis protocol based on the user's request and relevant literature excerpts provided as context.
        You MUST return the protocol as a single, valid JSON object. Do not add any explanation before or after the JSON.

        The JSON object should have the following structure:
        {
          "suggested_protocol": {
            "metal_source_suggestion": "e.g., Copper(II) nitrate trihydrate, Cu(NO3)2·3H2O",
            "linker_suggestion": "e.g., 1,3,5-Benzenetricarboxylic acid (H3BTC)",
            "solvent_suggestion": "e.g., A mixture of DMF/Ethanol/Water in a 1:1:1 ratio",
            "temperature_celsius": "e.g., 110",
            "time_hours": "e.g., 24",
            "procedure_details": "A step-by-step description of the synthesis procedure.",
            "reasoning": "A brief explanation of why this protocol was chosen based on the provided context."
          }
        }
        """
        
        user_prompt = f"""
        Here is the user's request: {query}

        Here is the relevant context from my knowledge base:
        ---
        {context_str}
        ---

        Please provide the synthesis protocol as a JSON object based on this information.
        """
        return system_prompt, user_prompt

    def query_synthesis_method(self, metal: str, organic_linker: str) -> Dict[str, Any]:
        """
        The main method to perform a RAG query for a synthesis method.

        Returns:
            A dictionary containing the suggestion and sources.
        """
        console.info(f"Received query: Metal='{metal}', Linker='{organic_linker}'")
        
        query_text = f"A synthesis method for a MOF with metal site {metal} and organic linker {organic_linker}"
        query_embedding = self.embedding_model.encode(query_text, normalize_embeddings=True).tolist()
        
        console.info("Retrieving relevant documents from knowledge base...")
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=5
        )
        
        documents = results.get('documents', [[]])[0]
        metadatas = results.get('metadatas', [[]])[0]

        if not documents:
            console.warning(f"No relevant documents found for query: '{query_text}'")
            raise ValueError("Could not find relevant synthesis information in the knowledge base.")
        
        console.success(f"Retrieved {len(documents)} relevant document(s).")
        
        console.info(f"Generating synthesis suggestion with LLM ('{LLM_MODEL}')...")
        system_prompt, user_prompt = self._create_llm_prompt(query=query_text, contexts=documents)
        
        response = self.llm_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"}
        )
        
        suggestion_json = json.loads(response.choices[0].message.content)
        console.success("LLM suggestion generated successfully.")
        
        return {
            "suggestion": suggestion_json.get("suggested_protocol", {}),
            "sources": metadatas
        }

# A single instance to be used across the application (singleton-like).
rag_service = RAGService()