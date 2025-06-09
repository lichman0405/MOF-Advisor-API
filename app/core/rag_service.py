# The module provides the core RAG service for the application.
# Author: shiboli & Gemini
# Date: 2025-06-09
# Version: 0.1.0

# The module provides the core RAG service for the application,
# featuring a pre-validation layer for chemical feasibility.
# Author: shiboli & Gemini
# Date: 2025-06-09
# Version: 2.2.0 (With Pre-validation Layer)

import json
from openai import OpenAI
import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Tuple

# Import our custom console manager and centralized settings
from app.core.logger import console
from app.config import settings

class RAGService:
    """
    Encapsulates the entire RAG pipeline, with an intelligent fallback
    to pure LLM generation when no relevant context is found.
    Includes a pre-validation step to check for chemical feasibility.
    """
    def __init__(self):
        """
        Initializes the service by loading models and connecting to the database.
        """
        try:
            console.info("Initializing RAGService...")
            
            active_llm = settings.active_llm_config
            
            self.llm_client = OpenAI(
                api_key=active_llm.api_key,
                base_url=active_llm.base_url
            )
            self.embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL, device='cpu')
            self.db_client = chromadb.PersistentClient(path=settings.DB_PATH)
            self.collection = self.db_client.get_or_create_collection(name=settings.COLLECTION_NAME)
            
            console.info(f"Active LLM Provider: {settings.LLM_PROVIDER}")
            console.info(f"Embedding Model: {settings.EMBEDDING_MODEL}")
            console.info(f"Database Path: {settings.DB_PATH}")
            console.info(f"Collection Name: {settings.COLLECTION_NAME}")
            console.success("RAGService initialized successfully.")
        except Exception as e:
            console.exception("Failed to initialize RAGService.")
            raise e

    def _check_feasibility(self, metal: str, linker: str) -> Tuple[bool, str]:
        """
        Gate 1: Performs a quick check with the LLM to see if the request is chemically plausible.
        """
        console.info(f"Performing feasibility check for Metal='{metal}', Linker='{linker}'...")
        
        system_prompt = """You are an expert chemist. Your task is to briefly determine if it is chemically plausible to synthesize a Metal-Organic Framework (MOF) using the given components. Respond with a single JSON object with two keys: {"is_plausible": boolean, "reasoning": "your brief explanation"}. For example, for Sodium and Methane, respond with {"is_plausible": false, "reasoning": "Methane lacks the necessary functional groups to act as a linker."}. For Copper and BTC, respond with {"is_plausible": true, "reasoning": "This is a well-known and classic MOF synthesis (HKUST-1)."}. """
        user_prompt = f"Metal: '{metal}', Organic Linker: '{linker}'"
        
        active_llm_config = settings.active_llm_config
        
        try:
            response = self.llm_client.chat.completions.create(
                model=active_llm_config.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.0 # Use low temperature for deterministic checking
            )
            result = json.loads(response.choices[0].message.content)
            is_plausible = result.get("is_plausible", False)
            reasoning = result.get("reasoning", "No reasoning provided.")
            
            if is_plausible:
                console.success(f"Feasibility check passed. Reason: {reasoning}")
            else:
                console.warning(f"Feasibility check failed. Reason: {reasoning}")
                
            return is_plausible, reasoning
        except Exception:
            console.exception("An error occurred during feasibility check. Assuming plausible to proceed.")
            # As a fallback, assume it's plausible to not block valid requests due to checker failure.
            return True, "Feasibility check failed, proceeding with caution."

    def _create_rag_prompt(self, query: str, contexts: List[str]) -> tuple[str, str]:
        """Creates the prompt for the RAG-based generation."""
        context_str = "\n\n---\n\n".join(contexts)
        system_prompt = """You are a world-class chemist specializing in MOF synthesis. Your task is to devise a reasonable synthesis protocol based on the user's request and the following relevant literature excerpts provided as context. You MUST return the protocol as a single, valid JSON object. Do not add any explanation before or after the JSON. The JSON object should have the following structure: {"suggested_protocol": {"metal_source_suggestion": "e.g., Copper(II) nitrate trihydrate, Cu(NO3)2·3H2O", "linker_suggestion": "e.g., 1,3,5-Benzenetricarboxylic acid (H3BTC)", "solvent_suggestion": "e.g., A mixture of DMF/Ethanol/Water in a 1:1:1 ratio", "temperature_celsius": "e.g., 110", "time_hours": "e.g., 24", "procedure_details": "A step-by-step description of the synthesis procedure.", "reasoning": "A brief explanation of why this protocol was chosen based on the provided context."}}"""
        user_prompt = f"""Here is the user's request: {query}\n\nHere is the relevant context from my knowledge base:\n---\n{context_str}\n---\nPlease provide the synthesis protocol as a JSON object based on this information."""
        return system_prompt, user_prompt

    def _create_fallback_prompt(self, query: str) -> tuple[str, str]:
        """Creates the prompt for the LLM-only fallback generation."""
        system_prompt = """You are a world-class chemist specializing in MOF synthesis. Your task is to devise a reasonable synthesis protocol based *only on your own general knowledge*, as no specific literature was found. You MUST state in the 'reasoning' field that this suggestion is based on general chemical principles, not specific provided literature. You MUST return the protocol as a single, valid JSON object. Do not add any explanation before or after the JSON. The JSON object should have the following structure: {"suggested_protocol": {"metal_source_suggestion": "e.g., Copper(II) nitrate trihydrate, Cu(NO3)2·3H2O", "linker_suggestion": "e.g., 1,3,5-Benzenetricarboxylic acid (H3BTC)", "solvent_suggestion": "e.g., A mixture of DMF/Ethanol/Water in a 1:1:1 ratio", "temperature_celsius": "e.g., 110", "time_hours": "e.g., 24", "procedure_details": "A step-by-step description of the synthesis procedure.", "reasoning": "Start with a statement that this protocol is generated from general knowledge as no specific examples were found."}}"""
        user_prompt = f"""My local knowledge base found no relevant examples for the user's request: {query}.\nPlease generate a plausible synthesis protocol from scratch based on your general expertise."""
        return system_prompt, user_prompt

    def _generate_synthesis_protocol(self, metal: str, linker: str) -> Dict[str, Any]:
        """Gate 2: Performs the RAG or LLM-fallback process to generate a synthesis protocol."""
        query_text = f"A synthesis method for a MOF with metal site {metal} and organic linker {linker}"
        query_embedding = self.embedding_model.encode(query_text, normalize_embeddings=True).tolist()
        
        console.info("Retrieving documents from knowledge base...")
        results = self.collection.query(query_embeddings=[query_embedding], n_results=3)
        
        documents, metadatas = results.get('documents', [[]])[0], results.get('metadatas', [[]])[0]
        
        active_llm_config = settings.active_llm_config

        if documents:
            console.success(f"Retrieved {len(documents)} relevant document(s). Proceeding with RAG.")
            generation_mode, sources = "RAG_BASED", metadatas
            system_prompt, user_prompt = self._create_rag_prompt(query_text, documents)
        else:
            console.warning("No relevant documents found. Switching to LLM fallback mode.")
            generation_mode, sources = "LLM_FALLBACK", []
            system_prompt, user_prompt = self._create_fallback_prompt(query_text)
        
        console.info(f"Generating synthesis suggestion with LLM: '{settings.LLM_PROVIDER}'...")
        response = self.llm_client.chat.completions.create(
            model=active_llm_config.model,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            response_format={"type": "json_object"}
        )
        
        suggestion_json = json.loads(response.choices[0].message.content)
        console.success(f"Suggestion generated successfully in '{generation_mode}' mode.")
        
        return {
            "generation_mode": generation_mode,
            "suggestion": suggestion_json.get("suggested_protocol", {}),
            "sources": sources
        }

    def query_synthesis_method(self, metal: str, organic_linker: str) -> Dict[str, Any]:
        """
        The main orchestrator method.
        It first checks for feasibility and then generates the protocol.
        """
        # --- Gate 1: Feasibility Check ---
        is_plausible, reasoning = self._check_feasibility(metal, organic_linker)
        
        if not is_plausible:
            # If not plausible, we stop here and raise a specific error with the reasoning.
            raise ValueError(f"The request is considered chemically implausible. Reason: {reasoning}")
        
        # --- Gate 2: Protocol Generation ---
        return self._generate_synthesis_protocol(metal, organic_linker)

# A single instance to be used across the application (singleton-like).
# This line runs when the module is first imported by the uvicorn server.
rag_service = RAGService()