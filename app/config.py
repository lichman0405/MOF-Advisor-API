# The module for application configuration, including LLM provider settings.
# Author: Shiboli
# Date: 2025-06-09
# Version: 0.1.0


from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, BaseModel

class ProviderConfig(BaseModel):
    """Represents the configuration for a single LLM provider."""
    api_key: str
    model: str
    base_url: str | None = None


class Settings(BaseSettings):
    """
    Defines the application's settings.
    It loads all variables from the .env file as top-level attributes
    and then manually constructs the active provider's config.
    """
    model_config = SettingsConfigDict(
        env_file='.env', 
        env_file_encoding='utf-8', 
        extra='ignore', 
        case_sensitive=False
    )

    # --- Master Switch ---
    LLM_PROVIDER: str = Field(default="DEEPSEEK_CHAT", description="The active LLM provider to use.")

    # --- Load ALL provider configurations as top-level fields ---
    # This is the most robust way to ensure pydantic-settings reads them.
    CHATGPT_API_KEY: str = "placeholder"
    CHATGPT_MODEL: str = "placeholder"
    CHATGPT_BASE_URL: str | None = None

    CLAUDE_API_KEY: str = "placeholder"
    CLAUDE_MODEL: str = "placeholder"
    CLAUDE_BASE_URL: str | None = None

    GEMINI_API_KEY: str = "placeholder"
    GEMINI_MODEL: str = "placeholder"
    GEMINI_BASE_URL: str | None = None

    DEEPSEEK_CHAT_API_KEY: str = "placeholder"
    DEEPSEEK_CHAT_MODEL: str = "placeholder"
    DEEPSEEK_CHAT_BASE_URL: str | None = None

    DEEPSEEK_REASONER_API_KEY: str = "placeholder"
    DEEPSEEK_REASONER_MODEL: str = "placeholder"
    DEEPSEEK_REASONER_BASE_URL: str | None = None
    
    # --- Other Application Settings ---
    PAPERS_DIR: str = "/data/papers"
    DB_PATH: str = "/chroma_db"
    EMBEDDING_MODEL: str = "BAAI/bge-m3"
    COLLECTION_NAME: str = "mof_synthesis_papers"

    @property
    def active_llm_config(self) -> ProviderConfig:
        """
        Manually and dynamically constructs the ProviderConfig object
        for the active LLM provider based on the loaded top-level settings.
        """
        provider = self.LLM_PROVIDER.upper()
        
        # Use getattr to dynamically get the attributes from self
        # For example, if provider is "CHATGPT", it gets self.CHATGPT_API_KEY
        api_key = getattr(self, f"{provider}_API_KEY", None)
        model = getattr(self, f"{provider}_MODEL", None)
        base_url = getattr(self, f"{provider}_BASE_URL", None)

        if not api_key or not model:
            raise ValueError(f"API Key or Model for provider '{provider}' is not configured in your .env file.")

        return ProviderConfig(
            api_key=api_key,
            model=model,
            base_url=base_url
        )

# Create the single, globally accessible instance of the settings.
settings = Settings()


# Optional test block to verify configuration loading
if __name__ == '__main__':
    from app.core.logger import console
    
    # Test the loading of ALL variables
    console.rule("All Top-Level Configurations Loaded")
    # We use model_dump to convert the pydantic model to a dict for display
    console.display_data_as_table(settings.model_dump(), "All Loaded Settings")
    
    # Test if the active configuration is constructed correctly
    console.rule("Active LLM Provider Configuration")
    try:
        active_config = settings.active_llm_config
        console.display_data_as_table(
            active_config.model_dump(),
            f"Active Provider: {settings.LLM_PROVIDER}"
        )
    except ValueError as e:
        console.error(str(e))