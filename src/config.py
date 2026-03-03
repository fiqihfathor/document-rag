from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    LLM_MODEL: str = Field(default="AMead10/Llama-3.2-3B-Instruct-AWQ")
    LLM_BASE_URL: str = Field(default="http://localhost:8000/v1")
    LLM_API_KEY: str = Field(default="none")
    
    QDRANT_URL: str = Field(default="http://localhost:6333")
    COLLECTION_NAME: str = Field(default="rag_documents")
    
    EMBEDDING_URL: str = Field(default="http://localhost:8001/embed")
    
    RERANK_MODEL: str = Field(default="mixedbread-ai/mxbai-rerank-xsmall-v1")
    
    API_TIMEOUT: int = Field(default=30)
    
    config = SettingsConfigDict(env_file=".env")
    
settings = Settings()