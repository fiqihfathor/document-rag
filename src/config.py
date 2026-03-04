from pydantic_settings import BaseSettings
from pydantic import Field
from pydantic import ConfigDict


class Settings(BaseSettings):
    LLM_MODEL: str = Field(default="AMead10/Llama-3.2-3B-Instruct-AWQ")
    LLM_BASE_URL: str = Field(default="http://localhost:8000")
    LLM_API_KEY: str = Field(default="none")
    
    QDRANT_URL: str = Field(default="http://localhost:6333")
    COLLECTION_NAME: str = Field(default="rag_documents")
    
    EMBEDDING_URL: str = Field(default="http://localhost:8001")
    
    RERANK_URL: str = Field(default="http://localhost:8002")
    
    API_TIMEOUT: int = Field(default=30)
    
    CHUNK_SIZE: int = Field(default=512)
    
    CHUNK_OVERLAP: int = Field(default=10)
    
    TOP_K: int = Field(default=5)
    
    model_config = ConfigDict(env_file=".env")
    
settings = Settings()