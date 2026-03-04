import httpx 
from src.utils.logger import logger
from src.config import settings


TIMEOUT= settings.API_TIMEOUT
EMBEDDING_URL = settings.EMBEDDING_URL

async def embed_texts(texts: list[str]) -> list[list[float]]:
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            response = await client.post(
                f"{settings.EMBEDDING_URL}/embed",
                json={"inputs": texts},
            )
            response.raise_for_status()

        vectors = response.json()
        return vectors
    except Exception as e:
        logger.error(f"Error embbed texts : {e}")

async def embed_query(query: str) -> list[float]:
    vectors = await embed_texts([query])
    return vectors[0]