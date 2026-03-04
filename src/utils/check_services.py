import httpx
from src.config import settings
from src.utils.logger import logger


async def check_service(name: str, url: str) -> bool:
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            response = await client.get(f"{url}/health")
            response.raise_for_status()
            logger.info(f"{name} is ready")
            return True
    except Exception as e:
        logger.error(f"{name} is not ready: {e}")
        return False


async def check_all_services() -> None:
    services = {
        "TEI Embedding": settings.EMBEDDING_URL,
        "TEI Reranker": settings.RERANK_URL,
        "LLM": settings.LLM_BASE_URL,
    }

    failed = []
    for name, url in services.items():
        ok = await check_service(name, url)
        if not ok:
            failed.append(name)

    if failed:
        raise RuntimeError(f"Services not ready: {', '.join(failed)}")