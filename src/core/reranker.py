import httpx
from src.utils.logger import logger
from src.config import settings

async def rerank(query: str, chunks: list[dict], top_k: int) -> list[dict]:
    texts = [chunk["text"] for chunk in chunks]
    
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            f"{settings.RERANK_URL}/rerank",
            json={
                "query": query,
                "texts": texts,
                "return_text":False
            }
        )
        response.raise_for_status()
        
    results = response.json()
    ranked = sorted(results, key=lambda x: x["score"], reverse=True)
    
    reranked_chunks = [
        {**chunks[r["index"]], "rerank_score": r["score"]}
        for r in ranked[:top_k]
    ]
    logger.debug(f"Reranked {len(chunks)} → top {top_k}")
    return reranked_chunks