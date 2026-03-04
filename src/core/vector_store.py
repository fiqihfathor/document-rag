from qdrant_client import AsyncQdrantClient
from qdrant_client.models import PointStruct, Prefetch, SparseVector, FusionQuery, Fusion, VectorParams, Distance, SparseVectorParams, Filter, FieldCondition, MatchValue
from src.config import settings
from src.utils.logger import logger
from src.core.sparse import build_sparse_vector
import uuid

client = AsyncQdrantClient(url=settings.QDRANT_URL)
DENSE_VECTOR = "dense"
SPARSE_VECTOR = "sparse"

async def upsert_chunks(chunks: list[dict]):
    logger.info(f"Try to upsert : {len(chunks)}")
    points = []
    for chunk in chunks:
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector={
                    DENSE_VECTOR: chunk["embedding"],
                    SPARSE_VECTOR: SparseVector(
                        indices=chunk["sparse"]["indices"],
                        values=chunk["sparse"]["values"],
                    ),
                },
                payload={
                    "text": chunk["text"],
                    "filename": chunk["filename"],
                    "page_number": chunk["page_number"],
                    "chunk_index": chunk["chunk_index"],
                }
            )
        )

    await client.upsert(
        collection_name=settings.COLLECTION_NAME,
        points=points,
    )
    logger.info(f"Upserted {len(points)} chunks")
    
async def search(query_text: str, query_vector: list[float], top_k: int) -> list[dict]:
    sparse = build_sparse_vector(query_text)
    
    results = await client.query_points(
        collection_name=settings.COLLECTION_NAME,
        prefetch=[
            Prefetch(
                query=query_vector,
                using=DENSE_VECTOR,
                limit=top_k*2,
            ),
            Prefetch(
                query=SparseVector(
                    indices=sparse["indices"],
                    values=sparse["values"]
                ),
                using=SPARSE_VECTOR,
                limit=top_k*2
            )
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=top_k
    )
    
    return [
        {
            "text": data.payload["text"],
            "filename": data.payload["filename"],
            "page_number": data.payload["page_number"],
            "chunk_index": data.payload["chunk_index"],
            "score": data.score,
        }
        for data in results.points 
    ]
    
async def ensure_collection(vector_size: int) -> None:
    existing = await client.collection_exists(settings.COLLECTION_NAME)
    if not existing:
        await client.create_collection(
            collection_name=settings.COLLECTION_NAME,
            vectors_config={
                DENSE_VECTOR: VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE,
                )
            },
            sparse_vectors_config={
                SPARSE_VECTOR: SparseVectorParams()
            },
        )
        logger.info(f"Created collection: {settings.COLLECTION_NAME}")
    else:
        logger.debug(f"Collection already exists: {settings.COLLECTION_NAME}")
        
async def delete_by_filename(filename: str) -> None:
    await client.delete(
        collection_name=settings.COLLECTION_NAME,
        points_selector=Filter(
            must=[
                FieldCondition(
                    key="filename",
                    match=MatchValue(value=filename),
                )
            ]
        ),
    )
    logger.info(f"Deleted chunks for: {filename}")