from src.core.pdf_parser import parse_pdf
from src.core.chunker import chunk_pages
from src.core.embedder import embed_texts, embed_query
from src.core.sparse import build_sparse_vector
from src.core.reranker import rerank
from src.core import vector_store
from src.core.llm_client import generate
from src.config import settings
from src.utils.logger import logger
from dataclasses import dataclass

@dataclass
class GenerateResult:
    answer: str
    prompt_used: str
    prompt_tokens: int
    completion_tokens: int
    sources: list[dict]
    
async def ingest_pipeline(file_bytes: bytes, filename: str) -> dict:
    pages = parse_pdf(file_bytes, filename)
    if not pages:
        raise ValueError("No text extracted from PDF")

    chunks = chunk_pages(pages, chunk_size=settings.CHUNK_SIZE, overlap=settings.CHUNK_OVERLAP)
    texts = [c.text for c in chunks]

    embeddings = await embed_texts(texts)
    sparse_vectors = [build_sparse_vector(t) for t in texts]

    await vector_store.ensure_collection(vector_size=len(embeddings[0]))
    await vector_store.upsert_chunks([
        {
            "text": c.text,
            "filename": c.filename,
            "page_number": c.page_number,
            "chunk_index": c.chunk_index,
            "embedding": emb,
            "sparse": sparse,
        }
        for c, emb, sparse in zip(chunks, embeddings, sparse_vectors)
    ])

    logger.info(f"Ingest done: {filename} — {len(chunks)} chunks")
    return {
        "filename": filename,
        "pages": len(pages),
        "chunks": len(chunks),
    }


async def query_pipeline(query: str) -> GenerateResult:
    query_vector = await embed_query(query)

    results = await vector_store.search(
        query_text=query,
        query_vector=query_vector,
        top_k=settings.TOP_K,
    )

    reranked = await rerank(query, results, top_k=settings.TOP_K)
    reranked = [c for c in reranked if c["rerank_score"] > 0.01]

    if not reranked:
        return GenerateResult(
            answer="The information was not found in the provided documents",
            prompt_used="No prompt used to llm",
            prompt_tokens=0,
            completion_tokens=0,
            sources=[]   
        )
            

    output = await generate(query, reranked)

    logger.info(f"Query done: '{query[:50]}'")
    
    return GenerateResult(
        answer=output["answer"],
        prompt_used=output["prompt_used"],
        prompt_tokens=output["token_usage"]["prompt_tokens"],
        completion_tokens=output["token_usage"]["completion_tokens"],
        sources=[
            {
                "filename": c["filename"],
                "page_number": c["page_number"],
                "text": c["text"],
                "score": c["rerank_score"],
            }
            for c in reranked
        ],
    )