from fastapi import FastAPI
from contextlib import asynccontextmanager
from src.api.routes import ingest, query
from src.utils.logger import logger
from src.utils.check_services import check_all_services
from src.core import vector_store


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("RAG Service starting up")
    await vector_store.client.get_collections()
    await check_all_services()
    logger.info("Qdrant connected")
    yield
    await vector_store.client.close()
    logger.info("RAG Service shutting down")


app = FastAPI(
    title="RAG Service",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(ingest.router, prefix="/api/v1", tags=["ingest"])
app.include_router(query.router, prefix="/api/v1", tags=["query"])

@app.get("/health", tags=["health"])
async def health():
    return {"status": "ok"}