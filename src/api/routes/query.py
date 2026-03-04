from fastapi import APIRouter
from pydantic import BaseModel
from src.core.pipeline import query_pipeline

router = APIRouter()


class QueryRequest(BaseModel):
    query: str


@router.post("/query")
async def query(request: QueryRequest):
    result = await query_pipeline(request.query)
    return result