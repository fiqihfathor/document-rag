from fastapi import APIRouter, UploadFile, File, HTTPException
from src.core.pipeline import ingest_pipeline
from src.core import vector_store

router = APIRouter()


@router.post("/ingest")
async def ingest_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    file_bytes = await file.read()
    try:
        result = await ingest_pipeline(file_bytes, file.filename)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    return result


@router.delete("/ingest/{filename}")
async def delete_document(filename: str):
    await vector_store.delete_by_filename(filename)
    return {"deleted": filename}