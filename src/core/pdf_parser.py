import pymupdf4llm
import tempfile
import os
from src.utils.logger import logger
from dataclasses import dataclass


@dataclass
class PageData:
    filename:str
    total_pages: int
    page_number: int
    text: str


def parse_pdf(file_bytes: bytes, filename: str)->list[PageData]:
    
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(file_bytes)
        temp_path = tmp.name
        
    try:
        md = pymupdf4llm.to_markdown(temp_path, page_chunks=True)
        pages= []
        for page in md:
            text = page["text"].strip()
            if not text:
                continue
            pages.append(PageData(
                filename=filename,
                total_pages=page["metadata"]["page_count"],
                page_number=page["metadata"]["page"],
                text=text
            ))
        return pages
    except Exception as e:
        logger.error(f"Failed to parse PDFv {filename}: {e}")
        raise
    finally:
        os.unlink(temp_path)
