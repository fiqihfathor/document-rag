import re
from dataclasses import dataclass
from src.core.pdf_parser import PageData
from src.utils.logger import logger

@dataclass
class ChunkData:
    text: str
    page_number: int
    filename: str
    chunk_index: int


def _split_by_sentences(text: str, chunk_size: int, overlap: int) -> list[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks = []
    current = []
    current_len = 0

    for sentence in sentences:
        words = sentence.split()
        if current_len + len(words) > chunk_size and current:
            chunks.append(" ".join(current))
            overlap_sentences = []
            overlap_len = 0
            for s in reversed(current):
                s_len = len(s.split())
                if overlap_len + s_len <= overlap:
                    overlap_sentences.insert(0, s)
                    overlap_len += s_len
                else:
                    break
            current = overlap_sentences
            current_len = overlap_len

        current.extend(words)
        current_len += len(words)

    if current:
        chunks.append(" ".join(current))

    return chunks


def _split_markdown_sections(text: str) -> list[str]:
    pattern = r"(?=^#{1,3}\s)"
    sections = re.split(pattern, text, flags=re.MULTILINE)
    return [s.strip() for s in sections if s.strip()]


def chunk_pages(
    pages: list[PageData],
    chunk_size: int = 512,
    overlap: int = 64,
) -> list[ChunkData]:
    chunks = []
    global_idx = 0
    logger.info(f"Total Pages: {len(pages)}")
    for page in pages:
        sections = _split_markdown_sections(page.text)

        if not sections:
            sections = [page.text]

        for section in sections:
            words = section.split()

            if len(words) <= chunk_size:
                chunks.append(ChunkData(
                    text=section,
                    page_number=page.page_number,
                    filename=page.filename,
                    chunk_index=global_idx,
                ))
                global_idx += 1
            else:
                sub_chunks = _split_by_sentences(section, chunk_size, overlap)
                for sub in sub_chunks:
                    chunks.append(ChunkData(
                        text=sub,
                        page_number=page.page_number,
                        filename=page.filename,
                        chunk_index=global_idx,
                    ))
                    global_idx += 1
    
    logger.info(f"Total Chunks: {len(chunks)}")
    return chunks
