from src.core.chunker import chunk_pages, ChunkData
from src.core.pdf_parser import PageData


def make_page(text: str, page_number: int = 1) -> PageData:
    return PageData(
        text=text,
        page_number=page_number,
        total_pages=1,
        filename="test.pdf",
    )


def test_returns_chunk_data():
    page = make_page("Kalimat pertama. Kalimat kedua.")
    chunks = chunk_pages([page])
    assert all(isinstance(c, ChunkData) for c in chunks)

def test_short_text_single_chunk():
    page = make_page("Teks pendek.")
    chunks = chunk_pages([page])
    assert len(chunks) == 1

def test_empty_pages_returns_empty():
    chunks = chunk_pages([])
    assert chunks == []

def test_split_by_heading():
    text = "# Bab 1\nIsi bab satu.\n\n# Bab 2\nIsi bab dua."
    page = make_page(text)
    chunks = chunk_pages([page])
    assert len(chunks) == 2

def test_fallback_chunks_not_empty():
    long_text = "Kalimat satu. Kalimat dua. Kalimat tiga. " * 200
    page = make_page(long_text)
    chunks = chunk_pages([page], chunk_size=100, overlap=10)
    assert all(c.text.strip() for c in chunks)

