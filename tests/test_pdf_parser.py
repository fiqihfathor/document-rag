from unittest.mock import patch
from src.core.pdf_parser import parse_pdf, PageData


MOCK_PAGES = [
    {
        "text": "Halaman 1",
        "metadata": {"page": 1, "page_count":2},
    },
    {
        "text": "Halaman 2",
        "metadata": {"page": 2,"page_count":2},
    },
]


@patch("src.core.pdf_parser.pymupdf4llm.to_markdown", return_value=MOCK_PAGES)
@patch("src.core.pdf_parser.os.unlink")
def test_parse_pdf_returns_non_empty_pages(mock_unlink, mock_to_markdown):
    pages = parse_pdf(b"%PDF-Test", "test.pdf")
    assert all(isinstance(p, PageData) for p in pages)


@patch("src.core.pdf_parser.pymupdf4llm.to_markdown", return_value=MOCK_PAGES)
@patch("src.core.pdf_parser.os.unlink")
def test_parse_pdf_metadata(mock_unlink, mock_to_markdown):
    pages = parse_pdf(b"%PDF-Test", "test.pdf")

    assert pages[0].filename == "test.pdf"
    assert pages[0].page_number == 1
    assert pages[0].total_pages == 2
    assert pages[1].page_number == 2

