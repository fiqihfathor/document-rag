import pytest
from pathlib import Path
from src.core.pdf_parser import parse_pdf


SAMPLE_PDF = Path("tests/fixtures/SAMPLE_CV.pdf")

@pytest.mark.integration
@pytest.mark.skipif(not SAMPLE_PDF.exists(), reason="no fixture pdf")
def test_parse_real_pdf():
    pages = parse_pdf(SAMPLE_PDF.read_bytes(), "SAMPLE_CV.pdf")

    print(pages)
    assert len(pages) > 0
    assert all(p.text for p in pages)
    assert all(p.page_number >= 1 for p in pages)
    assert pages[0].filename == "SAMPLE_CV.pdf"