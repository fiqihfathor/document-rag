.
├── Dockerfile
├── README.md
├── data
│   └── SAMPLE_CV.pdf
├── docker-compose.yml
├── docs
│   └── structure.md
├── flowchart TD.mmd
├── logs
├── main.py
├── notebook
│   └── 1-research.ipynb
├── pyproject.toml
├── src
│   ├── __init__.py
│   ├── __pycache__
│   ├── api
│   │   └── routes
│   ├── config.py
│   ├── core
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   ├── chunker.py
│   │   ├── embedder.py
│   │   ├── llm_client.py
│   │   ├── pdf_parser.py
│   │   ├── pipeline.py
│   │   ├── reranker.py
│   │   ├── sparse.py
│   │   └── vector_store.py
│   ├── main.py
│   └── utils
│       ├── __init__.py
│       ├── __pycache__
│       ├── check_services.py
│       └── logger.py
├── tests
│   ├── __pycache__
│   ├── fixtures
│   │   └── SAMPLE_CV.pdf
│   ├── integration
│   │   ├── __pycache__
│   │   └── test_pdf_parser_integration.py
│   ├── test_chuncker.py
│   └── test_pdf_parser.py
└── uv.lock

18 directories, 29 files
