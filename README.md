# Intelligent Document Analysis Agent

A production-ready **Multi-Agent RAG System** with Computer Vision integration for intelligent document analysis and question answering.

## Features

- **Multi-Agent Orchestration** — Three specialized AI agents (Document Processor, Retrieval Specialist, Analysis Expert) coordinated via CrewAI
- **RAG Pipeline** — Semantic chunking, HuggingFace embeddings, ChromaDB vector search with LangChain
- **Computer Vision** — Table/chart detection with OpenCV, OCR with Tesseract for scanned documents
- **Multi-Modal LLM** — GPT-4o / Claude for understanding text, tables, charts, and images
- **Production REST API** — FastAPI with async processing, validation, structured logging
- **Interactive UI** — Streamlit web interface for upload and querying
- **Docker Ready** — Multi-stage Dockerfile with docker-compose for one-command deployment

## Architecture

```
User → Streamlit UI / API Client
         ↓
    FastAPI Backend (/upload, /query, /health)
         ↓
    CrewAI Orchestration Layer
    ├── Document Processor Agent (pdfplumber, OpenCV, Tesseract)
    ├── Retrieval Agent (HuggingFace embeddings, ChromaDB)
    └── Analysis Agent (GPT-4o / Claude with citations)
         ↓
    Data Layer (ChromaDB vectors, local file storage)
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Backend | FastAPI, Python 3.11 |
| Agents | CrewAI |
| RAG Pipeline | LangChain |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` (384-dim) |
| Vector DB | ChromaDB (local) / Pinecone (cloud) |
| LLM | OpenAI GPT-4o / Anthropic Claude |
| Computer Vision | OpenCV, pytesseract |
| Frontend | Streamlit |
| Deployment | Docker, docker-compose |

## Quick Start

### Prerequisites

- Python 3.10+
- Tesseract OCR installed (`apt install tesseract-ocr` or `brew install tesseract`)
- OpenAI or Anthropic API key

### Local Setup

```bash
# Clone the repository
git clone <repo-url>
cd intelligent-doc-agent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys

# Start the API
python main.py

# In a separate terminal, start the UI
streamlit run app.py
```

### Docker Setup

```bash
# Build and run
docker-compose -f docker/docker-compose.yml up --build -d

# Access
# API:       http://localhost:8000
# API Docs:  http://localhost:8000/docs
# UI:        http://localhost:8501
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/upload` | Upload a document (PDF/JPG/PNG/WEBP) |
| `POST` | `/api/v1/query` | Ask a question about documents |
| `GET` | `/api/v1/documents` | List all documents |
| `GET` | `/api/v1/documents/{id}` | Get document status |
| `GET` | `/api/v1/health` | System health check |

### Example Usage

```bash
# Upload a document
curl -X POST http://localhost:8000/api/v1/upload \
  -F "file=@report.pdf"

# Ask a question
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What was the Q3 revenue?", "top_k": 5}'

# Check health
curl http://localhost:8000/api/v1/health
```

## Project Structure

```
├── agents/              # CrewAI agents and tools
│   ├── config/          # Agent/task YAML definitions
│   ├── tools/           # PDF extractor, OCR, OpenCV, vector search
│   └── orchestrator.py  # Crew assembly and execution
├── api/                 # FastAPI application
│   ├── routes/          # Endpoint handlers
│   ├── schemas/         # Pydantic request/response models
│   └── middleware.py    # Logging, request tracking
├── services/            # Business logic layer
│   ├── document_service.py    # File handling, content extraction
│   ├── embedding_service.py   # Chunking, HuggingFace embeddings
│   ├── vector_db_service.py   # ChromaDB/Pinecone abstraction
│   └── llm_service.py         # OpenAI/Anthropic provider
├── core/                # Configuration, logging, prompts
├── tests/               # Unit and integration tests
├── docker/              # Dockerfile and compose files
├── app.py               # Streamlit UI
└── main.py              # Entry point
```

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=services --cov=api --cov-report=term-missing

# Run specific test file
pytest tests/unit/test_embedding_service.py -v
```

## Configuration

All configuration is via environment variables (`.env` file):

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `openai` | LLM provider (`openai` or `anthropic`) |
| `LLM_MODEL` | `gpt-4o` | Model to use for analysis |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | HuggingFace embedding model |
| `VECTOR_DB_TYPE` | `chromadb` | Vector DB (`chromadb` or `pinecone`) |
| `CHUNK_SIZE` | `750` | Token count per text chunk |
| `CHUNK_OVERLAP` | `150` | Overlap between chunks |
| `MAX_FILE_SIZE_MB` | `50` | Max upload file size |

## Author

**M Karthikeya Reddy** — Data Scientist / AI-ML Engineer

## License

MIT
