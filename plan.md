# Integration & Implementation Plan

## Multi-Agent RAG System with Computer Vision Integration

**Project:** Intelligent Document Analysis Agent with Visual Understanding
**Author:** M Karthikeya Reddy
**Plan Version:** 1.0
**Date:** February 10, 2026

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Project Structure & Setup](#2-project-structure--setup)
3. [AI Engineering Foundation](#3-ai-engineering-foundation)
4. [LangChain RAG Pipeline](#4-langchain-rag-pipeline)
5. [HuggingFace Model Selection & Configuration](#5-huggingface-model-selection--configuration)
6. [Vector Database Layer](#6-vector-database-layer)
7. [Vector Index Tuning](#7-vector-index-tuning)
8. [CrewAI Multi-Agent Orchestration](#8-crewai-multi-agent-orchestration)
9. [API Design & FastAPI Implementation](#9-api-design--fastapi-implementation)
10. [Docker Deployment Strategy](#10-docker-deployment-strategy)
11. [Integration Flow & Data Pipeline](#11-integration-flow--data-pipeline)
12. [Testing & Validation Strategy](#12-testing--validation-strategy)
13. [Production Hardening Checklist](#13-production-hardening-checklist)

---

## 1. Architecture Overview

### 1.1 High-Level System Design

```
┌──────────────────────────────────────────────────────────────────────┐
│                        CLIENT LAYER                                   │
│   ┌─────────────────┐          ┌──────────────────────────┐          │
│   │  Streamlit UI    │          │  REST API Clients         │          │
│   │  (port 8501)     │────────▶│  (curl / Postman / SDK)   │          │
│   └─────────────────┘          └──────────────────────────┘          │
└───────────────────────────────────┬──────────────────────────────────┘
                                    │ HTTP
                                    ▼
┌──────────────────────────────────────────────────────────────────────┐
│                     API GATEWAY LAYER                                  │
│                    FastAPI (port 8000)                                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ /upload   │  │ /query   │  │ /documents/  │  │  /health     │     │
│  │ (POST)    │  │ (POST)   │  │   {id} (GET) │  │  (GET)       │     │
│  └──────────┘  └──────────┘  └──────────────┘  └──────────────┘     │
│         │              │                                               │
│         │    Request Validation / Rate Limiting / CORS                 │
└─────────┼──────────────┼─────────────────────────────────────────────┘
          │              │
          ▼              ▼
┌──────────────────────────────────────────────────────────────────────┐
│                 ORCHESTRATION LAYER (CrewAI)                           │
│                                                                        │
│  ┌────────────────────────────────────────────────────────────┐       │
│  │              Crew: DocumentAnalysisCrew                     │       │
│  │              Process: Sequential                            │       │
│  │              Memory: Enabled                                │       │
│  │                                                              │       │
│  │  ┌──────────────────┐                                       │       │
│  │  │ Agent 1:          │  Tools: pdf_extractor, ocr_tool,     │       │
│  │  │ Document          │         opencv_table_detector,       │       │
│  │  │ Processor         │         image_preprocessor           │       │
│  │  └────────┬─────────┘                                       │       │
│  │           │ Structured Content (text + images + tables)      │       │
│  │           ▼                                                  │       │
│  │  ┌──────────────────┐                                       │       │
│  │  │ Agent 2:          │  Tools: embedding_generator,         │       │
│  │  │ Retrieval         │         vector_search,               │       │
│  │  │ Specialist        │         metadata_filter              │       │
│  │  └────────┬─────────┘                                       │       │
│  │           │ Top-K Relevant Chunks + Metadata                 │       │
│  │           ▼                                                  │       │
│  │  ┌──────────────────┐                                       │       │
│  │  │ Agent 3:          │  Tools: llm_analyzer,                │       │
│  │  │ Analysis          │         confidence_scorer,           │       │
│  │  │ Expert            │         citation_builder             │       │
│  │  └──────────────────┘                                       │       │
│  └────────────────────────────────────────────────────────────┘       │
└──────────────────────────────────────────────────────────────────────┘
          │              │              │
          ▼              ▼              ▼
┌──────────────────────────────────────────────────────────────────────┐
│                       SERVICE LAYER                                    │
│  ┌──────────────┐ ┌───────────────┐ ┌───────────────┐                │
│  │ DocumentSvc   │ │ EmbeddingSvc  │ │ LLMService    │                │
│  │ - PDF parse   │ │ - HuggingFace │ │ - GPT-4V      │                │
│  │ - OCR         │ │ - Chunking    │ │ - Claude       │                │
│  │ - OpenCV      │ │ - Batch embed │ │ - Prompts      │                │
│  └──────────────┘ └───────────────┘ └───────────────┘                │
└──────────────────────────────────────────────────────────────────────┘
          │              │              │
          ▼              ▼              ▼
┌──────────────────────────────────────────────────────────────────────┐
│                        DATA LAYER                                      │
│  ┌──────────────┐ ┌───────────────┐ ┌───────────────┐                │
│  │  ChromaDB     │ │  Local FS     │ │  LRU Cache    │                │
│  │  (Vector DB)  │ │  (Uploads)    │ │  (Queries)    │                │
│  │  HNSW Index   │ │  (Processed)  │ │               │                │
│  └──────────────┘ └───────────────┘ └───────────────┘                │
└──────────────────────────────────────────────────────────────────────┘
```

### 1.2 Integration Principles

| Principle | Description |
|-----------|-------------|
| **Separation of Concerns** | Each agent owns a single domain; services wrap external dependencies |
| **Async-First** | FastAPI async endpoints; background tasks for document processing |
| **Fail Gracefully** | CrewAI agents have max iteration limits; services have retry with backoff |
| **Stateless API** | All state lives in ChromaDB + filesystem; API containers are interchangeable |
| **Config-Driven** | All tunable parameters in `.env` and `config.py`; no magic numbers in code |

---

## 2. Project Structure & Setup

### 2.1 Directory Layout

```
intelligent-doc-agent/
│
├── agents/                         # CrewAI Agent Definitions
│   ├── __init__.py
│   ├── config/
│   │   ├── agents.yaml             # Agent roles, goals, backstories
│   │   └── tasks.yaml              # Task definitions & expected outputs
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── pdf_extractor.py        # PyPDF2 + pdfplumber tool
│   │   ├── ocr_tool.py             # pytesseract wrapper
│   │   ├── opencv_detector.py      # Table/chart detection with OpenCV
│   │   ├── embedding_tool.py       # HuggingFace embedding generation
│   │   ├── vector_search_tool.py   # ChromaDB semantic search
│   │   └── llm_analyzer_tool.py    # Multi-modal LLM query tool
│   ├── document_processor.py       # Document Processor Agent
│   ├── retrieval_agent.py          # Retrieval Agent
│   ├── analysis_agent.py           # Analysis Agent
│   └── orchestrator.py             # Crew assembly & kickoff
│
├── api/                            # FastAPI Layer
│   ├── __init__.py
│   ├── main.py                     # App factory, lifespan events
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── upload.py               # POST /api/v1/upload
│   │   ├── query.py                # POST /api/v1/query
│   │   ├── documents.py            # GET /api/v1/documents/{id}
│   │   └── health.py               # GET /api/v1/health
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── requests.py             # Pydantic request models
│   │   └── responses.py            # Pydantic response models
│   ├── middleware.py               # CORS, rate limiting, logging
│   └── dependencies.py            # Dependency injection
│
├── services/                       # Business Logic Layer
│   ├── __init__.py
│   ├── document_service.py         # Document processing orchestration
│   ├── embedding_service.py        # Chunking + embedding pipeline
│   ├── vector_db_service.py        # ChromaDB / Pinecone abstraction
│   └── llm_service.py              # LLM provider abstraction
│
├── core/                           # Cross-Cutting Concerns
│   ├── __init__.py
│   ├── config.py                   # Pydantic Settings (reads .env)
│   ├── logger.py                   # Structured JSON logging
│   ├── prompts.py                  # All prompt templates
│   ├── exceptions.py               # Custom exception hierarchy
│   └── constants.py                # Enums, magic strings
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py                 # Shared fixtures
│   ├── unit/
│   │   ├── test_document_service.py
│   │   ├── test_embedding_service.py
│   │   ├── test_vector_db_service.py
│   │   └── test_llm_service.py
│   ├── integration/
│   │   ├── test_upload_flow.py
│   │   ├── test_query_flow.py
│   │   └── test_agent_orchestration.py
│   └── fixtures/
│       ├── sample.pdf
│       ├── sample_scan.png
│       └── sample_table.png
│
├── data/
│   ├── uploads/                    # Raw uploaded files
│   ├── processed/                  # Extracted text/images
│   └── chroma_db/                  # ChromaDB persistence
│
├── docker/
│   ├── Dockerfile                  # Multi-stage production build
│   ├── Dockerfile.dev              # Development build with hot-reload
│   └── docker-compose.yml          # Full stack definition
│
├── app.py                          # Streamlit UI entry point
├── main.py                         # FastAPI entry point (uvicorn)
├── requirements.txt                # Pinned dependencies
├── .env.example                    # Template environment variables
├── .dockerignore
├── .gitignore
└── pyproject.toml                  # Black, pylint, pytest config
```

### 2.2 Environment Configuration

```bash
# .env.example
# ─── LLM Providers ────────────────────────────────
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
LLM_PROVIDER=openai                    # openai | anthropic
LLM_MODEL=gpt-4o                      # gpt-4o | claude-sonnet-4-20250514

# ─── Embeddings ───────────────────────────────────
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DIMENSION=384

# ─── Vector Database ──────────────────────────────
VECTOR_DB_TYPE=chromadb                # chromadb | pinecone
CHROMA_PERSIST_DIR=./data/chroma_db
CHROMA_COLLECTION_NAME=documents
PINECONE_API_KEY=
PINECONE_ENVIRONMENT=
PINECONE_INDEX_NAME=doc-analysis

# ─── Document Processing ─────────────────────────
MAX_FILE_SIZE_MB=50
CHUNK_SIZE=750                         # tokens per chunk
CHUNK_OVERLAP=150                      # overlap between chunks
OCR_LANGUAGE=eng

# ─── API Configuration ───────────────────────────
API_HOST=0.0.0.0
API_PORT=8000
CORS_ORIGINS=["http://localhost:8501"]
RATE_LIMIT_PER_MINUTE=30

# ─── Logging ──────────────────────────────────────
LOG_LEVEL=INFO
LOG_FORMAT=json
```

---

## 3. AI Engineering Foundation

### 3.1 Model Selection Matrix

| Component | Model | Dimension | Why This Choice |
|-----------|-------|-----------|-----------------|
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` | 384 | Fast, small footprint (80MB), good quality for general docs. Runs locally, no API cost |
| **Embeddings (Upgrade)** | `BAAI/bge-base-en-v1.5` | 768 | Better retrieval accuracy (MTEB rank), still reasonable size. Use if accuracy < 85% with MiniLM |
| **Multi-Modal LLM** | `gpt-4o` (primary) | — | Best vision + text understanding. Handles charts, tables, diagrams natively |
| **Multi-Modal LLM (Alt)** | `claude-sonnet-4-20250514` | — | Strong alternative, better at long-context document analysis |
| **OCR Fallback** | `pytesseract` (Tesseract 5) | — | Free, local, no API dependency. Good enough for clean scans |
| **Reranker (Optional)** | `BAAI/bge-reranker-base` | — | Cross-encoder reranking for improved retrieval precision |

### 3.2 LLM Service Abstraction

```python
# services/llm_service.py — Provider-agnostic interface

from abc import ABC, abstractmethod
from dataclasses import dataclass
from base64 import b64encode

@dataclass
class LLMResponse:
    content: str
    model: str
    usage: dict  # {"prompt_tokens": int, "completion_tokens": int}

class BaseLLMProvider(ABC):
    @abstractmethod
    async def generate(self, prompt: str, images: list[bytes] = None) -> LLMResponse:
        ...

class OpenAIProvider(BaseLLMProvider):
    """GPT-4o with vision support."""
    async def generate(self, prompt: str, images: list[bytes] = None) -> LLMResponse:
        messages = [{"role": "user", "content": []}]
        messages[0]["content"].append({"type": "text", "text": prompt})
        if images:
            for img in images:
                messages[0]["content"].append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64encode(img).decode()}"}
                })
        response = await self.client.chat.completions.create(
            model=self.model, messages=messages, max_tokens=2048
        )
        return LLMResponse(
            content=response.choices[0].message.content,
            model=response.model,
            usage=dict(response.usage)
        )

class AnthropicProvider(BaseLLMProvider):
    """Claude with vision support."""
    async def generate(self, prompt: str, images: list[bytes] = None) -> LLMResponse:
        content = []
        if images:
            for img in images:
                content.append({
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/png",
                               "data": b64encode(img).decode()}
                })
        content.append({"type": "text", "text": prompt})
        response = await self.client.messages.create(
            model=self.model, max_tokens=2048,
            messages=[{"role": "user", "content": content}]
        )
        return LLMResponse(
            content=response.content[0].text,
            model=response.model,
            usage={"prompt_tokens": response.usage.input_tokens,
                   "completion_tokens": response.usage.output_tokens}
        )

# Factory
def get_llm_provider(config) -> BaseLLMProvider:
    providers = {"openai": OpenAIProvider, "anthropic": AnthropicProvider}
    return providers[config.LLM_PROVIDER](config)
```

### 3.3 Error Handling & Retry Strategy

```python
# core/exceptions.py
class DocumentProcessingError(Exception):
    """Raised when document extraction fails."""

class EmbeddingError(Exception):
    """Raised when embedding generation fails."""

class VectorDBError(Exception):
    """Raised when vector DB operations fail."""

class LLMError(Exception):
    """Raised when LLM API call fails."""

class AgentOrchestrationError(Exception):
    """Raised when CrewAI agent flow fails."""
```

```python
# Retry decorator used across all external API calls
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import httpx

retry_llm = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type((httpx.TimeoutException, httpx.HTTPStatusError)),
)
```

### 3.4 Structured Logging

```python
# core/logger.py
import logging
import sys
from pythonjsonlogger import jsonlogger

def setup_logger(name: str, level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(name)
    handler = logging.StreamHandler(sys.stdout)
    formatter = jsonlogger.JsonFormatter(
        fmt="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(getattr(logging, level))
    return logger
```

---

## 4. LangChain RAG Pipeline

### 4.1 Document Chunking Strategy

```python
# services/embedding_service.py

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

class EmbeddingService:
    def __init__(self, config):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,       # 750 tokens
            chunk_overlap=config.CHUNK_OVERLAP,  # 150 tokens
            length_function=self._token_length,
            separators=[
                "\n\n",    # Paragraph breaks (highest priority)
                "\n",      # Line breaks
                ". ",      # Sentence boundaries
                ", ",      # Clause boundaries
                " ",       # Word boundaries (last resort)
            ],
        )
        self.embeddings = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={
                "normalize_embeddings": True,  # For cosine similarity
                "batch_size": 64,
            },
        )

    def chunk_document(self, text: str, metadata: dict) -> list[dict]:
        """Split text into semantically meaningful chunks with metadata."""
        chunks = self.text_splitter.create_documents(
            texts=[text],
            metadatas=[metadata],
        )
        # Enrich each chunk with position info
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
            chunk.metadata["total_chunks"] = len(chunks)
        return chunks

    def generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Batch embed texts using HuggingFace model."""
        return self.embeddings.embed_documents(texts)

    def generate_query_embedding(self, query: str) -> list[float]:
        """Embed a single query string."""
        return self.embeddings.embed_query(query)
```

### 4.2 Retrieval Chain with Citations

```python
# services/retrieval_chain.py

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma

DOCUMENT_QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are an expert document analyst. Answer the question based ONLY
on the provided context. If the context contains tables or chart descriptions,
reference them explicitly.

Context:
{context}

Question: {question}

Instructions:
1. Answer based ONLY on the provided context
2. Cite sources using [Page X, Section Y] format
3. If data comes from a table or chart, mention this
4. If you cannot answer from the context, say "Insufficient context to answer"
5. Be concise but complete
6. Rate your confidence: HIGH / MEDIUM / LOW

Answer:"""
)

class RetrievalChain:
    def __init__(self, vector_store, llm):
        self.retriever = vector_store.as_retriever(
            search_type="similarity",         # or "mmr" for diversity
            search_kwargs={
                "k": 5,                        # top-k results
                "score_threshold": 0.3,        # minimum relevance
            },
        )
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",               # All chunks in single prompt
            retriever=self.retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": DOCUMENT_QA_PROMPT},
        )

    async def query(self, question: str, document_id: str = None) -> dict:
        """Execute RAG query with optional document filtering."""
        search_kwargs = {"k": 5}
        if document_id:
            search_kwargs["filter"] = {"document_id": document_id}
            self.retriever.search_kwargs = search_kwargs

        result = await self.qa_chain.ainvoke({"query": question})
        return {
            "answer": result["result"],
            "sources": [
                {
                    "document_id": doc.metadata.get("document_id"),
                    "page": doc.metadata.get("page_number"),
                    "section": doc.metadata.get("section_title", ""),
                    "chunk_text": doc.page_content[:200],
                    "relevance_score": doc.metadata.get("score", 0.0),
                }
                for doc in result["source_documents"]
            ],
        }
```

### 4.3 Hybrid Search Pattern (Optional Enhancement)

```python
# For higher retrieval accuracy, combine vector + keyword search

from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

class HybridRetriever:
    """Combine BM25 keyword search with vector similarity search."""

    def __init__(self, documents, vector_store):
        self.bm25 = BM25Retriever.from_documents(documents, k=5)
        self.vector = vector_store.as_retriever(search_kwargs={"k": 5})
        self.ensemble = EnsembleRetriever(
            retrievers=[self.bm25, self.vector],
            weights=[0.3, 0.7],  # Favor semantic over keyword
        )

    def get_retriever(self):
        return self.ensemble
```

---

## 5. HuggingFace Model Selection & Configuration

### 5.1 Embedding Models

| Model ID | Dimensions | Size | Speed | Quality (MTEB) | Use Case |
|----------|-----------|------|-------|----------------|----------|
| `sentence-transformers/all-MiniLM-L6-v2` | 384 | 80MB | Fast | Good | **Default choice** — best speed/quality balance for MVP |
| `BAAI/bge-base-en-v1.5` | 768 | 440MB | Medium | Better | Upgrade if retrieval accuracy below 85% |
| `sentence-transformers/all-mpnet-base-v2` | 768 | 420MB | Medium | Better | Alternative to BGE, well-tested |
| `BAAI/bge-small-en-v1.5` | 384 | 130MB | Fast | Good | If BGE quality needed with less memory |

**Recommended default:** `all-MiniLM-L6-v2` — It runs entirely on CPU, loads in < 2 seconds, and produces 384-dim embeddings. This is the right tradeoff for a portfolio project handling up to 100 documents.

### 5.2 Model Loading Pattern

```python
# services/embedding_service.py — Model initialization

from sentence_transformers import SentenceTransformer
import torch

class EmbeddingModelManager:
    _instance = None

    def __init__(self, model_name: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=self.device)
        self.model.max_seq_length = 512  # Limit for chunked input
        self.dimension = self.model.get_sentence_embedding_dimension()

    @classmethod
    def get_instance(cls, model_name: str):
        """Singleton — load model once, reuse across requests."""
        if cls._instance is None:
            cls._instance = cls(model_name)
        return cls._instance

    def encode(self, texts: list[str], batch_size: int = 64) -> list[list[float]]:
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,  # Unit vectors for cosine sim
            convert_to_numpy=True,
        )
        return embeddings.tolist()
```

### 5.3 OCR & Document Understanding Models

| Task | Tool/Model | Notes |
|------|-----------|-------|
| **PDF Text Extraction** | `pdfplumber` | Handles native PDFs with layout preservation |
| **OCR (Scanned Docs)** | `pytesseract` (Tesseract 5) | Free, local, works for English. Preprocess with OpenCV first |
| **Table Detection** | OpenCV contour detection | Custom pipeline — detect horizontal/vertical lines |
| **Chart Understanding** | GPT-4o Vision | Send chart images directly to multi-modal LLM |
| **Layout Analysis (Future)** | `microsoft/layoutlmv3-base` | For structured document understanding if needed |

### 5.4 Preprocessing Pipeline for OCR

```python
# agents/tools/ocr_tool.py

import cv2
import numpy as np
import pytesseract
from PIL import Image

def preprocess_for_ocr(image: np.ndarray) -> np.ndarray:
    """Preprocess image to improve OCR accuracy."""
    # 1. Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2. Deskew (straighten tilted scans)
    coords = np.column_stack(np.where(gray > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = gray.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    gray = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC,
                           borderMode=cv2.BORDER_REPLICATE)

    # 3. Adaptive thresholding (handles uneven lighting)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 2
    )

    # 4. Noise removal
    denoised = cv2.medianBlur(binary, 3)

    return denoised

def extract_text_ocr(image_path: str) -> str:
    """Extract text from image using Tesseract with preprocessing."""
    image = cv2.imread(image_path)
    processed = preprocess_for_ocr(image)
    text = pytesseract.image_to_string(
        Image.fromarray(processed),
        lang="eng",
        config="--oem 3 --psm 6"  # LSTM engine, uniform block
    )
    return text.strip()
```

---

## 6. Vector Database Layer

### 6.1 ChromaDB Schema Design

```python
# services/vector_db_service.py

import chromadb
from chromadb.config import Settings

class ChromaDBService:
    def __init__(self, config):
        self.client = chromadb.PersistentClient(
            path=config.CHROMA_PERSIST_DIR,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True,
            ),
        )
        self.collection = self.client.get_or_create_collection(
            name=config.CHROMA_COLLECTION_NAME,
            metadata={
                "hnsw:space": "cosine",           # Similarity metric
                "hnsw:construction_ef": 200,       # Build-time quality
                "hnsw:search_ef": 100,             # Query-time quality
                "hnsw:M": 16,                      # Connections per node
            },
        )

    def add_documents(self, documents: list[dict]) -> None:
        """Batch insert document chunks with metadata."""
        self.collection.add(
            ids=[doc["id"] for doc in documents],
            embeddings=[doc["embedding"] for doc in documents],
            documents=[doc["text"] for doc in documents],
            metadatas=[{
                "document_id": doc["document_id"],
                "filename": doc["filename"],
                "page_number": doc["page_number"],
                "section_title": doc.get("section_title", ""),
                "chunk_index": doc["chunk_index"],
                "content_type": doc.get("content_type", "text"),
                "char_count": len(doc["text"]),
            } for doc in documents],
        )

    def search(self, query_embedding: list[float], top_k: int = 5,
               filters: dict = None) -> list[dict]:
        """Semantic search with optional metadata filtering."""
        where_clause = None
        if filters:
            if "document_id" in filters:
                where_clause = {"document_id": filters["document_id"]}

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_clause,
            include=["documents", "metadatas", "distances"],
        )
        return [
            {
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
                "relevance_score": 1 - results["distances"][0][i],
            }
            for i in range(len(results["ids"][0]))
        ]

    def delete_document(self, document_id: str) -> None:
        """Remove all chunks for a document."""
        self.collection.delete(where={"document_id": document_id})

    def get_document_count(self) -> int:
        return self.collection.count()
```

### 6.2 Metadata Schema

```
┌──────────────────────────────────────────────────┐
│              ChromaDB Collection: "documents"      │
├──────────────────────────────────────────────────┤
│  id           : str   (UUID - unique per chunk)   │
│  embedding    : float[] (384-dim or 768-dim)      │
│  document     : str   (chunk text content)        │
│  metadata:                                         │
│    ├── document_id   : str  (UUID - groups chunks)│
│    ├── filename      : str  (original filename)   │
│    ├── page_number   : int  (source page)         │
│    ├── section_title : str  (detected heading)    │
│    ├── chunk_index   : int  (position in doc)     │
│    ├── content_type  : str  (text|table|image)    │
│    └── char_count    : int  (chunk length)        │
└──────────────────────────────────────────────────┘
```

### 6.3 Pinecone Migration Path

```python
# services/vector_db_service.py — Abstract interface + Pinecone implementation

from abc import ABC, abstractmethod

class VectorDBInterface(ABC):
    @abstractmethod
    def add_documents(self, documents: list[dict]) -> None: ...
    @abstractmethod
    def search(self, query_embedding, top_k, filters) -> list[dict]: ...
    @abstractmethod
    def delete_document(self, document_id: str) -> None: ...

class PineconeService(VectorDBInterface):
    """Drop-in replacement for ChromaDBService."""

    def __init__(self, config):
        from pinecone import Pinecone
        pc = Pinecone(api_key=config.PINECONE_API_KEY)
        self.index = pc.Index(config.PINECONE_INDEX_NAME)

    def add_documents(self, documents: list[dict]) -> None:
        vectors = [
            {
                "id": doc["id"],
                "values": doc["embedding"],
                "metadata": {
                    "document_id": doc["document_id"],
                    "filename": doc["filename"],
                    "page_number": doc["page_number"],
                    "section_title": doc.get("section_title", ""),
                    "chunk_index": doc["chunk_index"],
                    "content_type": doc.get("content_type", "text"),
                    "text": doc["text"][:1000],  # Pinecone metadata limit
                },
            }
            for doc in documents
        ]
        # Batch upsert in groups of 100
        for i in range(0, len(vectors), 100):
            self.index.upsert(vectors=vectors[i:i+100])

    def search(self, query_embedding, top_k=5, filters=None):
        filter_dict = {}
        if filters and "document_id" in filters:
            filter_dict = {"document_id": {"$eq": filters["document_id"]}}
        results = self.index.query(
            vector=query_embedding, top_k=top_k,
            filter=filter_dict if filter_dict else None,
            include_metadata=True,
        )
        return [
            {
                "id": match.id,
                "text": match.metadata.get("text", ""),
                "metadata": match.metadata,
                "relevance_score": match.score,
            }
            for match in results.matches
        ]

# Factory
def get_vector_db(config) -> VectorDBInterface:
    if config.VECTOR_DB_TYPE == "pinecone":
        return PineconeService(config)
    return ChromaDBService(config)  # default
```

---

## 7. Vector Index Tuning

### 7.1 HNSW Parameter Optimization for ChromaDB

ChromaDB uses HNSW (Hierarchical Navigable Small World) index internally. For your workload (~100 documents, ~5000-10000 chunks), these are the optimal settings:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `hnsw:space` | `cosine` | Normalized embeddings from sentence-transformers work best with cosine. More stable than dot product for variable-length text |
| `hnsw:M` | `16` | Default is fine for < 100K vectors. Higher M = better recall but more memory. 16 is the sweet spot |
| `hnsw:construction_ef` | `200` | Build-time effort. Higher = better graph quality. 200 is good since we build once per document upload |
| `hnsw:search_ef` | `100` | Query-time effort. 100 gives ~99% recall@10 for your scale. Increase to 200 if accuracy drops |

### 7.2 Performance Targets & Benchmarks

```
Target: < 500ms query latency (p90)
Breakdown:
  ├── Embedding generation:  ~50ms  (MiniLM-L6 on CPU)
  ├── Vector search:         ~10ms  (ChromaDB, < 100K vectors)
  ├── Metadata filtering:    ~5ms   (in-memory post-filter)
  └── LLM generation:        ~400ms (GPT-4o API, excluding network)
                             ─────────
  Total:                     ~465ms  (under 500ms target)
```

### 7.3 Similarity Metric Decision

```
cosine  <-- CHOSEN
  + Works with normalized embeddings (sentence-transformers default)
  + Invariant to embedding magnitude
  + Standard for text similarity

dot product
  - Requires careful normalization
  - Sensitive to vector magnitude differences

euclidean
  - Penalizes magnitude differences
  - Less meaningful for high-dimensional text embeddings
```

### 7.4 Metadata Filtering Strategy

```python
# Filtering patterns for different query types

# 1. Search within a specific document
results = collection.query(
    query_embeddings=[query_vec],
    n_results=5,
    where={"document_id": "abc-123"},
)

# 2. Search only text content (exclude table/image descriptions)
results = collection.query(
    query_embeddings=[query_vec],
    n_results=5,
    where={"content_type": "text"},
)

# 3. Search a specific page range
results = collection.query(
    query_embeddings=[query_vec],
    n_results=5,
    where={
        "$and": [
            {"document_id": "abc-123"},
            {"page_number": {"$gte": 5}},
            {"page_number": {"$lte": 10}},
        ]
    },
)
```

---

## 8. CrewAI Multi-Agent Orchestration

### 8.1 Agent Definitions (YAML Config)

```yaml
# agents/config/agents.yaml

document_processor:
  role: "Senior Document Processing Engineer"
  goal: >
    Extract ALL content from uploaded documents with maximum fidelity.
    This includes text paragraphs, tables, charts, images, and metadata.
    Preserve page numbers and section structure.
  backstory: >
    You are an expert in document parsing with 10+ years of experience.
    You can handle PDFs, scanned documents, and images. You know when
    to use OCR vs direct text extraction. You detect tables using
    computer vision and preserve their structure.
  verbose: true
  max_iter: 10
  allow_delegation: false

retrieval_specialist:
  role: "Information Retrieval Specialist"
  goal: >
    For any given query, find the most relevant document chunks from
    the vector database. Chunk documents semantically, generate high
    quality embeddings, and perform accurate semantic search with
    metadata filtering.
  backstory: >
    You are a search and retrieval expert specializing in semantic
    search and vector databases. You understand embedding spaces and
    can optimize retrieval by combining vector similarity with
    metadata filtering. You ensure retrieved chunks provide maximum
    context for answering questions.
  verbose: true
  max_iter: 10
  allow_delegation: false

analysis_expert:
  role: "Senior Document Analysis Expert"
  goal: >
    Generate accurate, well-cited answers from retrieved document
    content. Analyze both text and visual elements. Provide confidence
    scores and always cite page numbers and sections.
  backstory: >
    You are an expert analyst capable of synthesizing information from
    multiple document sections. You understand tables, charts, and
    diagrams. You always cite your sources and express confidence
    levels. You never fabricate information — if the context is
    insufficient, you say so clearly.
  verbose: true
  max_iter: 10
  allow_delegation: false
```

### 8.2 Task Definitions

```yaml
# agents/config/tasks.yaml

process_document_task:
  description: >
    Process the uploaded document: {file_path}

    Steps:
    1. Detect file type (PDF, image, scanned)
    2. Extract text from native PDF pages using pdfplumber
    3. Extract embedded images from PDF
    4. Run OCR on scanned pages or image files
    5. Detect tables using OpenCV and extract their content
    6. Record page numbers and section titles for each content block

    Return structured output with all extracted content.
  agent: document_processor
  expected_output: >
    A JSON object with:
    - document_id: UUID
    - filename: string
    - pages: list of page objects, each containing:
      - page_number: int
      - text_blocks: list of text strings
      - tables: list of table objects (headers + rows)
      - images: list of image descriptions or base64 data
      - section_title: detected section heading (if any)

index_document_task:
  description: >
    Index the processed document content into the vector database.

    Input: Processed document from Document Processor.

    Steps:
    1. Split text into semantic chunks (500-1000 tokens, 150 overlap)
    2. Generate embeddings for each chunk using HuggingFace model
    3. Store in ChromaDB with metadata (document_id, page, section, type)
    4. Verify storage by running a test query

    Document ID: {document_id}
  agent: retrieval_specialist
  expected_output: >
    A JSON object with:
    - document_id: UUID
    - chunks_created: int
    - embedding_model: string
    - collection_name: string
    - status: "indexed" | "failed"
  context:
    - process_document_task

answer_query_task:
  description: >
    Answer the user's question using retrieved document context.

    User Question: {query}
    Document Filter: {document_id}

    Steps:
    1. Retrieve top-5 relevant chunks from vector database
    2. If chunks contain references to visual elements, include them
    3. Synthesize an answer from the retrieved context
    4. Cite sources with page numbers and sections
    5. Assign a confidence score (HIGH/MEDIUM/LOW)
  agent: analysis_expert
  expected_output: >
    A JSON object with:
    - answer: string (the generated answer)
    - confidence: string (HIGH | MEDIUM | LOW)
    - sources: list of citation objects with page, section, relevance_score
    - processing_notes: any caveats about the answer
```

### 8.3 CrewAI Tool Wrappers

```python
# agents/tools/pdf_extractor.py
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

class PDFExtractorInput(BaseModel):
    file_path: str = Field(description="Path to the PDF file")

class PDFExtractorTool(BaseTool):
    name: str = "pdf_extractor"
    description: str = "Extract text and images from a PDF file page by page"
    args_schema: type[BaseModel] = PDFExtractorInput

    def _run(self, file_path: str) -> dict:
        import pdfplumber
        pages = []
        with pdfplumber.open(file_path) as pdf:
            for i, page in enumerate(pdf.pages):
                pages.append({
                    "page_number": i + 1,
                    "text": page.extract_text() or "",
                    "tables": page.extract_tables() or [],
                })
        return {"pages": pages, "total_pages": len(pages)}
```

```python
# agents/tools/vector_search_tool.py
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

class VectorSearchInput(BaseModel):
    query: str = Field(description="Search query text")
    document_id: str = Field(default=None, description="Optional document ID filter")
    top_k: int = Field(default=5, description="Number of results to return")

class VectorSearchTool(BaseTool):
    name: str = "vector_search"
    description: str = "Search the vector database for relevant document chunks"
    args_schema: type[BaseModel] = VectorSearchInput

    def __init__(self, vector_db_service, embedding_service):
        super().__init__()
        self._vector_db = vector_db_service
        self._embedding_svc = embedding_service

    def _run(self, query: str, document_id: str = None, top_k: int = 5) -> list:
        query_embedding = self._embedding_svc.generate_query_embedding(query)
        filters = {"document_id": document_id} if document_id else None
        results = self._vector_db.search(query_embedding, top_k, filters)
        return results
```

### 8.4 Crew Assembly & Orchestration

```python
# agents/orchestrator.py

from crewai import Agent, Task, Crew, Process
from core.logger import setup_logger

logger = setup_logger("orchestrator")

class DocumentAnalysisCrew:
    def __init__(self, tools: dict, config):
        self.tools = tools
        self.config = config

    def _build_agents(self) -> dict:
        return {
            "processor": Agent(
                role="Senior Document Processing Engineer",
                goal="Extract ALL content from uploaded documents with maximum fidelity",
                backstory="Expert in document parsing, OCR, and computer vision",
                tools=[
                    self.tools["pdf_extractor"],
                    self.tools["ocr_tool"],
                    self.tools["opencv_detector"],
                ],
                verbose=True,
                max_iter=10,
                allow_delegation=False,
            ),
            "retriever": Agent(
                role="Information Retrieval Specialist",
                goal="Find the most relevant document chunks for any query",
                backstory="Expert in semantic search and vector databases",
                tools=[
                    self.tools["embedding_tool"],
                    self.tools["vector_search"],
                ],
                verbose=True,
                max_iter=10,
                allow_delegation=False,
            ),
            "analyst": Agent(
                role="Senior Document Analysis Expert",
                goal="Generate accurate answers with citations and confidence scores",
                backstory="Expert analyst for text and visual information",
                tools=[
                    self.tools["llm_analyzer"],
                ],
                verbose=True,
                max_iter=10,
                allow_delegation=False,
            ),
        }

    def process_document(self, file_path: str, document_id: str, filename: str) -> dict:
        """Run document processing + indexing pipeline."""
        agents = self._build_agents()

        process_task = Task(
            description=f"Process the uploaded document: {file_path}",
            agent=agents["processor"],
            expected_output="Structured document content as JSON",
        )
        index_task = Task(
            description=f"Index the processed content for document {document_id}",
            agent=agents["retriever"],
            expected_output="Indexing confirmation with chunk count",
            context=[process_task],
        )

        crew = Crew(
            agents=[agents["processor"], agents["retriever"]],
            tasks=[process_task, index_task],
            process=Process.sequential,
            verbose=True,
        )

        logger.info("Starting document processing crew", extra={"file": file_path})
        result = crew.kickoff()
        logger.info("Document processing complete", extra={"result": str(result)})
        return result

    def answer_query(self, query: str, document_id: str = None, top_k: int = 5) -> dict:
        """Run query answering pipeline."""
        agents = self._build_agents()

        retrieve_task = Task(
            description=f"Find top-{top_k} relevant chunks for: {query}",
            agent=agents["retriever"],
            expected_output="Top relevant document chunks with metadata",
        )
        analyze_task = Task(
            description=f"Answer this question: {query}",
            agent=agents["analyst"],
            expected_output="Answer with citations and confidence score as JSON",
            context=[retrieve_task],
        )

        crew = Crew(
            agents=[agents["retriever"], agents["analyst"]],
            tasks=[retrieve_task, analyze_task],
            process=Process.sequential,
            verbose=True,
        )

        logger.info("Starting query crew", extra={"query": query})
        result = crew.kickoff()
        logger.info("Query complete", extra={"result": str(result)})
        return result
```

### 8.5 Agent Error Handling & Guardrails

```python
# agents/orchestrator.py — Error handling wrapper

import time
from core.exceptions import AgentOrchestrationError

class SafeCrewRunner:
    MAX_RETRIES = 2
    TIMEOUT_SECONDS = 120

    @staticmethod
    def run_crew(crew: Crew, context: dict) -> dict:
        for attempt in range(SafeCrewRunner.MAX_RETRIES + 1):
            try:
                start = time.time()
                result = crew.kickoff()
                elapsed = time.time() - start

                if elapsed > SafeCrewRunner.TIMEOUT_SECONDS:
                    logger.warning("Crew exceeded soft timeout",
                                   extra={"elapsed": elapsed})

                return {"status": "success", "result": result, "elapsed": elapsed}

            except Exception as e:
                logger.error(f"Crew attempt {attempt+1} failed",
                            extra={"error": str(e)})
                if attempt == SafeCrewRunner.MAX_RETRIES:
                    raise AgentOrchestrationError(
                        f"Crew failed after {SafeCrewRunner.MAX_RETRIES+1} attempts: {e}"
                    )
                time.sleep(2 ** attempt)  # Exponential backoff
```

---

## 9. API Design & FastAPI Implementation

### 9.1 Application Factory

```python
# api/main.py

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from core.config import get_settings
from core.logger import setup_logger
from api.routes import upload, query, documents, health
from api.dependencies import init_services

logger = setup_logger("api")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: load models, connect to vector DB. Shutdown: cleanup."""
    settings = get_settings()
    logger.info("Initializing services...")
    app.state.services = init_services(settings)
    logger.info("Services ready")
    yield
    logger.info("Shutting down services...")

def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="Intelligent Document Analysis API",
        description="Multi-agent RAG system with computer vision",
        version="1.0.0",
        lifespan=lifespan,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Routes
    app.include_router(upload.router, prefix="/api/v1", tags=["Upload"])
    app.include_router(query.router, prefix="/api/v1", tags=["Query"])
    app.include_router(documents.router, prefix="/api/v1", tags=["Documents"])
    app.include_router(health.router, prefix="/api/v1", tags=["Health"])

    return app

app = create_app()
```

### 9.2 Request / Response Schemas

```python
# api/schemas/requests.py

from pydantic import BaseModel, Field
from typing import Optional

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=2000,
                       description="Natural language question")
    document_id: Optional[str] = Field(None,
                       description="Filter to specific document (UUID)")
    top_k: int = Field(5, ge=1, le=20,
                       description="Number of chunks to retrieve")
    include_images: bool = Field(True,
                       description="Include visual content in analysis")

    model_config = {"json_schema_extra": {
        "examples": [{"query": "What was the Q3 revenue?", "top_k": 5}]
    }}
```

```python
# api/schemas/responses.py

from pydantic import BaseModel, Field
from typing import Optional

class SourceCitation(BaseModel):
    document_id: str
    page: int
    section: str = ""
    chunk_text: str
    relevance_score: float

class QueryResponse(BaseModel):
    answer: str
    confidence: float = Field(ge=0.0, le=1.0)
    sources: list[SourceCitation]
    processing_time_ms: int

class UploadResponse(BaseModel):
    document_id: str
    filename: str
    status: str  # "processing" | "completed" | "failed"
    pages: int
    chunks_created: int
    uploaded_at: str

class HealthResponse(BaseModel):
    status: str
    version: str
    agents: dict[str, str]
    vector_db: str
    document_count: int
    timestamp: str

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
```

### 9.3 Route Implementations

```python
# api/routes/upload.py

import uuid
import time
from fastapi import APIRouter, UploadFile, File, HTTPException, Request, BackgroundTasks
from api.schemas.responses import UploadResponse, ErrorResponse

router = APIRouter()

ALLOWED_TYPES = {
    "application/pdf": "pdf",
    "image/jpeg": "jpg",
    "image/png": "png",
    "image/webp": "webp",
}
MAX_SIZE = 50 * 1024 * 1024  # 50MB

@router.post("/upload", response_model=UploadResponse,
             responses={400: {"model": ErrorResponse}})
async def upload_document(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    # Validate file type
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(400, detail=f"Unsupported format: {file.content_type}. "
                                        f"Allowed: {list(ALLOWED_TYPES.values())}")

    # Validate file size
    contents = await file.read()
    if len(contents) > MAX_SIZE:
        raise HTTPException(400, detail=f"File exceeds {MAX_SIZE // (1024*1024)}MB limit")

    # Save file
    document_id = str(uuid.uuid4())
    file_path = f"data/uploads/{document_id}_{file.filename}"
    with open(file_path, "wb") as f:
        f.write(contents)

    # Process in background
    services = request.app.state.services
    background_tasks.add_task(
        services["orchestrator"].process_document,
        file_path=file_path,
        document_id=document_id,
        filename=file.filename,
    )

    return UploadResponse(
        document_id=document_id,
        filename=file.filename,
        status="processing",
        pages=0,
        chunks_created=0,
        uploaded_at=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
```

```python
# api/routes/query.py

import time
from fastapi import APIRouter, HTTPException, Request
from api.schemas.requests import QueryRequest
from api.schemas.responses import QueryResponse, ErrorResponse

router = APIRouter()

@router.post("/query", response_model=QueryResponse,
             responses={404: {"model": ErrorResponse}})
async def query_documents(request: Request, body: QueryRequest):
    start = time.time()
    services = request.app.state.services

    try:
        result = services["orchestrator"].answer_query(
            query=body.query,
            document_id=body.document_id,
            top_k=body.top_k,
        )
    except Exception as e:
        raise HTTPException(500, detail=f"Query processing failed: {str(e)}")

    elapsed_ms = int((time.time() - start) * 1000)

    return QueryResponse(
        answer=result["answer"],
        confidence=result.get("confidence", 0.5),
        sources=result.get("sources", []),
        processing_time_ms=elapsed_ms,
    )
```

```python
# api/routes/health.py

import time
from fastapi import APIRouter, Request
from api.schemas.responses import HealthResponse

router = APIRouter()

@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request):
    services = request.app.state.services
    vector_db = services["vector_db"]

    return HealthResponse(
        status="healthy",
        version="1.0.0",
        agents={
            "document_processor": "operational",
            "retrieval": "operational",
            "analysis": "operational",
        },
        vector_db="connected",
        document_count=vector_db.get_document_count(),
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
```

### 9.4 Dependency Injection

```python
# api/dependencies.py

from core.config import get_settings
from services.embedding_service import EmbeddingService
from services.vector_db_service import get_vector_db
from services.llm_service import get_llm_provider
from agents.orchestrator import DocumentAnalysisCrew
from agents.tools.pdf_extractor import PDFExtractorTool
from agents.tools.ocr_tool import OCRTool
from agents.tools.opencv_detector import OpenCVTableDetector
from agents.tools.vector_search_tool import VectorSearchTool
from agents.tools.llm_analyzer_tool import LLMAnalyzerTool

def init_services(config) -> dict:
    """Initialize all services at startup. Called once in lifespan."""

    embedding_svc = EmbeddingService(config)
    vector_db = get_vector_db(config)
    llm_provider = get_llm_provider(config)

    tools = {
        "pdf_extractor": PDFExtractorTool(),
        "ocr_tool": OCRTool(),
        "opencv_detector": OpenCVTableDetector(),
        "embedding_tool": embedding_svc,
        "vector_search": VectorSearchTool(vector_db, embedding_svc),
        "llm_analyzer": LLMAnalyzerTool(llm_provider),
    }

    orchestrator = DocumentAnalysisCrew(tools=tools, config=config)

    return {
        "embedding_svc": embedding_svc,
        "vector_db": vector_db,
        "llm_provider": llm_provider,
        "orchestrator": orchestrator,
    }
```

---

## 10. Docker Deployment Strategy

### 10.1 Multi-Stage Dockerfile

```dockerfile
# docker/Dockerfile

# ────────────────────────────────────────────────────────
# Stage 1: System dependencies
# ────────────────────────────────────────────────────────
FROM python:3.11-slim AS base

# System packages for OpenCV, Tesseract, pdf2image
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    poppler-utils \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ────────────────────────────────────────────────────────
# Stage 2: Python dependencies
# ────────────────────────────────────────────────────────
FROM base AS dependencies

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download the HuggingFace embedding model at build time
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# ────────────────────────────────────────────────────────
# Stage 3: Production image
# ────────────────────────────────────────────────────────
FROM dependencies AS production

# Create non-root user
RUN groupadd -g 1001 appgroup && \
    useradd -u 1001 -g appgroup -m appuser

# Copy application code
COPY --chown=appuser:appgroup . .

# Create data directories
RUN mkdir -p data/uploads data/processed data/chroma_db && \
    chown -R appuser:appgroup data/

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "2", "--timeout-keep-alive", "30"]
```

### 10.2 Docker Compose (Full Stack)

```yaml
# docker/docker-compose.yml

version: "3.8"

services:
  # ── FastAPI Backend ────────────────────────────────
  api:
    build:
      context: ..
      dockerfile: docker/Dockerfile
      target: production
    container_name: doc-agent-api
    ports:
      - "8000:8000"
    volumes:
      - upload_data:/app/data/uploads
      - processed_data:/app/data/processed
      - chroma_data:/app/data/chroma_db
      - hf_cache:/home/appuser/.cache/huggingface
    env_file:
      - ../.env
    environment:
      - CHROMA_PERSIST_DIR=/app/data/chroma_db
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/v1/health"]
      interval: 30s
      timeout: 10s
      start_period: 60s
      retries: 3
    restart: unless-stopped
    deploy:
      resources:
        limits:
          cpus: "2.0"
          memory: 4G
        reservations:
          cpus: "1.0"
          memory: 2G

  # ── Streamlit Frontend ─────────────────────────────
  ui:
    build:
      context: ..
      dockerfile: docker/Dockerfile
      target: production
    container_name: doc-agent-ui
    ports:
      - "8501:8501"
    env_file:
      - ../.env
    environment:
      - API_URL=http://api:8000
    depends_on:
      api:
        condition: service_healthy
    command: ["streamlit", "run", "app.py", "--server.port", "8501",
              "--server.address", "0.0.0.0", "--server.headless", "true"]
    restart: unless-stopped
    deploy:
      resources:
        limits:
          cpus: "0.5"
          memory: 512M

volumes:
  upload_data:
    driver: local
  processed_data:
    driver: local
  chroma_data:
    driver: local
  hf_cache:
    driver: local
```

### 10.3 .dockerignore

```
.git
.env
__pycache__
*.pyc
.pytest_cache
.mypy_cache
data/uploads/*
data/processed/*
data/chroma_db/*
notebooks/
*.md
!requirements.txt
.venv
venv
```

### 10.4 Docker Commands Cheatsheet

```bash
# Build and run full stack
docker-compose -f docker/docker-compose.yml up --build -d

# View logs
docker-compose -f docker/docker-compose.yml logs -f api

# Stop everything
docker-compose -f docker/docker-compose.yml down

# Rebuild single service
docker-compose -f docker/docker-compose.yml build api

# Shell into running container
docker exec -it doc-agent-api /bin/bash
```

---

## 11. Integration Flow & Data Pipeline

### 11.1 Document Upload Flow (End-to-End)

```
User uploads PDF via POST /api/v1/upload
        |
        v
[1] FastAPI validates file (type, size)
        |
        v
[2] Save raw file to data/uploads/{uuid}_{filename}
        |
        v
[3] Return immediate response: {"status": "processing", "document_id": "..."}
        |
        v (Background Task)
        |
[4] CrewAI Crew kickoff (Sequential Process)
        |
        |-->  Task 1: Document Processor Agent
        |     |-- Detect file type (PDF / image)
        |     |-- If PDF: pdfplumber extracts text per page
        |     |-- If PDF: extract embedded images
        |     |-- If scanned: OpenCV preprocess then pytesseract OCR
        |     |-- OpenCV detects tables and extracts structure
        |     +-- Output: structured content dict per page
        |
        |-->  Task 2: Retrieval Agent (receives Task 1 output)
        |     |-- RecursiveCharacterTextSplitter chunks text
        |     |-- HuggingFace model generates 384-dim embeddings
        |     |-- ChromaDB stores chunks + metadata
        |     +-- Output: {"chunks_created": N, "status": "indexed"}
        |
        +-->  Update document status to "completed"
```

### 11.2 Query Flow (End-to-End)

```
User sends POST /api/v1/query {"query": "What was Q3 revenue?"}
        |
        v
[1] FastAPI validates request
        |
        v
[2] CrewAI Crew kickoff (Sequential Process)
        |
        |-->  Task 1: Retrieval Agent
        |     |-- Embed query using same HuggingFace model
        |     |-- ChromaDB similarity search (cosine, top_k=5)
        |     |-- Apply document_id filter if provided
        |     +-- Return top-5 chunks with metadata
        |
        |-->  Task 2: Analysis Agent (receives Task 1 output)
        |     |-- Build prompt with context + question
        |     |-- If visual content referenced, include images
        |     |-- Call GPT-4o / Claude with multi-modal prompt
        |     |-- Parse response: answer + confidence + citations
        |     +-- Return structured answer
        |
        +-->  Return to user:
                {
                  "answer": "Q3 revenue was $4.2B...",
                  "confidence": 0.92,
                  "sources": [...],
                  "processing_time_ms": 1847
                }
```

### 11.3 Component Integration Map

```
                   HOW EVERYTHING CONNECTS

  FastAPI ──────────> Orchestrator (CrewAI)
    |                     |
    | (DI)                |--> Agent 1 uses:
    |                     |     |-- PDFExtractorTool
    v                     |     |     +-- pdfplumber
  Dependencies            |     |-- OCRTool
    |-- EmbeddingSvc      |     |     |-- OpenCV (preprocess)
    |   +-- HuggingFace   |     |     +-- pytesseract
    |       model         |     +-- OpenCVDetectorTool
    |-- VectorDBSvc       |           +-- cv2 (table detect)
    |   +-- ChromaDB      |
    +-- LLMService        |--> Agent 2 uses:
        |-- OpenAI        |     |-- EmbeddingTool
        +-- Anthropic     |     |     +-- EmbeddingSvc
                          |     +-- VectorSearchTool
  Streamlit UI --HTTP-->  |           +-- VectorDBSvc
    +-- calls /upload,    |
       /query endpoints   |--> Agent 3 uses:
                          |     +-- LLMAnalyzerTool
                          |           +-- LLMService
                          |               |-- GPT-4o (vision)
                          |               +-- Claude (alt)
```

---

## 12. Testing & Validation Strategy

### 12.1 Unit Tests

```python
# tests/unit/test_embedding_service.py

import pytest
from services.embedding_service import EmbeddingService

@pytest.fixture
def embedding_svc(test_config):
    return EmbeddingService(test_config)

def test_chunk_document_splits_correctly(embedding_svc):
    text = "Word " * 2000  # ~2000 tokens
    chunks = embedding_svc.chunk_document(text, {"document_id": "test"})
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.page_content.split()) <= 800

def test_generate_embeddings_correct_dimension(embedding_svc):
    texts = ["Hello world", "Test document"]
    embeddings = embedding_svc.generate_embeddings(texts)
    assert len(embeddings) == 2
    assert len(embeddings[0]) == 384  # MiniLM dimension

def test_query_embedding_matches_dimension(embedding_svc):
    embedding = embedding_svc.generate_query_embedding("test query")
    assert len(embedding) == 384
```

```python
# tests/unit/test_vector_db_service.py

import pytest
from services.vector_db_service import ChromaDBService

@pytest.fixture
def chroma_svc(tmp_path, test_config):
    test_config.CHROMA_PERSIST_DIR = str(tmp_path / "test_chroma")
    return ChromaDBService(test_config)

def test_add_and_search(chroma_svc):
    docs = [{
        "id": "chunk-1",
        "embedding": [0.1] * 384,
        "text": "Q3 revenue was $4.2 billion",
        "document_id": "doc-1",
        "filename": "report.pdf",
        "page_number": 5,
        "chunk_index": 0,
    }]
    chroma_svc.add_documents(docs)
    results = chroma_svc.search([0.1] * 384, top_k=1)
    assert len(results) == 1
    assert "revenue" in results[0]["text"]

def test_filter_by_document_id(chroma_svc):
    chroma_svc.add_documents([
        {"id": "c1", "embedding": [0.1]*384, "text": "Doc 1",
         "document_id": "doc-1", "filename": "a.pdf",
         "page_number": 1, "chunk_index": 0},
        {"id": "c2", "embedding": [0.2]*384, "text": "Doc 2",
         "document_id": "doc-2", "filename": "b.pdf",
         "page_number": 1, "chunk_index": 0},
    ])
    results = chroma_svc.search([0.1]*384, top_k=5,
                                 filters={"document_id": "doc-1"})
    assert all(r["metadata"]["document_id"] == "doc-1" for r in results)
```

### 12.2 Integration Tests

```python
# tests/integration/test_upload_flow.py

import pytest
from httpx import AsyncClient
from api.main import create_app

@pytest.fixture
async def client():
    app = create_app()
    async with AsyncClient(app=app, base_url="http://test") as c:
        yield c

@pytest.mark.asyncio
async def test_upload_pdf(client, sample_pdf_path):
    with open(sample_pdf_path, "rb") as f:
        response = await client.post(
            "/api/v1/upload",
            files={"file": ("test.pdf", f, "application/pdf")},
        )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "processing"
    assert "document_id" in data

@pytest.mark.asyncio
async def test_upload_rejects_invalid_type(client):
    response = await client.post(
        "/api/v1/upload",
        files={"file": ("test.exe", b"fake", "application/octet-stream")},
    )
    assert response.status_code == 400
```

### 12.3 Manual Testing Checklist

```
[ ] Upload a native PDF (text-heavy) - verify text extraction
[ ] Upload a PDF with embedded tables - verify table detection
[ ] Upload a scanned document image - verify OCR output
[ ] Upload a PDF with charts - verify image extraction
[ ] Query text content - verify accurate answer with citations
[ ] Query table data - verify structured data in response
[ ] Query with document_id filter - verify scoped results
[ ] Query across all documents - verify cross-doc retrieval
[ ] Check /health endpoint - all agents "operational"
[ ] Upload file > 50MB - verify 400 rejection
[ ] Upload .exe file - verify 400 rejection
[ ] Docker build - verify clean build, no errors
[ ] Docker compose up - verify both services start
[ ] Streamlit UI - upload, query, view results
```

---

## 13. Production Hardening Checklist

### 13.1 Security

- [ ] All API keys in `.env`, never in code
- [ ] File type whitelist validation (PDF, JPG, PNG, WEBP only)
- [ ] File size limit enforced (50MB)
- [ ] User query sanitization (strip injection attempts)
- [ ] CORS restricted to Streamlit origin
- [ ] Docker runs as non-root user
- [ ] No secrets in Docker image layers
- [ ] Input validation on all Pydantic models

### 13.2 Reliability

- [ ] Retry with exponential backoff on LLM API calls (3 attempts)
- [ ] CrewAI agents have `max_iter=10` to prevent infinite loops
- [ ] Background task error handling (document status -> "failed")
- [ ] Health check endpoint verifies all dependencies
- [ ] Structured JSON logging for all operations
- [ ] Graceful shutdown handling in Docker

### 13.3 Performance

- [ ] Embedding model loaded once at startup (singleton)
- [ ] ChromaDB HNSW parameters tuned (ef=100, M=16)
- [ ] Batch embedding generation (batch_size=64)
- [ ] Async FastAPI endpoints for non-blocking IO
- [ ] Background tasks for document processing
- [ ] Response time target: < 5 seconds for queries

### 13.4 Observability

- [ ] Structured logging on every request (request_id, endpoint, latency)
- [ ] Agent action logging (CrewAI verbose mode)
- [ ] LLM token usage tracking per request
- [ ] Vector DB query latency logging
- [ ] Error rate tracking per endpoint
- [ ] Health endpoint for monitoring

---

## Summary: Build Order

This is the recommended implementation sequence:

```
Step 1  -> core/config.py + core/logger.py + core/exceptions.py
           (Foundation: config management, logging, error types)

Step 2  -> services/embedding_service.py + services/vector_db_service.py
           (Data layer: chunking, embeddings, ChromaDB)

Step 3  -> agents/tools/*.py
           (CrewAI tools wrapping services)

Step 4  -> services/llm_service.py
           (LLM provider abstraction with retry)

Step 5  -> agents/document_processor.py + agents/retrieval_agent.py
           + agents/analysis_agent.py + agents/orchestrator.py
           (CrewAI agents and crew assembly)

Step 6  -> api/schemas/*.py + api/routes/*.py + api/main.py
           (FastAPI layer with validation)

Step 7  -> app.py
           (Streamlit UI calling the API)

Step 8  -> tests/
           (Unit + integration tests)

Step 9  -> docker/Dockerfile + docker/docker-compose.yml
           (Containerization)

Step 10 -> Final testing, documentation, cleanup
```

Each step builds on the previous one. Services are independent and testable.
Agents wrap services via tools. The API layer orchestrates everything.
Docker packages it all.
