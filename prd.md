# Product Requirements Document (PRD)

## Multi-Agent RAG System with Computer Vision Integration

**Project Name:** Intelligent Document Analysis Agent with Visual Understanding  
**Version:** 1.0  
**Author:** M Karthikeya Reddy  
**Date:** February 10, 2026  
**Target Completion:** 2 Days

---

## 1. Executive Summary

### 1.1 Overview
An intelligent multi-agent AI system that processes documents (PDFs, images, scanned documents) to extract, analyze, and answer questions using Retrieval-Augmented Generation (RAG) combined with Computer Vision capabilities. The system employs multiple specialized AI agents working in orchestration to provide comprehensive document understanding.

### 1.2 Business Objectives
- Demonstrate production-ready AI/ML engineering skills for Data Scientist/AI-ML Engineer role
- Showcase expertise in GenAI, RAG pipelines, Agentic AI, and Computer Vision
- Build a scalable, deployable system that solves real-world document analysis problems
- Create a portfolio project that highlights skills matching the target job description

### 1.3 Success Metrics
- Successfully process PDFs, images, and scanned documents
- Achieve >85% accuracy in answering questions from document content
- Handle both textual and visual elements (charts, tables, diagrams)
- Response time <5 seconds for typical queries
- Production-ready code with proper error handling and logging
- Deployable via Docker with comprehensive documentation

---

## 2. Problem Statement

### 2.1 Current Challenges
- Traditional document QA systems struggle with visual elements (charts, tables, diagrams)
- Single-agent systems lack specialized capabilities for complex document analysis
- Many solutions don't handle multi-modal content (text + images) effectively
- Existing tools often require manual extraction of visual data

### 2.2 Target Users
- Data analysts needing insights from reports
- Researchers analyzing academic papers
- Business professionals reviewing financial documents
- Anyone needing to extract information from complex documents

### 2.3 Use Cases
1. **Financial Report Analysis:** "What was the Q3 revenue?" → Extracts from tables/charts
2. **Research Paper Review:** "Summarize the methodology section" → Reads text and diagrams
3. **Invoice Processing:** "Extract vendor details and line items" → OCR + structured extraction
4. **Medical Document Analysis:** "What were the patient's vitals?" → Reads scanned forms

---

## 3. Product Vision & Goals

### 3.1 Vision Statement
Create an intelligent document assistant that understands both textual and visual content, leveraging multi-agent orchestration to provide accurate, contextual answers from complex documents.

### 3.2 Core Goals
1. **Multi-Modal Understanding:** Process text, images, tables, and charts
2. **Agent Orchestration:** Demonstrate Agentic AI with specialized agents
3. **Production Quality:** Scalable, maintainable, deployable code
4. **User Experience:** Simple API and UI for easy interaction
5. **Performance:** Fast retrieval and accurate responses

### 3.3 Non-Goals (Out of Scope)
- Real-time collaboration features
- Advanced document editing capabilities
- Support for audio/video files
- Multi-language translation (English only for v1.0)
- Authentication/authorization (future enhancement)

---

## 4. Technical Architecture

### 4.1 System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     User Interface                       │
│              (Streamlit Web App / API Client)            │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│                   FastAPI Backend                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │   Upload     │  │    Query     │  │   Status     │  │
│  │  Endpoint    │  │  Endpoint    │  │  Endpoint    │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│              Multi-Agent Orchestration Layer             │
│                      (CrewAI)                            │
│  ┌─────────────────────────────────────────────────┐   │
│  │  Document Processor Agent                        │   │
│  │  - PDF text extraction                          │   │
│  │  - Image extraction & preprocessing (OpenCV)    │   │
│  │  - OCR for scanned documents                    │   │
│  │  - Table/chart detection                        │   │
│  └─────────────────────────────────────────────────┘   │
│                       │                                  │
│                       ▼                                  │
│  ┌─────────────────────────────────────────────────┐   │
│  │  Retrieval Agent                                 │   │
│  │  - Document chunking (semantic)                 │   │
│  │  - Embedding generation (HuggingFace)          │   │
│  │  - Vector storage (Pinecone/ChromaDB)          │   │
│  │  - Semantic search & retrieval                  │   │
│  └─────────────────────────────────────────────────┘   │
│                       │                                  │
│                       ▼                                  │
│  ┌─────────────────────────────────────────────────┐   │
│  │  Analysis Agent                                  │   │
│  │  - Multi-modal LLM (GPT-4 Vision/Claude)       │   │
│  │  - Prompt engineering                           │   │
│  │  - Answer generation with citations            │   │
│  │  - Confidence scoring                           │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│                   Data Layer                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │  Vector DB   │  │  Document    │  │   Cache      │  │
│  │  (Pinecone/  │  │   Storage    │  │   Layer      │  │
│  │  ChromaDB)   │  │  (Local FS)  │  │              │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────┘
```

### 4.2 Technology Stack

**Backend Framework:**
- FastAPI (async, high-performance REST API)
- Python 3.10+

**LLM & AI:**
- OpenAI GPT-4 Vision / Claude Sonnet 4.5 (multi-modal understanding)
- HuggingFace Transformers (embeddings)
- LangChain (RAG pipeline orchestration)
- CrewAI (multi-agent framework)

**Vector Database:**
- Primary: ChromaDB (local, easy setup)
- Alternative: Pinecone (cloud, production-scale)

**Computer Vision:**
- OpenCV (image preprocessing, table detection)
- pytesseract (OCR for scanned documents)
- pdf2image (PDF to image conversion)

**Document Processing:**
- PyPDF2 / pdfplumber (PDF text extraction)
- Pillow (image manipulation)

**Frontend:**
- Streamlit (interactive web UI)

**Deployment:**
- Docker & Docker Compose
- GitHub Actions (CI/CD)

**Development Tools:**
- pytest (unit testing)
- black (code formatting)
- pylint (linting)
- python-dotenv (environment management)

### 4.3 Data Flow

1. **Document Upload:**
   ```
   User → FastAPI → Document Processor Agent → Extract text/images → Store locally
   ```

2. **Document Indexing:**
   ```
   Extracted Content → Retrieval Agent → Chunk → Embed → Vector DB
   ```

3. **Query Processing:**
   ```
   User Query → Retrieval Agent → Semantic Search → Relevant Chunks
   → Analysis Agent → LLM Processing → Generated Answer → User
   ```

---

## 5. Feature Requirements

### 5.1 Core Features (MVP - Must Have)

#### 5.1.1 Document Upload & Processing
- **Priority:** P0
- **Description:** Accept PDF and image files for processing
- **Acceptance Criteria:**
  - Support PDF files up to 50MB
  - Support image formats: JPG, PNG, WEBP
  - Extract text from native PDFs
  - Extract embedded images from PDFs
  - Detect and extract tables using OpenCV
  - OCR support for scanned documents
  - Store processed documents with unique IDs

#### 5.1.2 Vector Embedding & Indexing
- **Priority:** P0
- **Description:** Convert document content to vector embeddings
- **Acceptance Criteria:**
  - Semantic chunking (500-1000 tokens per chunk)
  - Generate embeddings using HuggingFace models
  - Store embeddings in vector database
  - Support metadata tagging (page numbers, section titles)
  - Enable document versioning

#### 5.1.3 Question Answering
- **Priority:** P0
- **Description:** Answer questions based on document content
- **Acceptance Criteria:**
  - Accept natural language queries
  - Retrieve top-k relevant chunks (k=5)
  - Generate answers using multi-modal LLM
  - Provide source citations (page numbers, sections)
  - Handle both text and visual content queries
  - Response time <5 seconds for 90% of queries

#### 5.1.4 Multi-Agent Orchestration
- **Priority:** P0
- **Description:** Coordinate multiple specialized agents
- **Acceptance Criteria:**
  - Document Processor Agent functional
  - Retrieval Agent functional
  - Analysis Agent functional
  - Agents communicate via CrewAI framework
  - Error handling between agents
  - Logging of agent interactions

#### 5.1.5 REST API
- **Priority:** P0
- **Description:** Production-ready API endpoints
- **Acceptance Criteria:**
  - `POST /upload` - Upload documents
  - `POST /query` - Ask questions
  - `GET /documents/{id}` - Retrieve document info
  - `GET /health` - Health check endpoint
  - Proper HTTP status codes
  - Request/response validation
  - API documentation (auto-generated via FastAPI)

### 5.2 Enhanced Features (Nice to Have)

#### 5.2.1 Streamlit UI
- **Priority:** P1
- **Description:** Interactive web interface
- **Acceptance Criteria:**
  - Document upload interface
  - Query input field
  - Display answers with citations
  - Show confidence scores
  - Document preview functionality

#### 5.2.2 Batch Processing
- **Priority:** P2
- **Description:** Process multiple documents simultaneously
- **Acceptance Criteria:**
  - Accept multiple file uploads
  - Queue-based processing
  - Progress tracking

#### 5.2.3 Advanced Computer Vision
- **Priority:** P2
- **Description:** Enhanced visual element understanding
- **Acceptance Criteria:**
  - Chart/graph extraction
  - Table structure recognition
  - Diagram understanding

---

## 6. Agent Specifications

### 6.1 Document Processor Agent

**Role:** Extract and preprocess document content

**Responsibilities:**
- Extract text from PDFs using pdfplumber
- Extract images from PDFs
- Perform OCR on scanned documents
- Detect tables and charts using OpenCV
- Preprocess images (deskewing, noise removal)
- Generate structured output for indexing

**Tools:**
- PyPDF2, pdfplumber
- OpenCV
- pytesseract
- Pillow

**Input:** Uploaded document files
**Output:** Structured content (text blocks, images, tables, metadata)

### 6.2 Retrieval Agent

**Role:** Manage document embeddings and semantic search

**Responsibilities:**
- Chunk documents semantically
- Generate embeddings using HuggingFace models
- Store embeddings in vector database
- Perform semantic similarity search
- Rank and return relevant chunks
- Handle metadata filtering

**Tools:**
- LangChain (text splitters, retrievers)
- HuggingFace sentence-transformers
- ChromaDB/Pinecone
- FAISS (optional fallback)

**Input:** Structured content from Document Processor
**Output:** Top-k relevant document chunks for user query

### 6.3 Analysis Agent

**Role:** Generate intelligent answers from retrieved content

**Responsibilities:**
- Process multi-modal content (text + images)
- Apply prompt engineering for optimal responses
- Generate answers with citations
- Assess confidence levels
- Handle follow-up questions
- Ensure factual accuracy

**Tools:**
- OpenAI GPT-4 Vision / Claude Sonnet
- LangChain (prompt templates, output parsers)
- Custom prompt library

**Input:** User query + retrieved chunks + visual content
**Output:** Natural language answer with citations and confidence score

---

## 7. API Specifications

### 7.1 Endpoints

#### 7.1.1 Upload Document
```
POST /api/v1/upload
Content-Type: multipart/form-data

Request:
{
  "file": <binary>,
  "metadata": {
    "title": "string (optional)",
    "tags": ["string"] (optional)
  }
}

Response (200):
{
  "document_id": "uuid",
  "filename": "string",
  "status": "processing|completed|failed",
  "pages": integer,
  "chunks_created": integer,
  "uploaded_at": "timestamp"
}

Response (400):
{
  "error": "Invalid file format",
  "supported_formats": ["pdf", "jpg", "png", "webp"]
}
```

#### 7.1.2 Query Document
```
POST /api/v1/query
Content-Type: application/json

Request:
{
  "query": "string",
  "document_id": "uuid (optional - searches all if not provided)",
  "top_k": integer (default: 5),
  "include_images": boolean (default: true)
}

Response (200):
{
  "answer": "string",
  "confidence": float (0.0-1.0),
  "sources": [
    {
      "document_id": "uuid",
      "page": integer,
      "chunk_text": "string",
      "relevance_score": float
    }
  ],
  "processing_time_ms": integer
}

Response (404):
{
  "error": "Document not found",
  "document_id": "uuid"
}
```

#### 7.1.3 Get Document Info
```
GET /api/v1/documents/{document_id}

Response (200):
{
  "document_id": "uuid",
  "filename": "string",
  "uploaded_at": "timestamp",
  "pages": integer,
  "total_chunks": integer,
  "status": "string",
  "metadata": {
    "title": "string",
    "tags": ["string"]
  }
}
```

#### 7.1.4 Health Check
```
GET /api/v1/health

Response (200):
{
  "status": "healthy",
  "version": "1.0.0",
  "agents": {
    "document_processor": "operational",
    "retrieval": "operational",
    "analysis": "operational"
  },
  "vector_db": "connected",
  "timestamp": "timestamp"
}
```

---

## 8. Implementation Plan (2 Days)

### 8.1 Day 1: Core RAG + Agent Setup (8-10 hours)

#### Morning (4 hours)
- [ ] **Hour 1-2: Project Setup**
  - Initialize project structure
  - Set up virtual environment
  - Install dependencies
  - Configure environment variables (.env)
  - Set up logging infrastructure

- [ ] **Hour 3-4: Document Processor Agent**
  - Implement PDF text extraction
  - Implement image extraction from PDFs
  - Basic OCR integration
  - Create agent wrapper class

#### Afternoon (4 hours)
- [ ] **Hour 5-6: FastAPI Backend**
  - Create FastAPI application structure
  - Implement `/upload` endpoint
  - Implement file validation
  - Set up async request handling
  - Add basic error handling

- [ ] **Hour 7-8: Vector Database Integration**
  - Set up ChromaDB
  - Implement embedding generation
  - Create indexing pipeline
  - Test vector storage and retrieval

#### Evening (2 hours)
- [ ] **Hour 9-10: Basic RAG Pipeline**
  - Implement document chunking
  - Set up LangChain retriever
  - Create simple QA chain
  - Test end-to-end flow (upload → index → query)

**Day 1 Deliverables:**
- ✅ Working document upload and processing
- ✅ Vector database with indexed content
- ✅ Basic RAG query functionality
- ✅ FastAPI backend with 2 endpoints

---

### 8.2 Day 2: Multi-Agent + Production Polish (8-10 hours)

#### Morning (4 hours)
- [ ] **Hour 1-2: CrewAI Multi-Agent Setup**
  - Implement Retrieval Agent
  - Implement Analysis Agent
  - Configure agent orchestration
  - Set up agent communication flow

- [ ] **Hour 3-4: GPT-4 Vision Integration**
  - Integrate OpenAI GPT-4 Vision API
  - Implement multi-modal query handling
  - Add image preprocessing with OpenCV
  - Create prompt templates for visual understanding

#### Afternoon (3 hours)
- [ ] **Hour 5-6: Streamlit UI**
  - Create document upload interface
  - Build query interface
  - Display results with citations
  - Add document preview functionality

- [ ] **Hour 7: Production Enhancements**
  - Add comprehensive error handling
  - Implement structured logging
  - Add input validation
  - Create configuration management

#### Evening (3 hours)
- [ ] **Hour 8: Docker Containerization**
  - Create Dockerfile
  - Create docker-compose.yml
  - Test container build and run
  - Optimize image size

- [ ] **Hour 9-10: Documentation & Testing**
  - Write comprehensive README.md
  - Create API documentation
  - Write unit tests for critical functions
  - Create demo video/screenshots
  - Final testing and bug fixes

**Day 2 Deliverables:**
- ✅ Multi-agent system with CrewAI
- ✅ GPT-4 Vision integration
- ✅ Streamlit web interface
- ✅ Docker containerization
- ✅ Complete documentation
- ✅ Production-ready codebase

---

## 9. Technical Specifications

### 9.1 Performance Requirements
- Document processing: <30 seconds for 10-page PDF
- Query response time: <5 seconds (p90)
- Embedding generation: <2 seconds per chunk
- Vector search: <500ms
- API response time: <6 seconds total (p90)
- Concurrent requests: Support 10+ simultaneous users

### 9.2 Scalability Requirements
- Handle documents up to 50MB
- Support up to 100 documents in vector database (MVP)
- Efficient memory usage (<4GB RAM)
- Horizontal scaling capability (stateless design)

### 9.3 Quality Requirements
- Code coverage: >70% for core functions
- Type hints for all functions
- Docstrings for all classes and functions
- Black formatting (line length: 88)
- Pylint score: >8.0/10
- Error handling for all external API calls
- Structured logging (JSON format)

### 9.4 Security Requirements
- Input validation for all endpoints
- File type validation (whitelist approach)
- File size limits enforced
- Sanitize user queries
- Environment variables for API keys
- No hardcoded credentials
- CORS configuration for production

---

## 10. Testing Strategy

### 10.1 Unit Tests
- Document processing functions
- Embedding generation
- Vector storage and retrieval
- Agent communication
- API endpoint validation

### 10.2 Integration Tests
- End-to-end document upload → query flow
- Multi-agent orchestration
- Vector database operations
- LLM API integration

### 10.3 Manual Testing Checklist
- [ ] Upload PDF with text only
- [ ] Upload PDF with images and tables
- [ ] Upload scanned document (OCR)
- [ ] Query text-based content
- [ ] Query visual content (charts, tables)
- [ ] Test error handling (invalid files, network errors)
- [ ] Test concurrent requests
- [ ] Verify Docker deployment
- [ ] Test Streamlit UI functionality

### 10.4 Test Documents
- Financial report with charts (e.g., quarterly earnings)
- Research paper with diagrams
- Invoice with tables
- Scanned document requiring OCR

---

## 11. Deployment Strategy

### 11.1 Local Development
```bash
# Clone repository
git clone <repo-url>
cd intelligent-doc-agent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with API keys

# Run application
uvicorn main:app --reload
```

### 11.2 Docker Deployment
```bash
# Build image
docker build -t intelligent-doc-agent:latest .

# Run with docker-compose
docker-compose up -d

# Access application
# API: http://localhost:8000
# Streamlit: http://localhost:8501
```

### 11.3 Environment Variables
```
# .env file
OPENAI_API_KEY=sk-xxx
ANTHROPIC_API_KEY=sk-ant-xxx
VECTOR_DB_TYPE=chromadb  # or pinecone
PINECONE_API_KEY=xxx  # if using Pinecone
PINECONE_ENVIRONMENT=xxx
LOG_LEVEL=INFO
MAX_FILE_SIZE_MB=50
```

---

## 12. Monitoring & Observability

### 12.1 Logging Strategy
- **Structured Logging:** JSON format for easy parsing
- **Log Levels:**
  - ERROR: Critical failures, exceptions
  - WARNING: Degraded performance, fallback usage
  - INFO: Request/response logs, agent actions
  - DEBUG: Detailed execution flow

### 12.2 Metrics to Track
- Request count per endpoint
- Average response time
- Error rate
- Document processing success rate
- Vector DB query latency
- LLM API call success rate
- Token usage (cost tracking)

### 12.3 Health Checks
- API health endpoint (`/health`)
- Vector DB connectivity
- Agent status verification
- Disk space monitoring

---

## 13. Risks & Mitigations

### 13.1 Technical Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| LLM API rate limits | High | Medium | Implement retry logic with exponential backoff |
| Large file processing timeout | Medium | Medium | Add async processing queue, progress tracking |
| Vector DB performance degradation | High | Low | Optimize chunking strategy, add caching layer |
| OCR accuracy on poor scans | Medium | High | Preprocess images with OpenCV, provide confidence scores |
| Embedding model size/performance | Medium | Low | Use quantized models, cache embeddings |

### 13.2 Timeline Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| CrewAI learning curve | Medium | Medium | Fallback to simple sequential agents if needed |
| Docker setup issues | Low | Low | Test early, use docker-compose for simplicity |
| API integration bugs | Medium | Medium | Build incrementally, test each integration |

---

## 14. Success Criteria

### 14.1 Functional Success
- ✅ Successfully processes PDFs, images, and scanned documents
- ✅ Accurately answers questions about document content
- ✅ Handles both text and visual queries
- ✅ Multi-agent orchestration working correctly
- ✅ FastAPI endpoints functional and documented
- ✅ Streamlit UI provides good user experience

### 14.2 Technical Success
- ✅ Code is well-structured, type-hinted, and documented
- ✅ Docker containerization works without issues
- ✅ Unit tests cover critical functionality
- ✅ Comprehensive README with setup instructions
- ✅ GitHub repository with clear commit history
- ✅ Demonstrates production-ready coding practices

### 14.3 Portfolio Success
- ✅ Directly showcases skills from job description
- ✅ Demonstrates multi-agent AI expertise
- ✅ Shows RAG pipeline implementation
- ✅ Highlights computer vision integration
- ✅ Production-quality code suitable for technical interviews
- ✅ Clear documentation for hiring managers to review

---

## 15. Future Enhancements (Post-MVP)

### 15.1 Phase 2 Features
- Multi-document querying (cross-document insights)
- Document comparison functionality
- Conversation memory (follow-up questions)
- Export answers to formats (PDF, DOCX)
- Advanced table extraction with structure preservation

### 15.2 Phase 3 Features
- User authentication and document privacy
- Document collections/folders
- API key management for multi-tenant usage
- Advanced analytics dashboard
- Support for additional document formats (Excel, PowerPoint)

### 15.3 Technical Improvements
- Implement caching layer (Redis)
- Add message queue for async processing (Celery)
- Deploy to cloud (AWS/Azure/GCP)
- Set up CI/CD pipeline (GitHub Actions)
- Add monitoring (Prometheus, Grafana)
- Implement A/B testing for different LLM models

---

## 16. Appendices

### 16.1 Project Structure
```
intelligent-doc-agent/
├── agents/
│   ├── __init__.py
│   ├── document_processor.py
│   ├── retrieval_agent.py
│   ├── analysis_agent.py
│   └── orchestrator.py
├── api/
│   ├── __init__.py
│   ├── routes.py
│   ├── models.py
│   └── dependencies.py
├── services/
│   ├── __init__.py
│   ├── document_service.py
│   ├── embedding_service.py
│   ├── vector_db_service.py
│   └── llm_service.py
├── utils/
│   ├── __init__.py
│   ├── config.py
│   ├── logger.py
│   ├── prompts.py
│   └── validators.py
├── tests/
│   ├── __init__.py
│   ├── test_agents.py
│   ├── test_services.py
│   └── test_api.py
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── data/
│   ├── uploads/
│   └── processed/
├── notebooks/
│   └── demo.ipynb
├── app.py                 # Streamlit UI
├── main.py               # FastAPI entry point
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
├── PRD.md                # This document
└── LICENSE
```

### 16.2 Key Dependencies
```
# requirements.txt
fastapi==0.109.0
uvicorn[standard]==0.27.0
python-multipart==0.0.6
pydantic==2.5.3
python-dotenv==1.0.0

# LLM & AI
langchain==0.1.4
langchain-community==0.0.16
crewai==0.1.25
openai==1.10.0
anthropic==0.18.0

# Embeddings
sentence-transformers==2.3.1
transformers==4.37.2

# Vector DB
chromadb==0.4.22
# pinecone-client==3.0.2  # Alternative

# Document Processing
PyPDF2==3.0.1
pdfplumber==0.10.3
pdf2image==1.17.0
pytesseract==0.3.10
Pillow==10.2.0

# Computer Vision
opencv-python==4.9.0.80

# UI
streamlit==1.30.0

# Testing
pytest==7.4.4
pytest-asyncio==0.23.3

# Code Quality
black==24.1.1
pylint==3.0.3

# Utilities
python-json-logger==2.0.7
tenacity==8.2.3
```

### 16.3 Prompt Templates

**Document Analysis Prompt:**
```python
ANALYSIS_PROMPT = """
You are an expert document analyst. Based on the following context extracted from a document, answer the user's question accurately and concisely.

Context:
{context}

Visual Elements:
{visual_elements}

User Question: {question}

Instructions:
1. Answer based ONLY on the provided context
2. If the answer involves data from charts/tables, mention this explicitly
3. Provide page numbers or section references
4. If you cannot answer from the context, say "I don't have enough information to answer this question"
5. Be concise but complete

Answer:
"""
```

**Table Extraction Prompt:**
```python
TABLE_EXTRACTION_PROMPT = """
Analyze this image which contains a table. Extract the data in a structured format.

Image: {image_base64}

Return the data in this format:
{{
    "table_type": "financial|general|comparison",
    "headers": ["column1", "column2", ...],
    "rows": [
        ["value1", "value2", ...],
        ...
    ],
    "summary": "Brief description of what the table shows"
}}
"""
```

### 16.4 Agent Configuration

```python
# agents/config.py
from crewai import Agent, Task, Crew

# Document Processor Agent
document_processor = Agent(
    role="Document Processor",
    goal="Extract and preprocess all content from uploaded documents",
    backstory="Expert in document parsing, OCR, and image extraction",
    tools=[pdf_extractor, image_extractor, ocr_tool],
    verbose=True
)

# Retrieval Agent
retrieval_agent = Agent(
    role="Information Retriever",
    goal="Find the most relevant document chunks for user queries",
    backstory="Specialist in semantic search and information retrieval",
    tools=[vector_search, embedding_generator],
    verbose=True
)

# Analysis Agent
analysis_agent = Agent(
    role="Document Analyst",
    goal="Generate accurate answers from retrieved content",
    backstory="Expert analyst capable of understanding text and visual information",
    tools=[llm_analyzer, confidence_scorer],
    verbose=True
)
```

---

## 17. References & Resources

### 17.1 Documentation Links
- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction)
- [CrewAI Documentation](https://docs.crewai.com/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [OpenAI Vision API](https://platform.openai.com/docs/guides/vision)
- [Claude API Documentation](https://docs.anthropic.com/)

### 17.2 Tutorials Referenced
- Multi-Agent RAG Systems
- Document Understanding with Computer Vision
- Production FastAPI Best Practices
- Docker for ML Applications

---

## 18. Approval & Sign-off

**Product Owner:** M Karthikeya Reddy  
**Target Role:** Data Scientist / AI-ML Engineer (AI Agents & Machine Learning)  
**Company:** IndoSakura Solutions  

**Approval Date:** February 10, 2026  
**Target Completion:** February 12, 2026  

---

**Document Version History:**
- v1.0 (Feb 10, 2026) - Initial PRD created