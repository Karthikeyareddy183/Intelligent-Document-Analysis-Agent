"""CrewAI orchestrator — assembles agents, tasks, and crews."""

import time
import uuid
from typing import Optional

from crewai import Agent, Task, Crew, Process

from core.config import Settings
from core.logger import setup_logger
from core.constants import DocumentStatus, ContentType
from core.exceptions import AgentOrchestrationError

logger = setup_logger(__name__)


class DocumentAnalysisCrew:
    """Manages agent creation and crew execution for document processing and querying."""

    def __init__(self, services: dict, config: Settings):
        self.services = services
        self.config = config

    def _build_processor_agent(self) -> Agent:
        return Agent(
            role="Senior Document Processing Engineer",
            goal="Extract ALL content from uploaded documents with maximum fidelity",
            backstory=(
                "You are an expert in document parsing with deep experience in "
                "PDF extraction, OCR, and computer vision table detection. "
                "You never skip content and always preserve page structure."
            ),
            tools=[
                self.services["tools"]["pdf_extractor"],
                self.services["tools"]["ocr_tool"],
                self.services["tools"]["opencv_detector"],
            ],
            verbose=True,
            max_iter=10,
            allow_delegation=False,
        )

    def _build_retrieval_agent(self) -> Agent:
        return Agent(
            role="Information Retrieval Specialist",
            goal="Find the most relevant document chunks for any query",
            backstory=(
                "You are a semantic search expert. You generate high-quality "
                "embeddings, store them efficiently, and retrieve the most "
                "relevant context for answering questions."
            ),
            tools=[
                self.services["tools"]["embedding_tool"],
                self.services["tools"]["vector_search"],
            ],
            verbose=True,
            max_iter=10,
            allow_delegation=False,
        )

    def _build_analysis_agent(self) -> Agent:
        return Agent(
            role="Senior Document Analysis Expert",
            goal="Generate accurate answers with citations and confidence scores",
            backstory=(
                "You are an expert analyst who synthesizes information from "
                "multiple document sections. You cite sources, express confidence "
                "levels, and never fabricate information."
            ),
            tools=[
                self.services["tools"]["llm_analyzer"],
            ],
            verbose=True,
            max_iter=10,
            allow_delegation=False,
        )

    def process_document(
        self, file_path: str, document_id: str, filename: str
    ) -> dict:
        """Run the document processing + indexing pipeline.

        1. Document Processor Agent extracts content
        2. Retrieval Agent chunks and indexes into vector DB
        """
        logger.info(
            "Starting document processing",
            extra={"document_id": document_id, "file": filename},
        )

        start = time.time()

        try:
            # Step 1: Extract content from document
            doc_service = self.services["document_service"]
            is_pdf = filename.lower().endswith(".pdf")

            if is_pdf:
                pages = doc_service.extract_pdf_content(file_path)
            else:
                pages = doc_service.extract_image_content(file_path)

            total_pages = len(pages)

            # Step 2: Chunk and embed all text content
            embedding_svc = self.services["embedding_service"]
            vector_db = self.services["vector_db"]

            all_chunks = []
            for page in pages:
                text = page.get("text", "")
                if not text.strip():
                    continue

                # Chunk the page text
                chunks = embedding_svc.chunk_text(
                    text,
                    metadata={
                        "document_id": document_id,
                        "filename": filename,
                        "page_number": page["page_number"],
                        "content_type": ContentType.TEXT,
                    },
                )
                all_chunks.extend(chunks)

                # Also index table content as separate chunks
                for table in page.get("tables", []):
                    table_text = self._table_to_text(table)
                    if table_text:
                        table_chunks = embedding_svc.chunk_text(
                            table_text,
                            metadata={
                                "document_id": document_id,
                                "filename": filename,
                                "page_number": page["page_number"],
                                "content_type": ContentType.TABLE,
                            },
                        )
                        all_chunks.extend(table_chunks)

            if not all_chunks:
                doc_service.update_document_status(
                    document_id, DocumentStatus.FAILED, error="No content extracted"
                )
                return {
                    "status": "failed",
                    "document_id": document_id,
                    "error": "No content extracted from document",
                }

            # Step 3: Generate embeddings
            texts = [chunk["text"] for chunk in all_chunks]
            embeddings = embedding_svc.generate_embeddings(texts)

            # Step 4: Store in vector DB
            documents = []
            for i, (chunk, embedding) in enumerate(zip(all_chunks, embeddings)):
                documents.append({
                    "id": f"{document_id}_chunk_{i}",
                    "embedding": embedding,
                    "text": chunk["text"],
                    "document_id": document_id,
                    "filename": filename,
                    "page_number": chunk["metadata"].get("page_number", 0),
                    "section_title": chunk["metadata"].get("section_title", ""),
                    "chunk_index": i,
                    "content_type": chunk["metadata"].get("content_type", "text"),
                })

            chunks_indexed = vector_db.add_documents(documents)

            # Update document status
            doc_service.update_document_status(
                document_id,
                DocumentStatus.COMPLETED,
                pages=total_pages,
                chunks_created=chunks_indexed,
            )

            elapsed = round(time.time() - start, 2)
            logger.info(
                "Document processing complete",
                extra={
                    "document_id": document_id,
                    "pages": total_pages,
                    "chunks": chunks_indexed,
                    "elapsed_s": elapsed,
                },
            )

            return {
                "status": "completed",
                "document_id": document_id,
                "pages": total_pages,
                "chunks_created": chunks_indexed,
                "processing_time_s": elapsed,
            }

        except Exception as e:
            logger.error(
                "Document processing failed",
                extra={"document_id": document_id, "error": str(e)},
            )
            doc_service = self.services.get("document_service")
            if doc_service:
                doc_service.update_document_status(
                    document_id, DocumentStatus.FAILED, error=str(e)
                )
            return {
                "status": "failed",
                "document_id": document_id,
                "error": str(e),
            }

    def answer_query(
        self, query: str, document_id: Optional[str] = None, top_k: int = 5
    ) -> dict:
        """Run the query answering pipeline.

        1. Retrieval Agent finds relevant chunks
        2. Analysis Agent generates answer with citations
        """
        logger.info(
            "Starting query pipeline",
            extra={"query": query[:100], "document_id": document_id},
        )

        start = time.time()

        try:
            # Step 1: Embed query and search
            embedding_svc = self.services["embedding_service"]
            vector_db = self.services["vector_db"]

            query_embedding = embedding_svc.generate_query_embedding(query)
            filters = {"document_id": document_id} if document_id else None
            search_results = vector_db.search(query_embedding, top_k, filters)

            if not search_results:
                return {
                    "answer": "No relevant content found in the documents.",
                    "confidence": 0.0,
                    "sources": [],
                    "processing_time_ms": int((time.time() - start) * 1000),
                }

            # Step 2: Build context from search results
            context_parts = []
            sources = []
            for result in search_results:
                page = result["metadata"].get("page_number", 0)
                content_type = result["metadata"].get("content_type", "text")
                context_parts.append(
                    f"[Page {page}, Type: {content_type}]\n{result['text']}"
                )
                sources.append({
                    "document_id": result["metadata"].get("document_id", ""),
                    "page": page,
                    "section": result["metadata"].get("section_title", ""),
                    "chunk_text": result["text"][:200],
                    "relevance_score": result["relevance_score"],
                })

            context = "\n\n---\n\n".join(context_parts)

            # Step 3: Generate answer using LLM
            llm_service = self.services["llm_provider"]
            from core.prompts import DOCUMENT_QA_PROMPT

            prompt = DOCUMENT_QA_PROMPT.format(context=context, question=query)
            response = llm_service.generate_sync(prompt)

            # Parse confidence
            confidence = self._extract_confidence(response.content)

            elapsed_ms = int((time.time() - start) * 1000)
            logger.info(
                "Query complete",
                extra={
                    "elapsed_ms": elapsed_ms,
                    "sources": len(sources),
                    "confidence": confidence,
                },
            )

            return {
                "answer": response.content,
                "confidence": confidence,
                "sources": sources,
                "processing_time_ms": elapsed_ms,
            }

        except Exception as e:
            logger.error("Query pipeline failed", extra={"error": str(e)})
            raise AgentOrchestrationError(f"Query failed: {e}")

    def _table_to_text(self, table: dict) -> str:
        """Convert a table dict to readable text for embedding."""
        parts = []
        headers = table.get("headers", [])
        rows = table.get("rows", [])

        if headers:
            parts.append("Table headers: " + " | ".join(str(h) for h in headers))
        for row in rows:
            parts.append(" | ".join(str(cell) if cell else "" for cell in row))

        return "\n".join(parts)

    def _extract_confidence(self, text: str) -> float:
        """Extract confidence level from LLM response."""
        text_lower = text.lower()
        last_lines = text_lower.strip().split("\n")[-3:]
        last_text = " ".join(last_lines)

        if "high" in last_text:
            return 0.9
        elif "medium" in last_text:
            return 0.7
        elif "low" in last_text:
            return 0.4
        return 0.5
