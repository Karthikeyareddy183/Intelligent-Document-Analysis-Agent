"""CrewAI orchestrator — assembles agents, tasks, and crews."""

import re
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

    # ─── Document Processing Pipeline ────────────────────────

    def process_document(
        self, file_path: str, document_id: str, filename: str
    ) -> dict:
        """Run the document processing + indexing pipeline.

        1. Extract text, tables from each page
        2. Extract images and describe them via vision LLM
        3. Chunk all content with page-level metadata
        4. Generate embeddings and store in vector DB
        """
        logger.info(
            "Starting document processing",
            extra={"document_id": document_id, "file": filename},
        )

        start = time.time()

        try:
            doc_service = self.services["document_service"]
            embedding_svc = self.services["embedding_service"]
            vector_db = self.services["vector_db"]
            llm_service = self.services["llm_provider"]
            is_pdf = filename.lower().endswith(".pdf")

            # Step 1: Extract text + tables
            if is_pdf:
                pages = doc_service.extract_pdf_content(file_path)
            else:
                pages = doc_service.extract_image_content(file_path)

            total_pages = len(pages)

            # Step 2: Extract images from PDF (non-blocking — skips if fails)
            page_images: dict[int, list[bytes]] = {}
            if is_pdf:
                page_images = doc_service.extract_page_images(file_path)

            # Step 3: Build all chunks (text + tables + image descriptions)
            all_chunks = []

            for page in pages:
                page_num = page["page_number"]
                text = page.get("text", "")

                # 3a. Chunk page text
                if text.strip():
                    chunks = embedding_svc.chunk_text(
                        text,
                        metadata={
                            "document_id": document_id,
                            "filename": filename,
                            "page_number": page_num,
                            "content_type": ContentType.TEXT,
                        },
                    )
                    all_chunks.extend(chunks)

                # 3b. Chunk tables as whole units
                for table in page.get("tables", []):
                    table_text = self._table_to_text(table)
                    if table_text:
                        all_chunks.append({
                            "text": table_text,
                            "metadata": {
                                "document_id": document_id,
                                "filename": filename,
                                "page_number": page_num,
                                "content_type": ContentType.TABLE,
                                "chunk_index": 0,
                            },
                        })

                # 3c. Describe images via vision LLM and add as chunks
                if page_num in page_images:
                    for img_idx, img_bytes in enumerate(page_images[page_num]):
                        description = self._describe_image(llm_service, img_bytes, page_num, img_idx + 1)
                        if description:
                            all_chunks.append({
                                "text": f"[Image {img_idx + 1} on Page {page_num}]\n{description}",
                                "metadata": {
                                    "document_id": document_id,
                                    "filename": filename,
                                    "page_number": page_num,
                                    "content_type": ContentType.IMAGE,
                                    "image_index": img_idx + 1,
                                    "chunk_index": 0,
                                },
                            })

            if not all_chunks:
                doc_service.update_document_status(
                    document_id, DocumentStatus.FAILED, error="No content extracted"
                )
                return {
                    "status": "failed",
                    "document_id": document_id,
                    "error": "No content extracted from document",
                }

            # Step 4: Generate embeddings in batches
            texts = [chunk["text"] for chunk in all_chunks]
            embeddings = embedding_svc.generate_embeddings(texts)

            # Step 5: Store in vector DB
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

            image_count = sum(len(v) for v in page_images.values())
            elapsed = round(time.time() - start, 2)
            logger.info(
                "Document processing complete",
                extra={
                    "document_id": document_id,
                    "pages": total_pages,
                    "chunks": chunks_indexed,
                    "images_described": image_count,
                    "elapsed_s": elapsed,
                },
            )

            return {
                "status": "completed",
                "document_id": document_id,
                "pages": total_pages,
                "chunks_created": chunks_indexed,
                "images_processed": image_count,
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

    # ─── Query Pipeline ──────────────────────────────────────

    def answer_query(
        self, query: str, document_id: Optional[str] = None, top_k: int = 5
    ) -> dict:
        """Run the query answering pipeline with page-aware routing and reranking.

        1. Parse page numbers from query for targeted retrieval
        2. Semantic search (with page filter if detected)
        3. Rerank results with cross-encoder (if available)
        4. Generate answer with LLM
        """
        logger.info(
            "Starting query pipeline",
            extra={"query": query[:100], "document_id": document_id},
        )

        start = time.time()

        try:
            embedding_svc = self.services["embedding_service"]
            vector_db = self.services["vector_db"]

            # Step 1: Parse page numbers from the query
            target_pages = self._extract_page_numbers(query)
            filters = {}
            if document_id:
                filters["document_id"] = document_id

            # Step 2: Search — if user mentions specific pages, do page-filtered + broader search
            query_embedding = embedding_svc.generate_query_embedding(query)

            if target_pages:
                # First: get chunks from the specific pages
                page_results = []
                for page_num in target_pages:
                    page_filter = {**filters, "page_number": page_num}
                    page_hits = vector_db.search(query_embedding, top_k=10, filters=page_filter)
                    page_results.extend(page_hits)

                # Also do a broader search for surrounding context
                broad_results = vector_db.search(query_embedding, top_k=top_k, filters=filters if filters else None)

                # Merge: prioritize page-specific results, then add broader ones
                seen_ids = {r["id"] for r in page_results}
                for r in broad_results:
                    if r["id"] not in seen_ids:
                        page_results.append(r)
                        seen_ids.add(r["id"])

                search_results = page_results[:top_k + 5]  # Allow extra for reranking
            else:
                # No page mentioned — standard semantic search with more candidates for reranking
                search_results = vector_db.search(
                    query_embedding, top_k=top_k * 2, filters=filters if filters else None
                )

            if not search_results:
                return {
                    "answer": "No relevant content found in the documents.",
                    "confidence": 0.0,
                    "sources": [],
                    "processing_time_ms": int((time.time() - start) * 1000),
                }

            # Step 3: Rerank with cross-encoder for precision
            search_results = self._rerank(query, search_results, top_k)

            # Step 4: Build context from search results
            context_parts = []
            sources = []
            for result in search_results:
                page = result["metadata"].get("page_number", 0)
                content_type = result["metadata"].get("content_type", "text")
                image_index = result["metadata"].get("image_index", "")
                label = f"[Page {page}, Type: {content_type}"
                if image_index:
                    label += f", Image #{image_index}"
                label += "]"
                context_parts.append(f"{label}\n{result['text']}")
                sources.append({
                    "document_id": result["metadata"].get("document_id", ""),
                    "page": page,
                    "section": result["metadata"].get("section_title", ""),
                    "chunk_text": result["text"][:200],
                    "relevance_score": result["relevance_score"],
                })

            context = "\n\n---\n\n".join(context_parts)

            # Step 5: Generate answer using LLM
            llm_service = self.services["llm_provider"]
            from core.prompts import DOCUMENT_QA_PROMPT

            prompt = DOCUMENT_QA_PROMPT.format(context=context, question=query)
            response = llm_service.generate_sync(prompt)

            confidence = self._extract_confidence(response.content)

            elapsed_ms = int((time.time() - start) * 1000)
            logger.info(
                "Query complete",
                extra={
                    "elapsed_ms": elapsed_ms,
                    "sources": len(sources),
                    "confidence": confidence,
                    "page_filter": target_pages or "none",
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

    # ─── Helper Methods ──────────────────────────────────────

    def _describe_image(self, llm_service, image_bytes: bytes, page_num: int, image_index: int) -> str:
        """Send image to vision LLM and get a text description."""
        try:
            from core.prompts import IMAGE_DESCRIPTION_PROMPT
            response = llm_service.generate_sync(IMAGE_DESCRIPTION_PROMPT, images=[image_bytes])
            logger.info(
                "Image described",
                extra={"page": page_num, "image_index": image_index, "desc_len": len(response.content)},
            )
            return response.content
        except Exception as e:
            logger.warning(
                f"Failed to describe image (page {page_num}, #{image_index}): {e}"
            )
            return ""

    def _extract_page_numbers(self, query: str) -> list[int]:
        """Extract page numbers mentioned in the query.

        Handles patterns like:
        - "page 123"
        - "on page 45"
        - "pages 10-15"
        - "p. 7"
        - "pg 42"
        """
        pages = set()

        # Match "page(s) X" or "p. X" or "pg X"
        for match in re.finditer(r'(?:pages?|pg\.?|p\.)\s*(\d+)', query, re.IGNORECASE):
            pages.add(int(match.group(1)))

        # Match page ranges "pages 10-15" or "pages 10 to 15"
        for match in re.finditer(r'(?:pages?)\s*(\d+)\s*[-–to]+\s*(\d+)', query, re.IGNORECASE):
            start, end = int(match.group(1)), int(match.group(2))
            if end - start <= 20:  # Sanity limit
                pages.update(range(start, end + 1))

        return sorted(pages)

    def _rerank(self, query: str, results: list[dict], top_k: int) -> list[dict]:
        """Rerank search results using cross-encoder for better precision.

        Falls back to original ordering if cross-encoder is not available.
        """
        if len(results) <= top_k:
            return results

        try:
            from sentence_transformers import CrossEncoder
            reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)

            pairs = [[query, r["text"]] for r in results]
            scores = reranker.predict(pairs)

            # Attach scores and sort
            for i, score in enumerate(scores):
                results[i]["rerank_score"] = float(score)

            results.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)

            logger.info("Reranking complete", extra={"candidates": len(results), "top_k": top_k})
            return results[:top_k]

        except ImportError:
            logger.info("Cross-encoder not available, using original ranking")
            return results[:top_k]
        except Exception as e:
            logger.warning(f"Reranking failed (non-fatal): {e}")
            return results[:top_k]

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
