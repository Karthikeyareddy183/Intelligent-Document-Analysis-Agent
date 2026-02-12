"""CrewAI orchestrator — assembles agents, tasks, and crews."""

import json
import re
import time
import uuid
from collections.abc import AsyncIterator
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
        self, file_path: str, document_id: str, filename: str, user_id: Optional[str] = None
    ) -> dict:
        """Run the document processing + indexing pipeline.

        1. Extract text, tables from each page
        2. Optionally extract images and describe them via vision LLM
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

            # Step 2: Extract images from PDF (skip vision if configured)
            page_images: dict[int, list[bytes]] = {}
            if is_pdf and not self.config.SKIP_IMAGE_VISION:
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

                # 3c. Describe images via vision LLM (only when SKIP_IMAGE_VISION=False)
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

            # Step 4+5: Embed and store in batches of 50 chunks
            # This avoids OpenAI token limits and Supabase payload limits
            # for large documents (books with 500-1000+ chunks).
            BATCH_SIZE = 50
            total_chunks = len(all_chunks)
            chunks_indexed = 0

            for batch_start in range(0, total_chunks, BATCH_SIZE):
                batch_end = min(batch_start + BATCH_SIZE, total_chunks)
                batch_chunks = all_chunks[batch_start:batch_end]

                # 4a. Embed this batch
                batch_texts = [chunk["text"] for chunk in batch_chunks]
                batch_embeddings = embedding_svc.generate_embeddings(batch_texts)

                # 4b. Build documents for this batch
                batch_documents = []
                for i, (chunk, embedding) in enumerate(zip(batch_chunks, batch_embeddings)):
                    global_idx = batch_start + i
                    doc_entry = {
                        "id": f"{document_id}_chunk_{global_idx}",
                        "embedding": embedding,
                        "text": chunk["text"],
                        "document_id": document_id,
                        "filename": filename,
                        "page_number": chunk["metadata"].get("page_number", 0),
                        "section_title": chunk["metadata"].get("section_title", ""),
                        "chunk_index": global_idx,
                        "content_type": chunk["metadata"].get("content_type", "text"),
                    }
                    if user_id:
                        doc_entry["user_id"] = user_id
                    batch_documents.append(doc_entry)

                # 4c. Store this batch in vector DB
                batch_indexed = vector_db.add_documents(batch_documents)
                chunks_indexed += batch_indexed

                logger.info(
                    "Batch indexed",
                    extra={
                        "document_id": document_id,
                        "batch": batch_start // BATCH_SIZE + 1,
                        "batch_chunks": len(batch_chunks),
                        "total_indexed": chunks_indexed,
                        "total_chunks": total_chunks,
                    },
                )

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
                    "skip_image_vision": self.config.SKIP_IMAGE_VISION,
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
        self, query: str, document_id: Optional[str] = None, top_k: int = 5, user_id: Optional[str] = None
    ) -> dict:
        """Run the query answering pipeline (non-streaming).

        1. Parse page numbers from query for targeted retrieval
        2. Semantic search (with page filter if detected)
        3. Generate answer with LLM
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
            if user_id:
                filters["user_id"] = user_id

            # Step 2: Search
            query_embedding = embedding_svc.generate_query_embedding(query)

            if target_pages:
                page_results = []
                for page_num in target_pages:
                    page_filter = {**filters, "page_number": page_num}
                    page_hits = vector_db.search(query_embedding, top_k=10, filters=page_filter)
                    page_results.extend(page_hits)

                broad_results = vector_db.search(query_embedding, top_k=top_k, filters=filters if filters else None)

                seen_ids = {r["id"] for r in page_results}
                for r in broad_results:
                    if r["id"] not in seen_ids:
                        page_results.append(r)
                        seen_ids.add(r["id"])

                search_results = page_results[:top_k]
            else:
                search_results = vector_db.search(
                    query_embedding, top_k=top_k, filters=filters if filters else None
                )

            if not search_results:
                return {
                    "answer": "No relevant content found in the documents.",
                    "confidence": 0.0,
                    "sources": [],
                    "processing_time_ms": int((time.time() - start) * 1000),
                }

            # Step 3: Build context from search results
            context_parts, sources = self._build_context_and_sources(search_results)
            context = "\n\n---\n\n".join(context_parts)

            # Step 4: Generate answer using LLM
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

    # ─── Streaming Query Pipeline ─────────────────────────────

    async def answer_query_stream(
        self,
        query: str,
        document_id: Optional[str] = None,
        top_k: int = 5,
        chat_history: Optional[list[dict]] = None,
        user_id: Optional[str] = None,
    ) -> AsyncIterator[dict]:
        """Streaming query pipeline — yields SSE-compatible events.

        Events yielded:
        - {"event": "token", "data": "..."}   — each LLM token
        - {"event": "sources", "data": [...]}  — source citations
        - {"event": "done", "data": ""}        — stream complete
        """
        logger.info(
            "Starting streaming query pipeline",
            extra={"query": query[:100], "document_id": document_id},
        )

        try:
            embedding_svc = self.services["embedding_service"]
            vector_db = self.services["vector_db"]
            llm_service = self.services["llm_provider"]

            # Step 1: Parse page numbers
            target_pages = self._extract_page_numbers(query)
            filters = {}
            if document_id:
                filters["document_id"] = document_id
            if user_id:
                filters["user_id"] = user_id

            # Step 2: Search
            query_embedding = embedding_svc.generate_query_embedding(query)

            if target_pages:
                page_results = []
                for page_num in target_pages:
                    page_filter = {**filters, "page_number": page_num}
                    page_hits = vector_db.search(query_embedding, top_k=10, filters=page_filter)
                    page_results.extend(page_hits)

                broad_results = vector_db.search(query_embedding, top_k=top_k, filters=filters if filters else None)
                seen_ids = {r["id"] for r in page_results}
                for r in broad_results:
                    if r["id"] not in seen_ids:
                        page_results.append(r)
                        seen_ids.add(r["id"])
                search_results = page_results[:top_k]
            else:
                search_results = vector_db.search(
                    query_embedding, top_k=top_k, filters=filters if filters else None
                )

            if not search_results:
                yield {"event": "token", "data": "No relevant content found in the documents."}
                yield {"event": "sources", "data": []}
                yield {"event": "done", "data": ""}
                return

            # Step 3: Build context
            context_parts, sources = self._build_context_and_sources(search_results)
            context = "\n\n---\n\n".join(context_parts)

            # Step 4: Build messages list with system prompt + chat history + current query
            from core.prompts import DOCUMENT_QA_PROMPT

            system_prompt = (
                "You are an expert document analyst. Answer based ONLY on the provided context. "
                "Cite sources using [Page X] format. If you cannot answer, say so. "
                "Rate confidence at the end: HIGH / MEDIUM / LOW."
            )

            messages = [{"role": "system", "content": system_prompt}]

            # Add chat history if provided
            if chat_history:
                for msg in chat_history:
                    messages.append({"role": msg["role"], "content": msg["content"]})

            # Add current question with context
            user_message = f"Context:\n{context}\n\nQuestion: {query}"
            messages.append({"role": "user", "content": user_message})

            # Step 5: Stream LLM response
            async for token in llm_service.generate_stream(messages):
                yield {"event": "token", "data": token}

            # Step 6: Send sources and done
            yield {"event": "sources", "data": sources}
            yield {"event": "done", "data": ""}

        except Exception as e:
            logger.error("Streaming query failed", extra={"error": str(e)})
            yield {"event": "error", "data": str(e)}

    # ─── Helper Methods ──────────────────────────────────────

    def _build_context_and_sources(
        self, search_results: list[dict]
    ) -> tuple[list[str], list[dict]]:
        """Build context parts and sources list from search results."""
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
        return context_parts, sources

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
        """Extract page numbers mentioned in the query."""
        pages = set()

        for match in re.finditer(r'(?:pages?|pg\.?|p\.)\s*(\d+)', query, re.IGNORECASE):
            pages.add(int(match.group(1)))

        for match in re.finditer(r'(?:pages?)\s*(\d+)\s*[-\u2013to]+\s*(\d+)', query, re.IGNORECASE):
            start, end = int(match.group(1)), int(match.group(2))
            if end - start <= 20:
                pages.update(range(start, end + 1))

        return sorted(pages)

    def _rerank(self, query: str, results: list[dict], top_k: int) -> list[dict]:
        """Rerank search results using cross-encoder (kept for optional use)."""
        if len(results) <= top_k:
            return results

        try:
            from sentence_transformers import CrossEncoder
            reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)

            pairs = [[query, r["text"]] for r in results]
            scores = reranker.predict(pairs)

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
