"""CrewAI tool for generating embeddings and indexing documents."""

from crewai.tools import BaseTool
from pydantic import BaseModel, Field
import uuid

from core.logger import setup_logger

logger = setup_logger(__name__)


class EmbeddingIndexInput(BaseModel):
    texts: list[str] = Field(description="List of text chunks to embed and index")
    document_id: str = Field(description="Document ID to associate with chunks")
    filename: str = Field(default="", description="Original filename")
    page_numbers: list[int] = Field(
        default=[], description="Page number for each text chunk"
    )


class EmbeddingIndexTool(BaseTool):
    name: str = "embedding_indexer"
    description: str = (
        "Generate embeddings for text chunks and store them in the vector database. "
        "Takes a list of text chunks and indexes them with metadata."
    )
    args_schema: type[BaseModel] = EmbeddingIndexInput

    _embedding_service: object = None
    _vector_db_service: object = None

    def __init__(self, embedding_service, vector_db_service, **kwargs):
        super().__init__(**kwargs)
        self._embedding_service = embedding_service
        self._vector_db_service = vector_db_service

    def _run(
        self,
        texts: list[str],
        document_id: str,
        filename: str = "",
        page_numbers: list[int] = None,
    ) -> dict:
        if not texts:
            return {"status": "success", "chunks_indexed": 0}

        logger.info(
            "Indexing document chunks",
            extra={"document_id": document_id, "chunks": len(texts)},
        )

        try:
            # Generate embeddings
            embeddings = self._embedding_service.generate_embeddings(texts)

            # Prepare documents for vector DB
            documents = []
            for i, (text, embedding) in enumerate(zip(texts, embeddings)):
                page_num = page_numbers[i] if page_numbers and i < len(page_numbers) else 0
                documents.append({
                    "id": f"{document_id}_chunk_{i}",
                    "embedding": embedding,
                    "text": text,
                    "document_id": document_id,
                    "filename": filename,
                    "page_number": page_num,
                    "chunk_index": i,
                    "content_type": "text",
                })

            # Store in vector DB
            count = self._vector_db_service.add_documents(documents)

            logger.info(
                "Indexing complete",
                extra={"document_id": document_id, "indexed": count},
            )
            return {
                "status": "success",
                "chunks_indexed": count,
                "document_id": document_id,
            }

        except Exception as e:
            logger.error("Indexing failed", extra={"error": str(e)})
            return {"status": "error", "error": str(e), "chunks_indexed": 0}
