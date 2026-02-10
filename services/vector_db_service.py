"""Vector database service — ChromaDB and Pinecone abstraction."""

from abc import ABC, abstractmethod
from typing import Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from core.config import Settings
from core.logger import setup_logger
from core.exceptions import VectorDBError

logger = setup_logger(__name__)


class VectorDBInterface(ABC):
    """Abstract interface for vector database operations."""

    @abstractmethod
    def add_documents(self, documents: list[dict]) -> int:
        """Insert document chunks. Returns count of documents added."""
        ...

    @abstractmethod
    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filters: Optional[dict] = None,
    ) -> list[dict]:
        """Semantic search. Returns ranked results."""
        ...

    @abstractmethod
    def delete_document(self, document_id: str) -> None:
        """Remove all chunks belonging to a document."""
        ...

    @abstractmethod
    def get_document_count(self) -> int:
        """Return total number of chunks stored."""
        ...


class ChromaDBService(VectorDBInterface):
    """ChromaDB implementation with HNSW index and cosine similarity."""

    def __init__(self, config: Settings):
        try:
            self.client = chromadb.PersistentClient(
                path=config.CHROMA_PERSIST_DIR,
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                ),
            )
            self.collection = self.client.get_or_create_collection(
                name=config.CHROMA_COLLECTION_NAME,
                metadata={
                    "hnsw:space": "cosine",
                    "hnsw:construction_ef": 200,
                    "hnsw:search_ef": 100,
                    "hnsw:M": 16,
                },
            )
            logger.info(
                "ChromaDB initialized",
                extra={
                    "collection": config.CHROMA_COLLECTION_NAME,
                    "count": self.collection.count(),
                },
            )
        except Exception as e:
            raise VectorDBError(f"Failed to initialize ChromaDB: {e}")

    def add_documents(self, documents: list[dict]) -> int:
        """Batch insert document chunks with embeddings and metadata."""
        if not documents:
            return 0

        try:
            ids = [doc["id"] for doc in documents]
            embeddings = [doc["embedding"] for doc in documents]
            texts = [doc["text"] for doc in documents]
            metadatas = []

            for doc in documents:
                meta = {
                    "document_id": doc["document_id"],
                    "filename": doc.get("filename", ""),
                    "page_number": doc.get("page_number", 0),
                    "section_title": doc.get("section_title", ""),
                    "chunk_index": doc.get("chunk_index", 0),
                    "content_type": doc.get("content_type", "text"),
                    "char_count": len(doc["text"]),
                }
                metadatas.append(meta)

            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
            )

            logger.info("Documents added to ChromaDB", extra={"count": len(documents)})
            return len(documents)

        except Exception as e:
            raise VectorDBError(f"Failed to add documents: {e}")

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filters: Optional[dict] = None,
    ) -> list[dict]:
        """Semantic search with optional metadata filtering."""
        try:
            where_clause = None
            if filters:
                if "document_id" in filters:
                    where_clause = {"document_id": filters["document_id"]}
                elif "content_type" in filters:
                    where_clause = {"content_type": filters["content_type"]}

            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where_clause,
                include=["documents", "metadatas", "distances"],
            )

            if not results["ids"] or not results["ids"][0]:
                return []

            output = []
            for i in range(len(results["ids"][0])):
                distance = results["distances"][0][i]
                output.append({
                    "id": results["ids"][0][i],
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": distance,
                    "relevance_score": round(1 - distance, 4),
                })

            logger.info(
                "Vector search completed",
                extra={"results": len(output), "top_score": output[0]["relevance_score"] if output else 0},
            )
            return output

        except Exception as e:
            raise VectorDBError(f"Search failed: {e}")

    def delete_document(self, document_id: str) -> None:
        try:
            self.collection.delete(where={"document_id": document_id})
            logger.info("Document deleted from ChromaDB", extra={"document_id": document_id})
        except Exception as e:
            raise VectorDBError(f"Failed to delete document: {e}")

    def get_document_count(self) -> int:
        return self.collection.count()


class PineconeService(VectorDBInterface):
    """Pinecone cloud implementation — drop-in replacement for ChromaDB."""

    def __init__(self, config: Settings):
        try:
            from pinecone import Pinecone

            pc = Pinecone(api_key=config.PINECONE_API_KEY)
            self.index = pc.Index(config.PINECONE_INDEX_NAME)
            logger.info("Pinecone initialized", extra={"index": config.PINECONE_INDEX_NAME})
        except ImportError:
            raise VectorDBError("pinecone-client package is not installed")
        except Exception as e:
            raise VectorDBError(f"Failed to initialize Pinecone: {e}")

    def add_documents(self, documents: list[dict]) -> int:
        if not documents:
            return 0

        vectors = []
        for doc in documents:
            vectors.append({
                "id": doc["id"],
                "values": doc["embedding"],
                "metadata": {
                    "document_id": doc["document_id"],
                    "filename": doc.get("filename", ""),
                    "page_number": doc.get("page_number", 0),
                    "section_title": doc.get("section_title", ""),
                    "chunk_index": doc.get("chunk_index", 0),
                    "content_type": doc.get("content_type", "text"),
                    "text": doc["text"][:1000],
                },
            })

        # Batch upsert in groups of 100
        for i in range(0, len(vectors), 100):
            self.index.upsert(vectors=vectors[i : i + 100])

        logger.info("Documents added to Pinecone", extra={"count": len(documents)})
        return len(documents)

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filters: Optional[dict] = None,
    ) -> list[dict]:
        filter_dict = {}
        if filters and "document_id" in filters:
            filter_dict = {"document_id": {"$eq": filters["document_id"]}}

        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            filter=filter_dict if filter_dict else None,
            include_metadata=True,
        )

        return [
            {
                "id": match.id,
                "text": match.metadata.get("text", ""),
                "metadata": match.metadata,
                "distance": 1 - match.score,
                "relevance_score": round(match.score, 4),
            }
            for match in results.matches
        ]

    def delete_document(self, document_id: str) -> None:
        self.index.delete(filter={"document_id": {"$eq": document_id}})

    def get_document_count(self) -> int:
        stats = self.index.describe_index_stats()
        return stats.total_vector_count


def get_vector_db(config: Settings) -> VectorDBInterface:
    """Factory function — returns the configured vector DB implementation."""
    if config.VECTOR_DB_TYPE == "pinecone":
        return PineconeService(config)
    return ChromaDBService(config)
