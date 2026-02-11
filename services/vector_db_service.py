"""Vector database service — ChromaDB, Pinecone, and Supabase pgvector abstraction."""

from abc import ABC, abstractmethod
from typing import Optional

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
            import chromadb
            from chromadb.config import Settings as ChromaSettings

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


class SupabaseVectorService(VectorDBInterface):
    """Supabase pgvector implementation — unified vector + metadata + file storage."""

    def __init__(self, config: Settings):
        try:
            from supabase import create_client

            self.client = create_client(config.SUPABASE_URL, config.SUPABASE_SERVICE_ROLE_KEY)
            self.table_name = config.SUPABASE_TABLE_NAME
            self.bucket_name = config.SUPABASE_BUCKET_NAME

            # Verify connection by fetching count
            result = self.client.table(self.table_name).select("id", count="exact").limit(0).execute()
            count = result.count or 0

            logger.info(
                "Supabase pgvector initialized",
                extra={"table": self.table_name, "count": count},
            )
        except ImportError:
            raise VectorDBError("supabase package is not installed. Run: pip install supabase")
        except Exception as e:
            raise VectorDBError(f"Failed to initialize Supabase: {e}")

    def add_documents(self, documents: list[dict]) -> int:
        """Batch insert document chunks with embeddings and metadata into Supabase."""
        if not documents:
            return 0

        try:
            rows = []
            for doc in documents:
                rows.append({
                    "content": doc["text"],
                    "document_id": doc["document_id"],
                    "file_name": doc.get("filename", ""),
                    "page_number": doc.get("page_number", 0),
                    "section_title": doc.get("section_title", ""),
                    "chunk_index": doc.get("chunk_index", 0),
                    "content_type": doc.get("content_type", "text"),
                    "char_count": len(doc["text"]),
                    "embedding": doc["embedding"],
                })

            # Batch insert in groups of 500
            for i in range(0, len(rows), 500):
                batch = rows[i : i + 500]
                self.client.table(self.table_name).insert(batch).execute()

            logger.info("Documents added to Supabase", extra={"count": len(documents)})
            return len(documents)

        except Exception as e:
            raise VectorDBError(f"Failed to add documents to Supabase: {e}")

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filters: Optional[dict] = None,
    ) -> list[dict]:
        """Semantic search via RPC call to match_documents function in Supabase."""
        try:
            rpc_params = {
                "query_embedding": query_embedding,
                "match_threshold": 0.1,
                "match_count": top_k,
            }

            # Add filters if provided
            if filters:
                if "document_id" in filters:
                    rpc_params["filter_document_id"] = filters["document_id"]
                if "page_number" in filters:
                    rpc_params["filter_page_number"] = filters["page_number"]

            response = self.client.rpc("match_documents", rpc_params).execute()

            output = []
            for row in response.data:
                similarity = row.get("similarity", 0)
                metadata = {
                    "document_id": row.get("document_id", ""),
                    "filename": row.get("file_name", ""),
                    "page_number": row.get("page_number", 0),
                    "section_title": row.get("section_title", ""),
                    "chunk_index": row.get("chunk_index", 0),
                    "content_type": row.get("content_type", "text"),
                }
                # Extract image_index from content if it's an image chunk
                if metadata["content_type"] == "image":
                    import re
                    img_match = re.search(r'\[Image (\d+)', row.get("content", ""))
                    if img_match:
                        metadata["image_index"] = int(img_match.group(1))
                output.append({
                    "id": str(row["id"]),
                    "text": row["content"],
                    "metadata": metadata,
                    "distance": round(1 - similarity, 4),
                    "relevance_score": round(similarity, 4),
                })

            logger.info(
                "Supabase vector search completed",
                extra={
                    "results": len(output),
                    "top_score": output[0]["relevance_score"] if output else 0,
                },
            )
            return output

        except Exception as e:
            raise VectorDBError(f"Supabase search failed: {e}")

    def hybrid_search(
        self,
        query_text: str,
        query_embedding: list[float],
        top_k: int = 5,
        semantic_weight: float = 1.0,
        full_text_weight: float = 1.0,
        filters: Optional[dict] = None,
    ) -> list[dict]:
        """Hybrid search combining semantic + full-text via Reciprocal Rank Fusion."""
        try:
            rpc_params = {
                "query_text": query_text,
                "query_embedding": query_embedding,
                "match_count": top_k,
                "semantic_weight": semantic_weight,
                "full_text_weight": full_text_weight,
                "rrf_k": 50,
            }

            if filters:
                if "document_id" in filters:
                    rpc_params["filter_document_id"] = filters["document_id"]
                if "page_number" in filters:
                    rpc_params["filter_page_number"] = filters["page_number"]

            response = self.client.rpc("hybrid_search", rpc_params).execute()

            output = []
            for row in response.data:
                output.append({
                    "id": str(row["id"]),
                    "text": row["content"],
                    "metadata": {
                        "document_id": row.get("document_id", ""),
                        "filename": row.get("file_name", ""),
                        "page_number": row.get("page_number", 0),
                        "section_title": row.get("section_title", ""),
                        "chunk_index": row.get("chunk_index", 0),
                        "content_type": row.get("content_type", "text"),
                    },
                    "distance": 0,
                    "relevance_score": round(row.get("score", 0), 4),
                })

            logger.info(
                "Supabase hybrid search completed",
                extra={"results": len(output)},
            )
            return output

        except Exception as e:
            raise VectorDBError(f"Supabase hybrid search failed: {e}")

    def delete_document(self, document_id: str) -> None:
        """Remove all chunks belonging to a document."""
        try:
            self.client.table(self.table_name).delete().eq(
                "document_id", document_id
            ).execute()
            logger.info("Document deleted from Supabase", extra={"document_id": document_id})
        except Exception as e:
            raise VectorDBError(f"Failed to delete document from Supabase: {e}")

    def get_document_count(self) -> int:
        """Return total number of chunks stored."""
        result = self.client.table(self.table_name).select("id", count="exact").limit(0).execute()
        return result.count or 0

    def upload_file(self, file_bytes: bytes, document_id: str, filename: str, content_type: str) -> str:
        """Upload original file to Supabase Storage. Returns the storage path."""
        storage_path = f"{document_id}/{filename}"
        self.client.storage.from_(self.bucket_name).upload(
            path=storage_path,
            file=file_bytes,
            file_options={
                "content-type": content_type,
                "cache-control": "3600",
                "upsert": "true",
            },
        )
        logger.info("File uploaded to Supabase Storage", extra={"path": storage_path})
        return storage_path

    def get_download_url(self, storage_path: str, expires_in: int = 3600) -> str:
        """Generate a signed download URL for an uploaded file."""
        response = self.client.storage.from_(self.bucket_name).create_signed_url(
            path=storage_path,
            expires_in=expires_in,
        )
        return response["signedURL"]


def get_vector_db(config: Settings) -> VectorDBInterface:
    """Factory function — returns the configured vector DB implementation."""
    if config.VECTOR_DB_TYPE == "supabase":
        return SupabaseVectorService(config)
    if config.VECTOR_DB_TYPE == "pinecone":
        return PineconeService(config)
    return ChromaDBService(config)
