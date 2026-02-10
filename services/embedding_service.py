"""Embedding service — document chunking and HuggingFace embedding generation."""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np

from core.config import Settings
from core.logger import setup_logger
from core.exceptions import EmbeddingError

logger = setup_logger(__name__)


class EmbeddingService:
    """Handles document chunking and embedding generation."""

    _model_instance: SentenceTransformer = None

    def __init__(self, config: Settings):
        self.config = config
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", ", ", " "],
            length_function=len,
        )
        self._load_model()

    def _load_model(self):
        """Load HuggingFace embedding model (singleton)."""
        if EmbeddingService._model_instance is None:
            logger.info(
                "Loading embedding model",
                extra={"model": self.config.EMBEDDING_MODEL},
            )
            try:
                EmbeddingService._model_instance = SentenceTransformer(
                    self.config.EMBEDDING_MODEL
                )
                EmbeddingService._model_instance.max_seq_length = 512
                logger.info(
                    "Embedding model loaded",
                    extra={
                        "dimension": EmbeddingService._model_instance.get_sentence_embedding_dimension()
                    },
                )
            except Exception as e:
                raise EmbeddingError(f"Failed to load embedding model: {e}")

        self.model = EmbeddingService._model_instance
        self.dimension = self.model.get_sentence_embedding_dimension()

    def chunk_text(self, text: str, metadata: dict) -> list[dict]:
        """Split text into chunks with metadata.

        Args:
            text: Raw text content to chunk.
            metadata: Base metadata to attach to each chunk (document_id, page, etc).

        Returns:
            List of chunk dicts with text and enriched metadata.
        """
        if not text or not text.strip():
            return []

        documents = self.text_splitter.create_documents(
            texts=[text],
            metadatas=[metadata],
        )

        chunks = []
        for i, doc in enumerate(documents):
            chunk_meta = dict(doc.metadata)
            chunk_meta["chunk_index"] = i
            chunk_meta["total_chunks"] = len(documents)
            chunk_meta["char_count"] = len(doc.page_content)
            chunks.append({
                "text": doc.page_content,
                "metadata": chunk_meta,
            })

        logger.info(
            "Text chunked",
            extra={"chunks": len(chunks), "source": metadata.get("document_id", "")},
        )
        return chunks

    def generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of texts.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors (each a list of floats).
        """
        if not texts:
            return []

        try:
            embeddings = self.model.encode(
                texts,
                batch_size=64,
                show_progress_bar=False,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
            return embeddings.tolist()
        except Exception as e:
            raise EmbeddingError(f"Failed to generate embeddings: {e}")

    def generate_query_embedding(self, query: str) -> list[float]:
        """Generate embedding for a single query string."""
        try:
            embedding = self.model.encode(
                query,
                show_progress_bar=False,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
            return embedding.tolist()
        except Exception as e:
            raise EmbeddingError(f"Failed to generate query embedding: {e}")
