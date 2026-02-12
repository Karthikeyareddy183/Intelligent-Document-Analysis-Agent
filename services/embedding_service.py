"""Embedding service — dual-provider (OpenAI API / local SentenceTransformer)."""

from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.config import Settings
from core.logger import setup_logger
from core.exceptions import EmbeddingError

logger = setup_logger(__name__)


class EmbeddingService:
    """Handles document chunking and embedding generation.

    Supports two providers:
    - "openai": Uses OpenAI text-embedding-3-small (1536-dim) via API — fast, high quality
    - "local": Uses SentenceTransformer all-MiniLM-L6-v2 (384-dim) on CPU — no API cost
    """

    _local_model_instance = None

    def __init__(self, config: Settings):
        self.config = config
        self.provider = config.EMBEDDING_PROVIDER.lower()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", ", ", " "],
            length_function=len,
        )

        if self.provider == "openai":
            self._init_openai()
        else:
            self._init_local()

    def _init_openai(self):
        """Initialize OpenAI embeddings client."""
        try:
            from openai import OpenAI

            self._openai_client = OpenAI(api_key=self.config.OPENAI_API_KEY)
            self._openai_model = self.config.OPENAI_EMBEDDING_MODEL
            self.dimension = self.config.EMBEDDING_DIMENSION
            logger.info(
                "OpenAI embedding provider initialized",
                extra={"model": self._openai_model, "dimension": self.dimension},
            )
        except Exception as e:
            raise EmbeddingError(f"Failed to initialize OpenAI embeddings: {e}")

    def _init_local(self):
        """Load local SentenceTransformer model (singleton)."""
        try:
            from sentence_transformers import SentenceTransformer

            if EmbeddingService._local_model_instance is None:
                logger.info(
                    "Loading local embedding model",
                    extra={"model": self.config.EMBEDDING_MODEL},
                )
                EmbeddingService._local_model_instance = SentenceTransformer(
                    self.config.EMBEDDING_MODEL
                )
                EmbeddingService._local_model_instance.max_seq_length = 512

            self._local_model = EmbeddingService._local_model_instance
            self.dimension = self._local_model.get_sentence_embedding_dimension()
            logger.info(
                "Local embedding model loaded",
                extra={"dimension": self.dimension},
            )
        except Exception as e:
            raise EmbeddingError(f"Failed to load local embedding model: {e}")

    def chunk_text(self, text: str, metadata: dict) -> list[dict]:
        """Split text into chunks with metadata."""
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
        """Generate embeddings for a batch of texts."""
        if not texts:
            return []

        if self.provider == "openai":
            return self._embed_openai(texts)
        return self._embed_local(texts)

    def generate_query_embedding(self, query: str) -> list[float]:
        """Generate embedding for a single query string."""
        result = self.generate_embeddings([query])
        if not result:
            raise EmbeddingError("Failed to generate query embedding: empty result")
        return result[0]

    def _embed_openai(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings via OpenAI API.

        Batches in groups of 100 to stay well within OpenAI's
        per-request token limit (~8M tokens). Each 750-char chunk
        is ~200 tokens, so 100 chunks ≈ 20K tokens per call — safe.
        """
        try:
            all_embeddings = []
            batch_size = 100

            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                response = self._openai_client.embeddings.create(
                    model=self._openai_model,
                    input=batch,
                )
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
                logger.info(
                    "Embedding batch complete",
                    extra={"batch": i // batch_size + 1, "texts": len(batch), "total_done": len(all_embeddings)},
                )

            return all_embeddings
        except Exception as e:
            raise EmbeddingError(f"OpenAI embedding failed: {e}")

    def _embed_local(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings via local SentenceTransformer."""
        try:
            embeddings = self._local_model.encode(
                texts,
                batch_size=64,
                show_progress_bar=False,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
            return embeddings.tolist()
        except Exception as e:
            raise EmbeddingError(f"Local embedding failed: {e}")
