"""Dependency injection — initializes all services at application startup."""

from core.config import Settings
from core.logger import setup_logger
from services.document_service import DocumentService
from services.embedding_service import EmbeddingService
from services.vector_db_service import get_vector_db
from services.llm_service import get_llm_provider
from services.conversation_service import ConversationService
from agents.orchestrator import DocumentAnalysisCrew

logger = setup_logger(__name__)


def init_services(config: Settings) -> dict:
    """Initialize all services. Called once during application lifespan startup."""

    logger.info("Initializing services...")

    # Core services
    document_service = DocumentService(config)
    embedding_service = EmbeddingService(config)
    vector_db = get_vector_db(config)
    llm_provider = get_llm_provider(config)

    # Conversation service (optional — requires Supabase)
    conversation_service = None
    if config.SUPABASE_URL and config.SUPABASE_SERVICE_ROLE_KEY:
        try:
            conversation_service = ConversationService(config)
            logger.info("ConversationService initialized")
        except Exception as e:
            logger.warning(f"ConversationService not available: {e}")

    # Orchestrator (calls services directly — no CrewAI tools needed at runtime)
    services = {
        "document_service": document_service,
        "embedding_service": embedding_service,
        "vector_db": vector_db,
        "llm_provider": llm_provider,
        "conversation_service": conversation_service,
        "tools": {},  # Kept for backward compat with agent builder methods
    }

    orchestrator = DocumentAnalysisCrew(services=services, config=config)
    services["orchestrator"] = orchestrator

    logger.info(
        "All services initialized",
        extra={
            "vector_db": config.VECTOR_DB_TYPE,
            "llm": config.LLM_PROVIDER,
            "embeddings": config.EMBEDDING_PROVIDER,
        },
    )

    return services
