"""Dependency injection — initializes all services at application startup."""

from core.config import Settings
from core.logger import setup_logger
from services.document_service import DocumentService
from services.embedding_service import EmbeddingService
from services.vector_db_service import get_vector_db
from services.llm_service import get_llm_provider
from agents.orchestrator import DocumentAnalysisCrew
from agents.tools.pdf_extractor import PDFExtractorTool
from agents.tools.ocr_tool import OCRTool
from agents.tools.opencv_detector import OpenCVTableDetectorTool
from agents.tools.embedding_tool import EmbeddingIndexTool
from agents.tools.vector_search_tool import VectorSearchTool
from agents.tools.llm_analyzer_tool import LLMAnalyzerTool

logger = setup_logger(__name__)


def init_services(config: Settings) -> dict:
    """Initialize all services. Called once during application lifespan startup."""

    logger.info("Initializing services...")

    # Core services
    document_service = DocumentService(config)
    embedding_service = EmbeddingService(config)
    vector_db = get_vector_db(config)
    llm_provider = get_llm_provider(config)

    # CrewAI tools
    tools = {
        "pdf_extractor": PDFExtractorTool(),
        "ocr_tool": OCRTool(),
        "opencv_detector": OpenCVTableDetectorTool(),
        "embedding_tool": EmbeddingIndexTool(
            embedding_service=embedding_service,
            vector_db_service=vector_db,
        ),
        "vector_search": VectorSearchTool(
            vector_db_service=vector_db,
            embedding_service=embedding_service,
        ),
        "llm_analyzer": LLMAnalyzerTool(llm_service=llm_provider),
    }

    # Orchestrator
    services = {
        "document_service": document_service,
        "embedding_service": embedding_service,
        "vector_db": vector_db,
        "llm_provider": llm_provider,
        "tools": tools,
    }

    orchestrator = DocumentAnalysisCrew(services=services, config=config)
    services["orchestrator"] = orchestrator

    logger.info(
        "All services initialized",
        extra={"vector_db": config.VECTOR_DB_TYPE, "llm": config.LLM_PROVIDER},
    )

    return services
