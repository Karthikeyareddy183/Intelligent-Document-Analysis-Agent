"""Custom exception hierarchy for the application."""


class AppBaseError(Exception):
    """Base exception for all application errors."""

    def __init__(self, message: str, detail: str = ""):
        self.message = message
        self.detail = detail
        super().__init__(self.message)


class DocumentProcessingError(AppBaseError):
    """Raised when document extraction or preprocessing fails."""


class EmbeddingError(AppBaseError):
    """Raised when embedding generation fails."""


class VectorDBError(AppBaseError):
    """Raised when vector database operations fail."""


class LLMError(AppBaseError):
    """Raised when LLM API calls fail."""


class AgentOrchestrationError(AppBaseError):
    """Raised when CrewAI agent orchestration fails."""


class FileValidationError(AppBaseError):
    """Raised when uploaded file fails validation."""
