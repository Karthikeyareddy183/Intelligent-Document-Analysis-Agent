"""CrewAI tool for LLM-based document analysis and answer generation."""

from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Optional

from core.logger import setup_logger
from core.prompts import DOCUMENT_QA_PROMPT

logger = setup_logger(__name__)


class LLMAnalyzerInput(BaseModel):
    question: str = Field(description="User's question to answer")
    context: str = Field(description="Retrieved document context to base answer on")
    images: Optional[list[str]] = Field(
        default=None, description="Optional base64 encoded images for visual analysis"
    )


class LLMAnalyzerTool(BaseTool):
    name: str = "llm_analyzer"
    description: str = (
        "Analyze document context and generate an accurate answer to a question. "
        "Uses a multi-modal LLM to understand both text and visual content. "
        "Returns an answer with confidence score and source citations."
    )
    args_schema: type[BaseModel] = LLMAnalyzerInput

    _llm_service: object = None

    def __init__(self, llm_service, **kwargs):
        super().__init__(**kwargs)
        self._llm_service = llm_service

    def _run(
        self,
        question: str,
        context: str,
        images: Optional[list[str]] = None,
    ) -> dict:
        logger.info("LLM analysis", extra={"question": question[:100]})

        try:
            prompt = DOCUMENT_QA_PROMPT.format(
                context=context,
                question=question,
            )

            # Convert base64 images to bytes if provided
            image_bytes = None
            if images:
                import base64
                image_bytes = [base64.b64decode(img) for img in images]

            # Call LLM
            response = self._llm_service.generate_sync(prompt, image_bytes)

            # Parse confidence from response
            confidence = self._extract_confidence(response.content)

            logger.info(
                "LLM analysis complete",
                extra={"confidence": confidence, "tokens": response.usage},
            )

            return {
                "status": "success",
                "answer": response.content,
                "confidence": confidence,
                "model": response.model,
                "usage": response.usage,
            }

        except Exception as e:
            logger.error("LLM analysis failed", extra={"error": str(e)})
            return {
                "status": "error",
                "error": str(e),
                "answer": "Failed to generate answer due to an internal error.",
                "confidence": 0.0,
            }

    def _extract_confidence(self, text: str) -> float:
        """Extract confidence level from LLM response text."""
        text_lower = text.lower()
        if "high" in text_lower.split("confidence")[-1][:20] if "confidence" in text_lower else "":
            return 0.9
        if "medium" in text_lower.split("confidence")[-1][:20] if "confidence" in text_lower else "":
            return 0.7
        if "low" in text_lower.split("confidence")[-1][:20] if "confidence" in text_lower else "":
            return 0.4

        # Fallback: check last line
        last_line = text.strip().split("\n")[-1].lower()
        if "high" in last_line:
            return 0.9
        elif "medium" in last_line:
            return 0.7
        elif "low" in last_line:
            return 0.4

        return 0.5
