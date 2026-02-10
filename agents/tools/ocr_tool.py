"""CrewAI tool for OCR text extraction from images."""

from crewai.tools import BaseTool
from pydantic import BaseModel, Field
import cv2
import numpy as np
import pytesseract
from PIL import Image

from core.logger import setup_logger

logger = setup_logger(__name__)


class OCRInput(BaseModel):
    image_path: str = Field(description="Path to the image file for OCR")


class OCRTool(BaseTool):
    name: str = "ocr_extractor"
    description: str = (
        "Extract text from an image file using OCR (Optical Character Recognition). "
        "Preprocesses the image for better accuracy. Works with JPG, PNG, WEBP."
    )
    args_schema: type[BaseModel] = OCRInput

    def _run(self, image_path: str) -> dict:
        logger.info("Running OCR", extra={"image": image_path})

        try:
            image = cv2.imread(image_path)
            if image is None:
                return {"status": "error", "error": "Could not read image file", "text": ""}

            processed = self._preprocess(image)
            text = pytesseract.image_to_string(
                Image.fromarray(processed),
                lang="eng",
                config="--oem 3 --psm 6",
            )

            cleaned = text.strip()
            logger.info("OCR complete", extra={"chars": len(cleaned)})

            return {
                "status": "success",
                "text": cleaned,
                "char_count": len(cleaned),
                "word_count": len(cleaned.split()),
            }

        except pytesseract.TesseractNotFoundError:
            logger.error("Tesseract not installed")
            return {
                "status": "error",
                "error": "Tesseract OCR is not installed",
                "text": "",
            }
        except Exception as e:
            logger.error("OCR failed", extra={"error": str(e)})
            return {"status": "error", "error": str(e), "text": ""}

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for better OCR accuracy."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2
        )
        denoised = cv2.medianBlur(binary, 3)
        return denoised
