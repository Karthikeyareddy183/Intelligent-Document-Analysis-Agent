"""Document processing service — handles file I/O and content extraction."""

import os
import uuid
from pathlib import Path
from typing import Optional

import pdfplumber
from PyPDF2 import PdfReader
from PIL import Image
import pytesseract
import cv2
import numpy as np

from core.config import Settings
from core.logger import setup_logger
from core.constants import ContentType, DocumentStatus
from core.exceptions import DocumentProcessingError

logger = setup_logger(__name__)


class DocumentService:
    """Orchestrates document upload, storage, and content extraction."""

    def __init__(self, config: Settings):
        self.upload_dir = Path(config.CHROMA_PERSIST_DIR).parent / "uploads"
        self.processed_dir = Path(config.CHROMA_PERSIST_DIR).parent / "processed"
        self.ocr_language = config.OCR_LANGUAGE
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self._documents: dict[str, dict] = {}

    def save_upload(self, file_content: bytes, filename: str) -> tuple[str, str]:
        """Save uploaded file and return (document_id, file_path)."""
        document_id = str(uuid.uuid4())
        safe_name = f"{document_id}_{filename}"
        file_path = str(self.upload_dir / safe_name)

        with open(file_path, "wb") as f:
            f.write(file_content)

        self._documents[document_id] = {
            "document_id": document_id,
            "filename": filename,
            "file_path": file_path,
            "status": DocumentStatus.PROCESSING,
            "pages": 0,
            "chunks_created": 0,
        }

        logger.info("File saved", extra={"document_id": document_id, "file_name": filename})
        return document_id, file_path

    def get_document_info(self, document_id: str) -> Optional[dict]:
        return self._documents.get(document_id)

    def update_document_status(
        self, document_id: str, status: DocumentStatus, **kwargs
    ):
        if document_id in self._documents:
            self._documents[document_id]["status"] = status
            self._documents[document_id].update(kwargs)

    def list_documents(self) -> list[dict]:
        return list(self._documents.values())

    def extract_pdf_content(self, file_path: str) -> list[dict]:
        """Extract text, tables, and images from a PDF file."""
        pages = []
        try:
            with pdfplumber.open(file_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    page_data = {
                        "page_number": i + 1,
                        "text": page.extract_text() or "",
                        "tables": [],
                        "images": [],
                        "content_type": ContentType.TEXT,
                    }

                    # Extract tables
                    raw_tables = page.extract_tables()
                    if raw_tables:
                        for table in raw_tables:
                            if table and len(table) > 1:
                                headers = table[0] if table[0] else []
                                rows = table[1:]
                                page_data["tables"].append({
                                    "headers": headers,
                                    "rows": rows,
                                })

                    # Extract images metadata
                    if page.images:
                        for img in page.images:
                            page_data["images"].append({
                                "x0": img.get("x0", 0),
                                "y0": img.get("top", 0),
                                "width": img.get("width", 0),
                                "height": img.get("height", 0),
                            })

                    pages.append(page_data)

            logger.info("PDF extracted", extra={"pages": len(pages), "file": file_path})

        except Exception as e:
            raise DocumentProcessingError(
                f"Failed to extract PDF: {e}", detail=file_path
            )

        return pages

    def extract_image_content(self, file_path: str) -> list[dict]:
        """Extract text from an image file using OCR."""
        try:
            image = cv2.imread(file_path)
            if image is None:
                raise DocumentProcessingError(
                    "Could not read image file", detail=file_path
                )

            processed = self._preprocess_for_ocr(image)
            text = pytesseract.image_to_string(
                Image.fromarray(processed),
                lang=self.ocr_language,
                config="--oem 3 --psm 6",
            )

            return [{
                "page_number": 1,
                "text": text.strip(),
                "tables": [],
                "images": [],
                "content_type": ContentType.TEXT,
            }]

        except pytesseract.TesseractNotFoundError:
            raise DocumentProcessingError(
                "Tesseract OCR is not installed or not in PATH",
                detail="Install tesseract-ocr system package",
            )
        except Exception as e:
            raise DocumentProcessingError(
                f"Failed to extract image content: {e}", detail=file_path
            )

    def detect_tables_opencv(self, image_path: str) -> list[dict]:
        """Detect table regions in an image using OpenCV."""
        image = cv2.imread(image_path)
        if image is None:
            return []

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )

        # Detect horizontal lines
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)

        # Detect vertical lines
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)

        # Combine
        table_mask = cv2.add(h_lines, v_lines)
        contours, _ = cv2.findContours(
            table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        tables = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 100 and h > 50:  # Filter small noise
                tables.append({"x": x, "y": y, "width": w, "height": h})

        logger.info("Table detection complete", extra={"tables_found": len(tables)})
        return tables

    def _preprocess_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for better OCR accuracy."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2
        )
        denoised = cv2.medianBlur(binary, 3)
        return denoised
