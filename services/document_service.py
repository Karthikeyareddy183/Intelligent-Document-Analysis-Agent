"""Document processing service — handles file I/O and content extraction.

Document metadata is persisted in the Supabase `documents` table.
PDF/image extraction methods operate on local file paths (unchanged).
"""

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
        self.upload_dir = Path(config.DATA_DIR) / "uploads"
        self.processed_dir = Path(config.DATA_DIR) / "processed"
        self.ocr_language = config.OCR_LANGUAGE
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        # Initialize Supabase client for document metadata persistence
        self._supabase = None
        if config.SUPABASE_URL and config.SUPABASE_SERVICE_ROLE_KEY:
            try:
                from supabase import create_client
                self._supabase = create_client(config.SUPABASE_URL, config.SUPABASE_SERVICE_ROLE_KEY)
                logger.info("DocumentService connected to Supabase")
            except Exception as e:
                logger.warning(f"Supabase not available for document metadata: {e}")

    # ─── CRUD Operations (Supabase-backed) ────────────────────

    def save_upload(
        self,
        file_content: bytes,
        filename: str,
        user_id: Optional[str] = None,
        file_type: Optional[str] = None,
        file_size: Optional[int] = None,
    ) -> tuple[str, str]:
        """Save uploaded file and persist metadata to Supabase. Returns (document_id, file_path)."""
        document_id = str(uuid.uuid4())
        safe_name = f"{document_id}_{filename}"
        file_path = str(self.upload_dir / safe_name)

        with open(file_path, "wb") as f:
            f.write(file_content)

        if self._supabase and user_id:
            try:
                self._supabase.table("documents").insert({
                    "user_id": user_id,
                    "document_id": document_id,
                    "filename": filename,
                    "file_type": file_type,
                    "file_size": file_size or len(file_content),
                    "storage_path": file_path,
                    "status": "processing",
                }).execute()
            except Exception as e:
                logger.error(f"Failed to insert document metadata: {e}")
                raise DocumentProcessingError(f"Failed to save document metadata: {e}")
        else:
            logger.warning("No Supabase client or user_id — metadata not persisted")

        logger.info("File saved", extra={"document_id": document_id, "file_name": filename})
        return document_id, file_path

    def get_document_info(self, document_id: str) -> Optional[dict]:
        """Fetch document metadata from Supabase."""
        if not self._supabase:
            return None

        try:
            result = self._supabase.table("documents").select("*").eq(
                "document_id", document_id
            ).limit(1).execute()

            if result.data:
                row = result.data[0]
                return {
                    "document_id": row["document_id"],
                    "user_id": row.get("user_id"),
                    "filename": row["filename"],
                    "file_type": row.get("file_type"),
                    "file_size": row.get("file_size"),
                    "storage_path": row.get("storage_path"),
                    "status": row.get("status", "processing"),
                    "pages": row.get("pages", 0),
                    "chunks_created": row.get("chunks_created", 0),
                    "error": row.get("error"),
                    "created_at": row.get("created_at"),
                    "updated_at": row.get("updated_at"),
                }
            return None
        except Exception as e:
            logger.error(f"Failed to fetch document info: {e}")
            return None

    def update_document_status(
        self, document_id: str, status, **kwargs
    ):
        """Update document status and metadata in Supabase."""
        if not self._supabase:
            return

        # Convert DocumentStatus enum to string if needed
        status_str = status.value if hasattr(status, "value") else str(status)

        update_data = {"status": status_str}
        for key in ("pages", "chunks_created", "error"):
            if key in kwargs:
                update_data[key] = kwargs[key]

        try:
            self._supabase.table("documents").update(update_data).eq(
                "document_id", document_id
            ).execute()
        except Exception as e:
            logger.error(f"Failed to update document status: {e}")

    def list_documents(self, user_id: Optional[str] = None) -> list[dict]:
        """List documents, optionally filtered by user_id."""
        if not self._supabase:
            return []

        try:
            query = self._supabase.table("documents").select("*").order(
                "created_at", desc=True
            )
            if user_id:
                query = query.eq("user_id", user_id)

            result = query.execute()

            return [
                {
                    "document_id": row["document_id"],
                    "user_id": row.get("user_id"),
                    "filename": row["filename"],
                    "file_type": row.get("file_type"),
                    "file_size": row.get("file_size"),
                    "status": row.get("status", "processing"),
                    "pages": row.get("pages", 0),
                    "chunks_created": row.get("chunks_created", 0),
                    "error": row.get("error"),
                    "created_at": row.get("created_at"),
                    "updated_at": row.get("updated_at"),
                }
                for row in result.data
            ]
        except Exception as e:
            logger.error(f"Failed to list documents: {e}")
            return []

    def delete_document(self, document_id: str) -> bool:
        """Delete document metadata and local file. Returns True on success."""
        if not self._supabase:
            return False

        try:
            # Get storage path before deleting
            doc = self.get_document_info(document_id)
            if not doc:
                return False

            # Delete from documents table (conversations cascade via document_id FK)
            self._supabase.table("documents").delete().eq(
                "document_id", document_id
            ).execute()

            # Delete local file if it exists
            storage_path = doc.get("storage_path")
            if storage_path and os.path.exists(storage_path):
                os.remove(storage_path)

            logger.info("Document deleted", extra={"document_id": document_id})
            return True
        except Exception as e:
            logger.error(f"Failed to delete document: {e}")
            return False

    # ─── Content Extraction (unchanged) ───────────────────────

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

    def extract_page_images(self, file_path: str) -> dict[int, list[bytes]]:
        """Extract images from each PDF page as PNG bytes.

        Returns a dict mapping page_number (1-based) to a list of image bytes.
        Only extracts images large enough to be meaningful (diagrams, charts, figures).
        """
        page_images: dict[int, list[bytes]] = {}

        try:
            from PyPDF2 import PdfReader
            import io

            reader = PdfReader(file_path)
            for page_idx, page in enumerate(reader.pages):
                page_num = page_idx + 1
                images_on_page = []

                if "/XObject" not in (page.get("/Resources") or {}):
                    continue

                x_objects = page["/Resources"]["/XObject"].get_object()
                for obj_name in x_objects:
                    obj = x_objects[obj_name].get_object()
                    if obj.get("/Subtype") == "/Image":
                        width = obj.get("/Width", 0)
                        height = obj.get("/Height", 0)

                        # Skip tiny images (icons, bullets, logos < 100x100)
                        if width < 100 or height < 100:
                            continue

                        try:
                            data = obj.get_data()
                            # Convert raw image data to PNG via PIL
                            img = Image.open(io.BytesIO(data))
                            buf = io.BytesIO()
                            img.save(buf, format="PNG")
                            images_on_page.append(buf.getvalue())
                        except Exception:
                            # Some image formats can't be decoded — skip silently
                            continue

                if images_on_page:
                    page_images[page_num] = images_on_page

            logger.info(
                "PDF images extracted",
                extra={
                    "pages_with_images": len(page_images),
                    "total_images": sum(len(v) for v in page_images.values()),
                },
            )
        except Exception as e:
            logger.warning(f"Image extraction failed (non-fatal): {e}")

        return page_images

    def _preprocess_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for better OCR accuracy."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2
        )
        denoised = cv2.medianBlur(binary, 3)
        return denoised
