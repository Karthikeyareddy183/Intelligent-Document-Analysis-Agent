"""CrewAI tool for extracting text and tables from PDF files."""

from crewai.tools import BaseTool
from pydantic import BaseModel, Field
import pdfplumber

from core.logger import setup_logger

logger = setup_logger(__name__)


class PDFExtractorInput(BaseModel):
    file_path: str = Field(description="Absolute path to the PDF file")


class PDFExtractorTool(BaseTool):
    name: str = "pdf_extractor"
    description: str = (
        "Extract text and tables from a PDF file page by page. "
        "Returns structured content with page numbers, text blocks, and tables."
    )
    args_schema: type[BaseModel] = PDFExtractorInput

    def _run(self, file_path: str) -> dict:
        logger.info("Extracting PDF", extra={"file": file_path})
        pages = []

        try:
            with pdfplumber.open(file_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text() or ""
                    tables = []

                    raw_tables = page.extract_tables()
                    if raw_tables:
                        for table in raw_tables:
                            if table and len(table) > 1:
                                tables.append({
                                    "headers": [str(h) for h in (table[0] or [])],
                                    "rows": [
                                        [str(cell) if cell else "" for cell in row]
                                        for row in table[1:]
                                    ],
                                })

                    pages.append({
                        "page_number": i + 1,
                        "text": text,
                        "tables": tables,
                        "has_images": len(page.images) > 0,
                        "image_count": len(page.images),
                    })

            logger.info("PDF extraction complete", extra={"pages": len(pages)})
            return {
                "status": "success",
                "total_pages": len(pages),
                "pages": pages,
            }

        except Exception as e:
            logger.error("PDF extraction failed", extra={"error": str(e)})
            return {"status": "error", "error": str(e), "pages": []}
