"""CrewAI tool for table and chart detection using OpenCV."""

from crewai.tools import BaseTool
from pydantic import BaseModel, Field
import cv2
import numpy as np

from core.logger import setup_logger

logger = setup_logger(__name__)


class OpenCVDetectorInput(BaseModel):
    image_path: str = Field(description="Path to the image to analyze for tables/charts")


class OpenCVTableDetectorTool(BaseTool):
    name: str = "table_chart_detector"
    description: str = (
        "Detect tables and chart regions in an image using computer vision. "
        "Returns bounding boxes of detected table/chart areas."
    )
    args_schema: type[BaseModel] = OpenCVDetectorInput

    def _run(self, image_path: str) -> dict:
        logger.info("Detecting tables/charts", extra={"image": image_path})

        try:
            image = cv2.imread(image_path)
            if image is None:
                return {"status": "error", "error": "Could not read image", "tables": [], "charts": []}

            tables = self._detect_tables(image)
            charts = self._detect_charts(image)

            logger.info(
                "Detection complete",
                extra={"tables": len(tables), "charts": len(charts)},
            )

            return {
                "status": "success",
                "tables": tables,
                "charts": charts,
                "image_size": {"width": image.shape[1], "height": image.shape[0]},
            }

        except Exception as e:
            logger.error("Detection failed", extra={"error": str(e)})
            return {"status": "error", "error": str(e), "tables": [], "charts": []}

    def _detect_tables(self, image: np.ndarray) -> list[dict]:
        """Detect table regions by finding grid-like line structures."""
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

        # Combine lines to find table regions
        table_mask = cv2.add(h_lines, v_lines)
        contours, _ = cv2.findContours(
            table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        tables = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 100 and h > 50:
                tables.append({
                    "x": int(x),
                    "y": int(y),
                    "width": int(w),
                    "height": int(h),
                    "area": int(w * h),
                })

        return sorted(tables, key=lambda t: t["area"], reverse=True)

    def _detect_charts(self, image: np.ndarray) -> list[dict]:
        """Detect chart/graph regions using color and shape analysis."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Charts often have colored regions — detect non-white/non-black areas
        lower = np.array([0, 50, 50])
        upper = np.array([180, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)

        # Find large colored regions that might be charts
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
        dilated = cv2.dilate(mask, kernel, iterations=3)
        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        charts = []
        image_area = image.shape[0] * image.shape[1]
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            # Chart should be significant but not the entire image
            if area > image_area * 0.05 and area < image_area * 0.8:
                charts.append({
                    "x": int(x),
                    "y": int(y),
                    "width": int(w),
                    "height": int(h),
                    "area": int(area),
                })

        return sorted(charts, key=lambda c: c["area"], reverse=True)[:5]
