"""
Spatial/Layout Extractor for Procedural Knowledge Graphs

This module handles PDF text extraction with bounding boxes, spatial grouping,
and layout-based inference (column positions, shape types, spatial relations).
"""

import os
import re
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass

try:
    import pdfplumber

    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

try:
    from pdf2image import convert_from_path

    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False

try:
    import pytesseract

    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

try:
    import easyocr

    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

try:
    from paddleocr import PaddleOCR

    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False


@dataclass
class TextBlock:
    """Represents a text block extracted from a PDF page with spatial information."""

    page: int
    line_number: int
    text: str
    raw_text: str
    bbox: Optional[Tuple[float, float, float, float]] = (
        None  # (x0, y0, x1, y1) in PDF points
    )
    centroid: Optional[Tuple[float, float]] = None  # (x, y) center point


@dataclass
class SpatialInfo:
    """Spatial information for a procedural unit."""

    bbox: Optional[Tuple[float, float, float, float]] = None
    centroid: Optional[Tuple[float, float]] = None
    inferred_shape: Optional[str] = None  # "box", "diamond", "triangle"
    column_position: Optional[str] = None  # "left", "center", "right"
    x_norm: Optional[float] = None  # Normalized x position (0.0-1.0)


class SpatialExtractor:
    """
    Extracts text with spatial information from PDFs and performs
    layout-based analysis (grouping, column inference, shape detection).
    """

    def __init__(self, debug: bool = False, output_dir: Optional[str] = None):
        """
        Initialize the spatial extractor.

        Args:
            debug: Enable debug mode with intermediate outputs
            output_dir: Directory for intermediate outputs
        """
        self.debug = debug
        self.output_dir = (
            Path(output_dir) if output_dir else Path("./data/intermediate")
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._page_widths = {}

    def extract_text_blocks(
        self, pdf_path: str
    ) -> Tuple[List[TextBlock], Dict[int, float]]:
        """
        Extract text blocks with bounding boxes from PDF using PaddleOCR only.

        This method is deprecated - spatial extraction now happens in semantic_extractor.
        Returns empty list to maintain compatibility.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Tuple of (List of TextBlock objects, Dict of page_num -> page_width)
        """
        if self.debug:
            print(
                "Spatial extractor: Using PaddleOCR from semantic extractor (no separate extraction)"
            )

        # Return empty - spatial extraction is handled by semantic extractor with PaddleOCR
        return [], {}

    def _extract_text_with_pdfplumber(
        self, pdf_path: str
    ) -> Tuple[List[TextBlock], Dict[int, float]]:
        """
        Extract text with bounding boxes using pdfplumber.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Tuple of (List of TextBlock objects with bounding boxes, Dict of page_num -> page_width)
        """
        text_blocks = []
        line_counter = 0
        page_widths = {}

        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                # Store page width for column inference
                page_widths[page_num] = page.width
                # Extract words with bounding boxes
                words = page.extract_words()

                # Group words into lines based on vertical proximity
                lines = []
                current_line = []
                current_y = None
                y_threshold = 5  # Points

                for word in sorted(words, key=lambda w: (w["top"], w["x0"])):
                    word_y = word["top"]

                    if current_y is None or abs(word_y - current_y) > y_threshold:
                        # New line
                        if current_line:
                            lines.append(current_line)
                        current_line = [word]
                        current_y = word_y
                    else:
                        # Same line
                        current_line.append(word)

                if current_line:
                    lines.append(current_line)

                # Create TextBlock for each line
                for line_words in lines:
                    # Combine text
                    line_text = " ".join([w["text"] for w in line_words])
                    line_text = line_text.strip()

                    if not line_text:
                        continue

                    # Calculate bounding box (union of all words)
                    x0 = min(w["x0"] for w in line_words)
                    y0 = min(w["top"] for w in line_words)
                    x1 = max(w["x1"] for w in line_words)
                    y1 = max(w["bottom"] for w in line_words)

                    # Calculate centroid
                    centroid_x = (x0 + x1) / 2
                    centroid_y = (y0 + y1) / 2

                    line_counter += 1
                    text_blocks.append(
                        TextBlock(
                            page=page_num,
                            line_number=line_counter,
                            text=line_text,
                            raw_text=line_text,
                            bbox=(x0, y0, x1, y1),
                            centroid=(centroid_x, centroid_y),
                        )
                    )

        return text_blocks, page_widths

    def _group_text_blocks_spatially(
        self, text_blocks: List[TextBlock]
    ) -> List[TextBlock]:
        """
        Group text blocks that are spatially close (same flowchart box).

        Args:
            text_blocks: List of TextBlock objects

        Returns:
            List of grouped TextBlock objects (merged text, combined bbox)
        """
        if not text_blocks:
            return text_blocks

        # Group by page first
        page_groups = {}
        for block in text_blocks:
            if block.page not in page_groups:
                page_groups[block.page] = []
            page_groups[block.page].append(block)

        grouped_blocks = []

        for page_num, page_blocks in page_groups.items():
            # Sort by vertical position (top to bottom)
            sorted_blocks = sorted(
                page_blocks, key=lambda b: b.centroid[1] if b.centroid else 0
            )

            current_group = []
            current_bbox = None

            for block in sorted_blocks:
                if block.bbox is None:
                    # No bbox, keep as-is
                    grouped_blocks.append(block)
                    continue

                x0, y0, x1, y1 = block.bbox
                block_centroid_y = (
                    block.centroid[1] if block.centroid else (y0 + y1) / 2
                )

                if not current_group:
                    # Start new group
                    current_group = [block]
                    current_bbox = block.bbox
                else:
                    # Check if block belongs to current group
                    group_x0, group_y0, group_x1, group_y1 = current_bbox
                    group_centroid_y = (group_y0 + group_y1) / 2

                    # Vertical distance threshold (points)
                    vertical_gap = abs(block_centroid_y - group_centroid_y)
                    vertical_threshold = 20  # ~3 lines of text

                    # Horizontal overlap check
                    horizontal_overlap = not (x1 < group_x0 or x0 > group_x1)
                    horizontal_distance = (
                        min(abs(x0 - group_x1), abs(x1 - group_x0))
                        if not horizontal_overlap
                        else 0
                    )
                    horizontal_threshold = 50  # Points

                    # Check if block should be merged
                    should_merge = vertical_gap < vertical_threshold and (
                        horizontal_overlap or horizontal_distance < horizontal_threshold
                    )

                    if should_merge:
                        # Add to current group
                        current_group.append(block)
                        # Update bbox (union)
                        current_bbox = (
                            min(group_x0, x0),
                            min(group_y0, y0),
                            max(group_x1, x1),
                            max(group_y1, y1),
                        )
                    else:
                        # Finalize current group
                        if len(current_group) == 1:
                            grouped_blocks.append(current_group[0])
                        else:
                            # Merge group into single block
                            merged_text = " ".join([b.text for b in current_group])
                            merged_centroid = (
                                (current_bbox[0] + current_bbox[2]) / 2,
                                (current_bbox[1] + current_bbox[3]) / 2,
                            )
                            grouped_blocks.append(
                                TextBlock(
                                    page=current_group[0].page,
                                    line_number=current_group[0].line_number,
                                    text=merged_text,
                                    raw_text=merged_text,
                                    bbox=current_bbox,
                                    centroid=merged_centroid,
                                )
                            )

                        # Start new group
                        current_group = [block]
                        current_bbox = block.bbox

            # Finalize last group
            if current_group:
                if len(current_group) == 1:
                    grouped_blocks.append(current_group[0])
                else:
                    merged_text = " ".join([b.text for b in current_group])
                    merged_centroid = (
                        (current_bbox[0] + current_bbox[2]) / 2,
                        (current_bbox[1] + current_bbox[3]) / 2,
                    )
                    grouped_blocks.append(
                        TextBlock(
                            page=current_group[0].page,
                            line_number=current_group[0].line_number,
                            text=merged_text,
                            raw_text=merged_text,
                            bbox=current_bbox,
                            centroid=merged_centroid,
                        )
                    )

        return grouped_blocks

    def infer_spatial_info(
        self, text_blocks: List[TextBlock], page_widths: Dict[int, float]
    ) -> Dict[str, SpatialInfo]:
        """
        Infer spatial information (shape type, column position) for text blocks.

        Args:
            text_blocks: List of TextBlock objects
            page_widths: Dict mapping page number to page width

        Returns:
            Dict mapping block_id (f"p{page}_l{line}") to SpatialInfo
        """
        spatial_info = {}

        for block in text_blocks:
            block_id = f"p{block.page}_l{block.line_number}"

            if block.centroid and block.page in page_widths:
                page_width = page_widths[block.page]
                x_norm = block.centroid[0] / page_width if page_width > 0 else 0.5

                # Column-based shape inference
                if x_norm < 0.33:
                    column_position = "left"
                    inferred_shape = "box"
                elif 0.33 <= x_norm <= 0.66:
                    column_position = "center"
                    inferred_shape = "diamond"
                elif x_norm > 0.66:
                    column_position = "right"
                    inferred_shape = "box"
                else:
                    column_position = None
                    inferred_shape = None

                spatial_info[block_id] = SpatialInfo(
                    bbox=block.bbox,
                    centroid=block.centroid,
                    inferred_shape=inferred_shape,
                    column_position=column_position,
                    x_norm=x_norm,
                )
            else:
                spatial_info[block_id] = SpatialInfo()

        return spatial_info

    def infer_spatial_relations(
        self,
        units: List[dict],  # List of procedural units with ids and spatial info
    ) -> List[dict]:  # List of relations with spatial evidence
        """
        Infer relations based on spatial proximity and flow direction.

        Args:
            units: List of procedural units (dicts with 'id', 'centroid', 'bbox', etc.)

        Returns:
            List of relation dicts with spatial evidence
        """
        relations = []

        for i, unit in enumerate(units):
            if not unit.get("centroid") or unit.get("type") != "Condition":
                continue

            # Look for nearby actions
            for j, next_unit in enumerate(units[i + 1 : i + 6], start=i + 1):
                if next_unit.get("type") != "Action":
                    continue
                if not next_unit.get("centroid"):
                    continue
                if next_unit.get("page") != unit.get("page"):
                    continue

                # Calculate spatial distance and direction
                dx = next_unit["centroid"][0] - unit["centroid"][0]
                dy = next_unit["centroid"][1] - unit["centroid"][1]
                distance = (dx**2 + dy**2) ** 0.5

                # Flow direction: left to right, top to bottom
                is_flow_direction = (
                    dx > -50
                )  # Allow slight left movement, but prefer right
                is_downward = dy > -20  # Prefer downward flow

                # Spatial proximity threshold
                if distance < 300 and is_flow_direction and is_downward:
                    relations.append(
                        {
                            "source": unit["id"],
                            "target": next_unit["id"],
                            "relation": "leads_to",
                            "confidence": 0.85,
                            "evidence": f"Spatial proximity: {distance:.0f}pt, flow direction matches",
                            "spatial_distance": distance,
                        }
                    )

        return relations
