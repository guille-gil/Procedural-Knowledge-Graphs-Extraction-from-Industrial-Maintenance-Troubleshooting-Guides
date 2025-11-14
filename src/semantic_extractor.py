"""
Semantic Procedural Knowledge Extractor

This module extracts procedural knowledge from industrial troubleshooting guides
using semantic NLP techniques rather than visual layout parsing.
"""

import os
import re
import json
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from dataclasses import dataclass, asdict
import numpy as np

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

try:
    import spacy

    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

try:
    from pdf2image import convert_from_path

    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False

try:
    import pdfplumber

    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

# PyMuPDF (fitz) - optional, not currently used but available for future use
PYMUPDF_AVAILABLE = False
try:
    import pymupdf  # PyMuPDF (fitz)

    PYMUPDF_AVAILABLE = True
except ImportError:
    try:
        import fitz  # Alternative import name

        PYMUPDF_AVAILABLE = True
    except ImportError:
        pass


@dataclass
class TextBlock:
    """Represents a text block extracted from a PDF page."""

    page: int
    line_number: int
    text: str
    raw_text: str
    bbox: Optional[Tuple[float, float, float, float]] = (
        None  # (x0, y0, x1, y1) in PDF points
    )
    centroid: Optional[Tuple[float, float]] = None  # (x, y) center point


@dataclass
class ProceduralUnit:
    """Represents a procedural unit (condition, action, observation, etc.)."""

    id: str
    type: str  # Condition, Action, Observation, Connector
    text: str
    page: int
    line_number: int
    confidence: float = 1.0
    bbox: Optional[Tuple[float, float, float, float]] = None  # (x0, y0, x1, y1)
    centroid: Optional[Tuple[float, float]] = None  # (x, y)
    inferred_shape: Optional[str] = (
        None  # "box", "diamond", "triangle" based on position/content
    )


@dataclass
class Entity:
    """Represents an extracted entity (component, part, etc.)."""

    id: str
    text: str
    type: str  # Component, Part, System
    mentions: List[Tuple[int, int]]  # (page, line_number) tuples
    confidence: float = 1.0


@dataclass
class Relation:
    """Represents a relation between procedural units or entities."""

    source: str
    target: str
    relation_type: str  # leads_to, requires_check, affects, equivalent_to
    confidence: float
    evidence: Optional[str] = None


class SemanticExtractor:
    """
    Extracts procedural knowledge from PDF troubleshooting guides using
    semantic NLP techniques.
    """

    def __init__(
        self,
        ocr_engine: str = "tesseract",
        language: str = "nld",
        debug: bool = False,
        output_dir: Optional[str] = None,
    ):
        """
        Initialize the semantic extractor.

        Args:
            ocr_engine: OCR engine to use ("tesseract", "easyocr", or "paddleocr")
            language: Language code for OCR (default: "nld" for Dutch)
                     Note: Tesseract uses "nld", EasyOCR and PaddleOCR use "nl"
            debug: Enable debug mode
            output_dir: Directory for intermediate outputs
        """
        self.ocr_engine = ocr_engine
        self.language = language
        # Map language codes: Tesseract uses "nld", EasyOCR and PaddleOCR use "nl"
        self._easyocr_lang = "nl" if language == "nld" else language
        self._paddleocr_lang = "nl" if language == "nld" else language
        self.debug = debug
        self.output_dir = (
            Path(output_dir) if output_dir else Path("./data/intermediate")
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize OCR reader
        self._ocr_reader = None
        self._paddleocr_reader = None

        # Auto-detect Tesseract path on macOS (Homebrew installation)
        if ocr_engine == "tesseract" and TESSERACT_AVAILABLE:
            try:
                import shutil
                import pytesseract

                # Try common macOS Homebrew paths
                possible_paths = [
                    "/opt/homebrew/bin/tesseract",  # Apple Silicon
                    "/usr/local/bin/tesseract",  # Intel Mac
                    shutil.which("tesseract"),  # System PATH
                ]
                tesseract_path = None
                for path in possible_paths:
                    if path and os.path.exists(path):
                        tesseract_path = path
                        break

                if tesseract_path:
                    pytesseract.pytesseract.tesseract_cmd = tesseract_path
                    if self.debug:
                        print(f"Auto-detected Tesseract at: {tesseract_path}")
            except Exception as e:
                if self.debug:
                    print(f"Warning: Could not auto-detect Tesseract path: {e}")

        if ocr_engine == "easyocr" and EASYOCR_AVAILABLE:
            try:
                import ssl

                ssl._create_default_https_context = ssl._create_unverified_context
                self._ocr_reader = easyocr.Reader([self._easyocr_lang], gpu=False)
            except Exception as e:
                print(f"Warning: Could not initialize EasyOCR: {e}")
                self.ocr_engine = "tesseract"

        if ocr_engine == "paddleocr":
            if not PADDLEOCR_AVAILABLE:
                raise ImportError(
                    "PaddleOCR not available. Install with: pip install paddlepaddle paddleocr"
                )
            try:
                # PaddleOCR initialization
                # Minimal initialization - only language parameter
                self._paddleocr_reader = PaddleOCR(lang=self._paddleocr_lang)
                if self.debug:
                    print(
                        f"Initialized PaddleOCR with language: {self._paddleocr_lang}"
                    )
            except Exception as e:
                raise RuntimeError(
                    f"Could not initialize PaddleOCR: {e}\n"
                    "Make sure PaddleOCR is installed: pip install paddlepaddle paddleocr"
                )

        # Initialize SpaCy model (lazy loading)
        self._nlp = None

        # Domain glossary for component extraction
        self.component_glossary = self._load_component_glossary()

    def _load_component_glossary(self) -> List[str]:
        """Load domain-specific component glossary from JSON file or use default."""
        glossary_path = (
            Path(__file__).parent.parent / "data" / "glossary" / "components.json"
        )

        if glossary_path.exists():
            try:
                with open(glossary_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(
                    f"Warning: Could not load component glossary from {glossary_path}: {e}"
                )

        # Default glossary fallback
        return [
            "robotas",
            "robot as",
            "vacuumfilter",
            "vacuum filter",
            "cilinder",
            "bretslede",
            "kap",
            "opsteekplaat",
            "vacuümdruk",
            "vacuum druk",
            "demping",
            "sensor",
            "connector",
            "bekabeling",
            "kabel",
            "motor",
            "pomp",
            "ventiel",
            "druk",
            "temperatuur",
            "product",
            "maatvoering",
            "statisch",
            "vervuild",
            "defect",
            "versleten",
            "correct",
            "werkt",
        ]

    def _get_nlp_model(self):
        """Lazy load SpaCy model."""
        if not SPACY_AVAILABLE:
            raise ImportError(
                "SpaCy not available. Install with: pip install spacy\n"
                "Then download Dutch model: python -m spacy download nl_core_news_sm"
            )

        if self._nlp is None:
            try:
                self._nlp = spacy.load("nl_core_news_sm")
            except OSError:
                print("Warning: Dutch SpaCy model not found. Install with:")
                print("  python -m spacy download nl_core_news_sm")
                print("Falling back to basic tokenization.")
                self._nlp = None

        return self._nlp

    def extract_text_blocks(
        self, pdf_path: str, use_pdf_text: bool = True
    ) -> List[TextBlock]:
        """
        Step 1: Extract text blocks from PDF using PaddleOCR only.

        Uses PaddleOCR exclusively for text extraction with bounding boxes.
        Groups text by spatial proximity into logical flowchart boxes.

        Args:
            pdf_path: Path to PDF file
            use_pdf_text: Ignored (kept for compatibility)

        Returns:
            List of TextBlock objects with bounding boxes and centroids
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        # Only use PaddleOCR - no fallbacks
        if self.ocr_engine != "paddleocr":
            raise ValueError(
                f"Only PaddleOCR is supported. Selected engine: {self.ocr_engine}"
            )

        if not PADDLEOCR_AVAILABLE:
            raise ImportError(
                "PaddleOCR not available. Install with: pip install paddlepaddle paddleocr"
            )

        if self._paddleocr_reader is None:
            raise RuntimeError("PaddleOCR reader not initialized")

        if not PDF2IMAGE_AVAILABLE:
            raise ImportError(
                "pdf2image not available. Install with: pip install pdf2image"
            )

        if self.debug:
            print("Using PaddleOCR as the only extraction method")

        # Rasterize PDF pages
        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            images = convert_from_path(pdf_path, dpi=200)

        line_counter = 0
        text_blocks = []
        page_widths = {}  # Store page widths for spatial grouping

        for page_num, image in enumerate(images, 1):
            # Store page width for spatial grouping (convert from pixels to PDF points)
            # Image width in pixels at 200 DPI, convert to PDF points (72 DPI)
            page_widths[page_num] = image.width * (72.0 / 200.0)

            # Extract text using PaddleOCR only
            if not PADDLEOCR_AVAILABLE or self._paddleocr_reader is None:
                raise ImportError("PaddleOCR not available or not initialized")

            # Convert PIL image to numpy array
            img_array = np.array(image)
            results = self._paddleocr_reader.ocr(img_array)

            # PaddleOCR returns format may vary by version
            # Scale factor: image pixels to PDF points (200 DPI)
            scale_factor = 72.0 / 200.0

            if results:
                # Debug: print structure for first result to understand format
                if self.debug and len(results) > 0:
                    result_obj = results[0]
                    print(
                        f"DEBUG: PaddleOCR result type: {type(results)}, first item type: {type(result_obj)}"
                    )
                    # Check if it's an OCRResult object - try to access as dict first
                    if isinstance(result_obj, dict):
                        print(
                            f"DEBUG: Result is dict with keys: {list(result_obj.keys())}"
                        )
                        # Check nested structures
                        for key in result_obj.keys():
                            if key not in ["input_path", "page_index"]:
                                val = result_obj[key]
                                print(
                                    f"DEBUG: Key '{key}' type: {type(val)}, value preview: {str(val)[:200]}"
                                )
                    elif hasattr(result_obj, "__dict__"):
                        print(
                            f"DEBUG: OCRResult attributes: {list(result_obj.__dict__.keys())}"
                        )
                        # Check each attribute
                        for key, val in list(result_obj.__dict__.items())[:5]:
                            print(
                                f"DEBUG: Attr '{key}' type: {type(val)}, value preview: {str(val)[:200]}"
                            )
                    # Try to get all attributes using dir()
                    if hasattr(result_obj, "__getitem__"):
                        try:
                            # Try accessing like a dict
                            if "dt_polys" in result_obj or "rec_text" in result_obj:
                                print("DEBUG: Result has dt_polys or rec_text keys")
                            # List all available keys/attributes
                            if hasattr(result_obj, "keys"):
                                print(
                                    f"DEBUG: Available keys: {list(result_obj.keys())[:20]}"
                                )
                        except:
                            pass
                    # Try dir() to see all attributes
                    try:
                        attrs = [a for a in dir(result_obj) if not a.startswith("_")]
                        print(
                            f"DEBUG: Available attributes (non-private): {attrs[:20]}"
                        )
                    except:
                        pass

                # Handle OCRResult objects (new PaddleOCR format)
                if isinstance(results, list) and len(results) > 0:
                    # Try to access OCRResult as dict-like or object
                    for result_obj in results:
                        try:
                            # Try accessing as dict first
                            if isinstance(result_obj, dict):
                                result_dict = result_obj
                            elif hasattr(result_obj, "__getitem__"):
                                # Try dict-like access
                                try:
                                    result_dict = (
                                        dict(result_obj)
                                        if hasattr(result_obj, "keys")
                                        else result_obj
                                    )
                                except:
                                    # Try accessing attributes
                                    result_dict = (
                                        result_obj.__dict__
                                        if hasattr(result_obj, "__dict__")
                                        else {}
                                    )
                            else:
                                result_dict = (
                                    result_obj.__dict__
                                    if hasattr(result_obj, "__dict__")
                                    else {}
                                )

                            # Extract OCR data from OCRResult object
                            # New PaddleOCR format has separate lists: dt_polys, rec_texts, rec_scores
                            dt_polys = None
                            rec_texts = None
                            rec_scores = None

                            # Try to get dt_polys, rec_texts, rec_scores from result_dict
                            if isinstance(result_dict, dict):
                                dt_polys = result_dict.get("dt_polys")
                                rec_texts = result_dict.get("rec_texts")
                                rec_scores = result_dict.get("rec_scores")
                            elif hasattr(result_obj, "__getitem__"):
                                try:
                                    dt_polys = (
                                        result_obj.get("dt_polys")
                                        if hasattr(result_obj, "get")
                                        else (
                                            result_obj["dt_polys"]
                                            if "dt_polys" in result_obj
                                            else None
                                        )
                                    )
                                    rec_texts = (
                                        result_obj.get("rec_texts")
                                        if hasattr(result_obj, "get")
                                        else (
                                            result_obj["rec_texts"]
                                            if "rec_texts" in result_obj
                                            else None
                                        )
                                    )
                                    rec_scores = (
                                        result_obj.get("rec_scores")
                                        if hasattr(result_obj, "get")
                                        else (
                                            result_obj["rec_scores"]
                                            if "rec_scores" in result_obj
                                            else None
                                        )
                                    )
                                except:
                                    pass

                            # Process the OCR data if we have the required fields
                            if dt_polys is not None and rec_texts is not None:
                                # Ensure they are lists
                                if not isinstance(dt_polys, list):
                                    dt_polys = (
                                        [dt_polys] if dt_polys is not None else []
                                    )
                                if not isinstance(rec_texts, list):
                                    rec_texts = (
                                        [rec_texts] if rec_texts is not None else []
                                    )
                                if rec_scores is None or not isinstance(
                                    rec_scores, list
                                ):
                                    rec_scores = [1.0] * len(rec_texts)

                                # Zip them together and process
                                for i, (bbox_poly, text) in enumerate(
                                    zip(dt_polys, rec_texts)
                                ):
                                    if not text or not text.strip():
                                        continue

                                    try:
                                        # Convert numpy array to list if needed
                                        if hasattr(bbox_poly, "tolist"):
                                            bbox_coords = bbox_poly.tolist()
                                        elif isinstance(bbox_poly, list):
                                            bbox_coords = bbox_poly
                                        else:
                                            continue

                                        # Get confidence score
                                        confidence = (
                                            rec_scores[i]
                                            if i < len(rec_scores)
                                            else 1.0
                                        )

                                        # Process bounding box
                                        if (
                                            isinstance(bbox_coords, list)
                                            and len(bbox_coords) > 0
                                        ):
                                            # Handle polygon format (4 points)
                                            if (
                                                isinstance(
                                                    bbox_coords[0],
                                                    (list, tuple, np.ndarray),
                                                )
                                                and len(bbox_coords[0]) == 2
                                            ):
                                                # Convert numpy arrays to lists
                                                x_coords = [
                                                    float(pt[0])
                                                    if hasattr(pt, "__getitem__")
                                                    else float(pt)
                                                    for pt in bbox_coords
                                                ]
                                                y_coords = [
                                                    float(pt[1])
                                                    if hasattr(pt, "__getitem__")
                                                    else float(pt)
                                                    for pt in bbox_coords
                                                ]

                                                x0 = min(x_coords) * scale_factor
                                                y0 = min(y_coords) * scale_factor
                                                x1 = max(x_coords) * scale_factor
                                                y1 = max(y_coords) * scale_factor
                                                centroid_x = (x0 + x1) / 2
                                                centroid_y = (y0 + y1) / 2

                                                cleaned_text = self._clean_ocr_text(
                                                    text.strip()
                                                )
                                                if cleaned_text:
                                                    line_counter += 1
                                                    text_blocks.append(
                                                        TextBlock(
                                                            page=page_num,
                                                            line_number=line_counter,
                                                            text=cleaned_text,
                                                            raw_text=text.strip(),
                                                            bbox=(x0, y0, x1, y1),
                                                            centroid=(
                                                                centroid_x,
                                                                centroid_y,
                                                            ),
                                                        )
                                                    )
                                    except Exception as e:
                                        if self.debug:
                                            print(
                                                f"DEBUG: Error processing OCR item {i}: {e}"
                                            )
                                        continue

                        except Exception as e:
                            if self.debug:
                                print(
                                    f"DEBUG: Error processing OCRResult: {e}, type: {type(result_obj)}"
                                )
                            continue

                    # Check if we extracted anything from OCRResult objects
                    # If not, try old format handlers below
                    if (
                        len(text_blocks) == 0
                        and isinstance(results, list)
                        and len(results) > 0
                    ):
                        # Old format handlers (if OCRResult extraction didn't work)
                        # Check if it's the new format (list of dicts)
                        if isinstance(results[0], dict):
                            for result_dict in results:
                                if (
                                    "dt_polys" in result_dict
                                    and "rec_text" in result_dict
                                ):
                                    bbox_coords = result_dict["dt_polys"]
                                    text = result_dict.get("rec_text", "")
                                    confidence = result_dict.get("rec_score", 0.0)

                                    if text and text.strip():
                                        # Convert bbox from image pixels to PDF points
                                        if (
                                            isinstance(bbox_coords, list)
                                            and len(bbox_coords) > 0
                                        ):
                                            # Handle polygon format
                                            if (
                                                isinstance(
                                                    bbox_coords[0], (list, tuple)
                                                )
                                                and len(bbox_coords[0]) == 2
                                            ):
                                                x_coords = [pt[0] for pt in bbox_coords]
                                                y_coords = [pt[1] for pt in bbox_coords]
                                            else:
                                                # Single bbox format
                                                x_coords = [
                                                    bbox_coords[0][0],
                                                    bbox_coords[1][0],
                                                    bbox_coords[2][0],
                                                    bbox_coords[3][0],
                                                ]
                                                y_coords = [
                                                    bbox_coords[0][1],
                                                    bbox_coords[1][1],
                                                    bbox_coords[2][1],
                                                    bbox_coords[3][1],
                                                ]
                                        else:
                                            continue

                                        x0 = min(x_coords) * scale_factor
                                        y0 = min(y_coords) * scale_factor
                                        x1 = max(x_coords) * scale_factor
                                        y1 = max(y_coords) * scale_factor
                                        centroid_x = (x0 + x1) / 2
                                        centroid_y = (y0 + y1) / 2

                                        cleaned_text = self._clean_ocr_text(
                                            text.strip()
                                        )
                                        if cleaned_text:
                                            line_counter += 1
                                            text_blocks.append(
                                                TextBlock(
                                                    page=page_num,
                                                    line_number=line_counter,
                                                    text=cleaned_text,
                                                    raw_text=text.strip(),
                                                    bbox=(x0, y0, x1, y1),
                                                    centroid=(centroid_x, centroid_y),
                                                )
                                            )
                    # Handle old format: [[[[x1,y1], ...], (text, confidence)], ...]
                    # Or nested format: results[0] is a list of results
                    elif isinstance(results[0], list):
                        # Flatten if needed - results might be [[result1, result2, ...]]
                        flat_results = (
                            results[0] if isinstance(results[0][0], list) else results
                        )

                        for line_result in (
                            flat_results
                            if isinstance(flat_results[0], list)
                            else results[0]
                        ):
                            if not line_result:
                                continue
                            try:
                                # Try to extract bbox and text from various formats
                                bbox_coords = None
                                text = None
                                confidence = 1.0

                                # Format 1: [[[x,y], ...], (text, conf)]
                                if (
                                    isinstance(line_result, list)
                                    and len(line_result) >= 2
                                ):
                                    bbox_coords = line_result[0]
                                    text_conf = line_result[1]
                                    if (
                                        isinstance(text_conf, tuple)
                                        and len(text_conf) >= 1
                                    ):
                                        text = text_conf[0]
                                        confidence = (
                                            text_conf[1] if len(text_conf) > 1 else 1.0
                                        )
                                    elif isinstance(text_conf, str):
                                        text = text_conf
                                    else:
                                        continue
                                # Format 2: (bbox, text, conf) or similar
                                elif (
                                    isinstance(line_result, tuple)
                                    and len(line_result) >= 2
                                ):
                                    bbox_coords = line_result[0]
                                    if len(line_result) >= 2:
                                        text = (
                                            line_result[1]
                                            if isinstance(line_result[1], str)
                                            else str(line_result[1])
                                        )
                                    if len(line_result) >= 3:
                                        confidence = (
                                            line_result[2]
                                            if isinstance(line_result[2], (int, float))
                                            else 1.0
                                        )
                                else:
                                    continue

                                if not bbox_coords or not text:
                                    continue

                            except (ValueError, TypeError, IndexError) as e:
                                if self.debug:
                                    print(
                                        f"DEBUG: Skipping result due to format error: {e}, result: {str(line_result)[:100]}"
                                    )
                                continue

                            if text and text.strip() and bbox_coords:
                                try:
                                    # Convert bbox from image pixels to PDF points
                                    if (
                                        isinstance(bbox_coords, list)
                                        and len(bbox_coords) > 0
                                    ):
                                        # Handle polygon format [[x1,y1], [x2,y2], ...]
                                        if (
                                            isinstance(bbox_coords[0], (list, tuple))
                                            and len(bbox_coords[0]) == 2
                                        ):
                                            x_coords = [pt[0] for pt in bbox_coords]
                                            y_coords = [pt[1] for pt in bbox_coords]
                                        else:
                                            continue
                                    else:
                                        continue

                                    x0 = min(x_coords) * scale_factor
                                    y0 = min(y_coords) * scale_factor
                                    x1 = max(x_coords) * scale_factor
                                    y1 = max(y_coords) * scale_factor
                                    centroid_x = (x0 + x1) / 2
                                    centroid_y = (y0 + y1) / 2

                                    cleaned_text = self._clean_ocr_text(text.strip())
                                    if cleaned_text:
                                        line_counter += 1
                                        text_blocks.append(
                                            TextBlock(
                                                page=page_num,
                                                line_number=line_counter,
                                                text=cleaned_text,
                                                raw_text=text.strip(),
                                                bbox=(x0, y0, x1, y1),
                                                centroid=(centroid_x, centroid_y),
                                            )
                                        )
                                except (ValueError, TypeError, IndexError) as e:
                                    if self.debug:
                                        print(f"DEBUG: Error processing bbox: {e}")
                                    continue
            else:
                # No results from PaddleOCR
                if self.debug:
                    print(f"Warning: PaddleOCR returned no results for page {page_num}")

        # Save intermediate output
        if self.debug:
            output_file = self.output_dir / f"ocr_extraction_{Path(pdf_path).stem}.json"
            output_data = [
                {
                    "page": block.page,
                    "line_number": block.line_number,
                    "text": block.text,
                }
                for block in text_blocks
            ]
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            print(f"OCR extraction saved to: {output_file}")

        # Extract ja/nee branch labels BEFORE merging (they should be separate connectors)
        text_blocks = self._extract_branch_labels(text_blocks)

        # Merge fragmented text lines before processing
        text_blocks = self._merge_fragmented_lines(text_blocks)

        # Clean OCR garbage from merged text blocks (second pass after merging)
        cleaned_blocks = []
        for block in text_blocks:
            cleaned_text = self._clean_ocr_text(block.text)
            if cleaned_text:  # Only keep non-empty blocks
                cleaned_blocks.append(
                    TextBlock(
                        page=block.page,
                        line_number=block.line_number,
                        text=cleaned_text,
                        raw_text=block.raw_text,
                        bbox=block.bbox,
                        centroid=block.centroid,
                    )
                )

        # Filter out metadata, headers, footers, and non-relevant content
        filtered_blocks = self._filter_metadata(cleaned_blocks)

        # Group text blocks spatially (PaddleOCR provides bounding boxes)
        # This merges text that's in the same flowchart box
        if filtered_blocks and any(block.bbox for block in filtered_blocks):
            if self.debug:
                print(
                    f"Grouping {len(filtered_blocks)} PaddleOCR text blocks spatially..."
                )
            filtered_blocks = self._group_text_blocks_spatially(filtered_blocks)
            # Store page widths for later use in classification
            self._page_widths = page_widths
            if self.debug:
                print(f"After spatial grouping: {len(filtered_blocks)} text blocks")

        return filtered_blocks

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

        Uses simple proximity-based grouping:
        - Blocks on same page
        - Vertically close (within threshold)
        - Horizontally overlapping or close

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
                    # Get current group's bbox
                    group_x0, group_y0, group_x1, group_y1 = current_bbox
                    group_centroid_y = (group_y0 + group_y1) / 2

                    # Vertical distance threshold (points) - more conservative
                    vertical_gap = abs(block_centroid_y - group_centroid_y)
                    vertical_threshold = 15  # ~2 lines of text (more conservative)

                    # Horizontal overlap check
                    horizontal_overlap = not (x1 < group_x0 or x0 > group_x1)
                    horizontal_distance = (
                        min(abs(x0 - group_x1), abs(x1 - group_x0))
                        if not horizontal_overlap
                        else 0
                    )
                    horizontal_threshold = 30  # Points (more conservative)

                    # Additional check: Don't merge if blocks are in different columns
                    # Estimate page width from bboxes
                    estimated_page_width = max(x1, group_x1) * 1.1
                    block_x_norm = (
                        block.centroid[0] / estimated_page_width
                        if block.centroid and estimated_page_width > 0
                        else 0.5
                    )
                    group_center_x = (group_x0 + group_x1) / 2
                    group_x_norm = (
                        group_center_x / estimated_page_width
                        if estimated_page_width > 0
                        else 0.5
                    )
                    column_diff = abs(block_x_norm - group_x_norm)

                    # Check if block should be merged
                    # More conservative: require both vertical proximity AND horizontal overlap
                    should_merge = (
                        vertical_gap < vertical_threshold
                        and horizontal_overlap  # Require actual overlap, not just proximity
                        and column_diff < 0.3  # Same column
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

    def _clean_ocr_text(self, text: str) -> str:
        """
        Clean common OCR errors and noise from text.

        Args:
            text: Raw OCR text

        Returns:
            Cleaned text
        """

        # Fix repeated character patterns (OCR error where each char is repeated)
        # Pattern: "IIInnnhhhooouuudddsssooopppgggaaavvveee" -> "Inhoudsopgave"
        def fix_repeated_chars(match):
            chars = match.group(0)
            # Extract unique characters in order
            seen = set()
            fixed = []
            for char in chars:
                if char.lower() not in seen:
                    seen.add(char.lower())
                    fixed.append(char)
            return "".join(fixed)

        # Match sequences of 3+ repeated characters (case-insensitive)
        text = re.sub(r"([A-Za-z])\1{2,}", r"\1", text)

        # Dictionary-based corrections for common Dutch terms
        ocr_corrections = {
            r"IIInnnhhhooouuudddsssooopppgggaaavvveee": "Inhoudsopgave",
            r"IInnhhoouuddssooppggaavvee": "Inhoudsopgave",
            r"Innnhhhooouuudddsssooopppgggaaavvveee": "Inhoudsopgave",
        }
        for pattern, replacement in ocr_corrections.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        # Remove leading OCR noise patterns (e.g., "NS _", "LL me:", "PS 3a)")
        # But keep if followed by meaningful content
        text = re.sub(r"^[A-Z]{1,3}\s+[A-Z]{1,3}[:]?\s*", "", text)
        text = re.sub(r"^[A-Z]{1,3}\s+_\s*", "", text)

        # Remove known OCR garbage words at start
        text = re.sub(
            r"^(Benieetehen|ghressto|LL me|PS|NS|Ned|LG|TO|ZZZ|CAT|OON|Dee|ES|SS|DS)\s+",
            "",
            text,
            flags=re.IGNORECASE,
        )

        # Remove OCR garbage words anywhere in text (but preserve context)
        # Remove standalone garbage markers (with word boundaries to avoid removing parts of words)
        text = re.sub(r"\b(ES|SS|DS|CAT|OON|Dee)\s+", " ", text, flags=re.IGNORECASE)
        text = re.sub(r"\s+(ES|SS|DS|CAT|OON|Dee)\s+", " ", text, flags=re.IGNORECASE)
        text = re.sub(r"\s+lijn er\s+", " ", text, flags=re.IGNORECASE)
        # Remove "Dee" when it appears before other text
        text = re.sub(r"\bDee\s+", "", text, flags=re.IGNORECASE)

        # Clean up common OCR character errors
        text = re.sub(
            r"Processpecicafiehlad", "Processpecificatieblad", text, flags=re.IGNORECASE
        )
        text = re.sub(
            r"Processpee\s+eee", "Processpecificatieblad", text, flags=re.IGNORECASE
        )

        # Fix spacing errors (e.g., "o n derdelen" -> "onderdelen")
        text = re.sub(r"\b([a-z])\s+([a-z])\b", r"\1\2", text, flags=re.IGNORECASE)

        # Remove excessive whitespace and special characters
        text = re.sub(r"\s*[/\\]\s*", " ", text)  # Remove isolated slashes
        text = re.sub(r"\s+", " ", text)

        return text.strip()

    def _filter_metadata(self, text_blocks: List[TextBlock]) -> List[TextBlock]:
        """
        Filter out header, footer, and metadata content that is not part of the
        procedural workflow.

        Removes:
        - Header fields: "Naam:", "Periode:", "Unit: DPXX"
        - Table of contents: "Inhoudsopgave"
        - Status markers: "uitgevoerd" (when standalone)
        - Footer content: "Opmerkingen", "Laatste wijzigingen", "Auteur:", "Versiedatum:"
        - Author names and dates

        Args:
            text_blocks: List of text blocks to filter

        Returns:
            Filtered list of text blocks
        """
        if not text_blocks:
            return text_blocks

        # Exact metadata strings to filter (case-insensitive)
        exact_metadata_strings = [
            "naam:",
            "periode:",
            "unit: dpxx",
            "unit:",
            "inhoudsopgave",
            "uitgevoerd",
            "opmerkingen",
            "laatste wijzigingen",
            "laatste wijzigingen tsg",
            "auteur:",
            "versiedatum:",
            "start tocap",
        ]

        # Patterns for metadata to filter out (more flexible - partial matches)
        metadata_patterns = [
            # Header fields (flexible - matches anywhere in line)
            r"Naam:\s*Periode:",
            r"Periode:\s*Unit:",
            r"Unit:\s*DPXX",
            r"Unit:\s*DP\d+",
            r"Naam:\s*Periode:\s*Unit:\s*DPXX",
            # Table of contents (flexible)
            r"Inhoudsopgave",
            # Status markers (standalone or combined)
            r"^uitgevoerd\s*$",
            r"Start\s+Tocap\s+\d+\s+uitgevoerd",
            r"uitgevoerd\s+Start",
            # Footer fields (flexible)
            r"Opmerkingen",
            r"Laatste\s+wijzigingen",
            r"Laatste\s+wijzigingen\s+TSG",
            r"Auteur:",
            r"Versiedatum:",
            r"Versiedatum:\s*\d{2}-\d{2}-\d{4}",
            # Author names (common Dutch names pattern)
            r"(Robert|Nanco|Auke)\s+(Keuning|Nijdam|Dootjes)",
            # Date patterns
            r"^\d{2}-\d{2}-\d{4}\s*$",
            # Lines that are mostly metadata keywords
            r"^(Naam|Periode|Unit|Inhoudsopgave|Opmerkingen|Auteur|Versiedatum)[:\s]",
        ]

        # Keywords that indicate metadata (if line is mostly just this keyword)
        metadata_keywords = {
            "naam",
            "periode",
            "inhoudsopgave",
            "opmerkingen",
            "auteur",
            "versiedatum",
            "uitgevoerd",
            "unit",
        }

        filtered = []
        for block in text_blocks:
            text = block.text.strip()
            text_lower = text.lower().strip()

            # Skip empty or very short lines that are likely metadata
            if len(text) < 3:
                continue

            # FIRST: Check exact matches (most reliable)
            is_metadata = False
            for exact_string in exact_metadata_strings:
                if (
                    text_lower == exact_string
                    or text_lower.startswith(exact_string + " ")
                    or text_lower == exact_string.rstrip(":")
                ):
                    is_metadata = True
                    break
                # Also check if line is just the keyword with optional colon/whitespace
                if (
                    text_lower.replace(":", "").strip()
                    == exact_string.replace(":", "").strip()
                    and len(text_lower.split()) <= 2
                ):
                    is_metadata = True
                    break

            # SECOND: Check against metadata patterns (more flexible - partial matches)
            if not is_metadata:
                for pattern in metadata_patterns:
                    if re.search(pattern, text, re.IGNORECASE):
                        is_metadata = True
                        break

            # Check if line contains multiple metadata keywords (strong indicator)
            if not is_metadata:
                metadata_keyword_count = sum(
                    1 for keyword in metadata_keywords if keyword in text_lower
                )
                if metadata_keyword_count >= 2:
                    is_metadata = True
                # Also check if line starts with metadata keywords (even if only one)
                words = text_lower.split()
                if metadata_keyword_count >= 1 and len(words) <= 6:
                    # Short line with metadata keyword is likely metadata
                    if words[0] in metadata_keywords or any(
                        keyword in text_lower[:20] for keyword in metadata_keywords
                    ):
                        is_metadata = True

            # Check if line is mostly just a metadata keyword
            if not is_metadata:
                words = text_lower.split()
                # Check if first word is a metadata keyword (with or without colon)
                first_word = words[0].rstrip(":").lower() if words else ""
                if first_word in metadata_keywords:
                    is_metadata = True
                elif len(words) <= 2:
                    # Very short line - check if it's a metadata keyword
                    if any(keyword in text_lower for keyword in metadata_keywords):
                        is_metadata = True
                elif len(words) <= 4:
                    # Short line - check if it starts with metadata keyword
                    if words[0].rstrip(":").lower() in metadata_keywords:
                        is_metadata = True

            # Check if line is in header/footer region (top/bottom 10% of page)
            if not is_metadata and block.bbox:
                # This check would need page height, but we can approximate
                # For now, skip very short lines at start/end of page
                if block.line_number <= 3 or block.line_number >= len(text_blocks) - 3:
                    if len(text.strip()) < 20 and any(
                        keyword in text_lower for keyword in metadata_keywords
                    ):
                        is_metadata = True

            # Skip if it's metadata
            if is_metadata:
                continue

            # Additional check: Skip lines that are just author names or dates
            # (common patterns in footers)
            if re.match(r"^[A-Z][a-z]+\s+[A-Z][a-z]+(\s+[A-Z][a-z]+)?\s*$", text):
                # Looks like a name (2-3 capitalized words)
                if any(
                    name_part in text_lower
                    for name_part in [
                        "keuning",
                        "nijdam",
                        "dootjes",
                        "robert",
                        "nanco",
                        "auke",
                    ]
                ):
                    continue

            # Clean metadata prefixes/suffixes from lines that contain actual content
            # Remove "uitgevoerd" prefix
            text = re.sub(r"^uitgevoerd\s+", "", text, flags=re.IGNORECASE)
            # Remove "Opmerkingen" suffix
            text = re.sub(r"\s+Opmerkingen\s*$", "", text, flags=re.IGNORECASE)
            # Remove "Start Tocap X uitgevoerd" prefix
            text = re.sub(
                r"^Start\s+Tocap\s+\d+\s+uitgevoerd\s+", "", text, flags=re.IGNORECASE
            )

            # Update block text if cleaned
            if text != block.text:
                block = TextBlock(
                    page=block.page,
                    line_number=block.line_number,
                    text=text.strip(),
                    raw_text=block.raw_text,
                )

            # Skip if text becomes too short after cleaning
            if len(text.strip()) < 3:
                continue

            # Keep the block
            filtered.append(block)

        return filtered

    def _extract_branch_labels(self, text_blocks: List[TextBlock]) -> List[TextBlock]:
        """
        Extract "ja"/"nee" branch labels that appear anywhere in lines.
        These should be separate connector nodes, not merged with content.

        Args:
            text_blocks: List of text blocks

        Returns:
            List of text blocks with branch labels extracted as separate blocks
        """
        if not text_blocks:
            return text_blocks

        extracted_blocks = []

        for block in text_blocks:
            text = block.text.strip()
            original_text = text

            # Case 1: Line starts with "ja" or "nee" (with comma, space, or followed by capital/number)
            # Pattern: "ja,", "nee,", "ja ", "nee ", or "ja"/"nee" followed by capital/number/digit
            branch_match = re.match(r"^(ja|nee)[,\s]+", text, re.IGNORECASE)
            if branch_match:
                branch_label = branch_match.group(1).lower()
                # Extract remaining text (skip the branch label, comma, and optional space)
                match_end = branch_match.end(0)
                remaining_text = text[match_end:].strip()

                # Create separate connector block
                extracted_blocks.append(
                    TextBlock(
                        page=block.page,
                        line_number=block.line_number - 0.1,
                        text=branch_label,
                        raw_text=branch_label,
                        bbox=block.bbox,
                        centroid=block.centroid,
                    )
                )

                # Create new block with remaining text
                if remaining_text and len(remaining_text) > 2:
                    extracted_blocks.append(
                        TextBlock(
                            page=block.page,
                            line_number=block.line_number,
                            text=remaining_text,
                            raw_text=remaining_text,
                            bbox=block.bbox,
                            centroid=block.centroid,
                        )
                    )
                continue

            # Case 2: "ja" or "nee" at the end of a line (before punctuation or at end)
            # Pattern: text ending with " ja" or " nee" or " ja?" or " nee?"
            end_match = re.search(r"\s+(ja|nee)[\s\.\?\!]*$", text, re.IGNORECASE)
            if end_match:
                branch_label = end_match.group(1).lower()
                text_before = text[: end_match.start()].strip()

                if text_before and len(text_before) > 3:
                    # Create block for text before
                    extracted_blocks.append(
                        TextBlock(
                            page=block.page,
                            line_number=block.line_number - 0.1,
                            text=text_before,
                            raw_text=text_before,
                            bbox=block.bbox,
                            centroid=block.centroid,
                        )
                    )
                    # Create connector block
                    extracted_blocks.append(
                        TextBlock(
                            page=block.page,
                            line_number=block.line_number,
                            text=branch_label,
                            raw_text=branch_label,
                            bbox=block.bbox,
                            centroid=block.centroid,
                        )
                    )
                    continue

            # Case 3: "ja" or "nee" in the middle with spaces
            # Pattern: " ja " or " nee " (word boundaries)
            parts = re.split(r"\s+(ja|nee)\s+", text, maxsplit=1, flags=re.IGNORECASE)
            if len(parts) == 3:
                text_before = parts[0].strip()
                branch_label = parts[1].lower()
                text_after = parts[2].strip()

                # Split if both parts have meaningful content
                if (
                    text_before
                    and text_after
                    and len(text_before) > 3
                    and len(text_after) > 3
                ):
                    extracted_blocks.append(
                        TextBlock(
                            page=block.page,
                            line_number=block.line_number - 0.1,
                            text=text_before,
                            raw_text=text_before,
                            bbox=block.bbox,
                            centroid=block.centroid,
                        )
                    )
                    extracted_blocks.append(
                        TextBlock(
                            page=block.page,
                            line_number=block.line_number,
                            text=branch_label,
                            raw_text=branch_label,
                            bbox=block.bbox,
                            centroid=block.centroid,
                        )
                    )
                    extracted_blocks.append(
                        TextBlock(
                            page=block.page,
                            line_number=block.line_number + 0.1,
                            text=text_after,
                            raw_text=text_after,
                            bbox=block.bbox,
                            centroid=block.centroid,
                        )
                    )
                    continue

            # Case 4: "ja" or "nee" followed by action text (e.g., "nee B7)", "nee5a)")
            # Pattern: " ja" or " nee" followed by capital/number (with or without space)
            # Also handle at start: "nee B7)" or "nee5a)"
            no_space_match = re.search(
                r"(?:^|\s+)(ja|nee)\s*([A-Z]\d+\)|[A-Z0-9])", text, re.IGNORECASE
            )
            if no_space_match:
                branch_label = no_space_match.group(1).lower()
                match_start = no_space_match.start(1)  # Start of "ja"/"nee"
                text_before = text[:match_start].strip()
                # Extract text after "ja"/"nee" (including the action code)
                text_after_start = no_space_match.end(1)  # After "ja"/"nee"
                text_after = text[text_after_start:].strip()

                # If "ja"/"nee" is at the start, text_before will be empty
                if text_before and len(text_before) > 3:
                    extracted_blocks.append(
                        TextBlock(
                            page=block.page,
                            line_number=block.line_number - 0.1,
                            text=text_before,
                            raw_text=text_before,
                            bbox=block.bbox,
                            centroid=block.centroid,
                        )
                    )

                # Always extract the connector
                extracted_blocks.append(
                    TextBlock(
                        page=block.page,
                        line_number=block.line_number,
                        text=branch_label,
                        raw_text=branch_label,
                        bbox=block.bbox,
                        centroid=block.centroid,
                    )
                )

                # Extract text after if meaningful
                if text_after and len(text_after) > 2:
                    extracted_blocks.append(
                        TextBlock(
                            page=block.page,
                            line_number=block.line_number + 0.1,
                            text=text_after,
                            raw_text=text_after,
                            bbox=block.bbox,
                            centroid=block.centroid,
                        )
                    )
                continue

            # Case 5: Multiple "ja"/"nee" in one line - extract all
            # Find all occurrences
            all_matches = list(re.finditer(r"\b(ja|nee)\b", text, re.IGNORECASE))
            if len(all_matches) > 1:
                # Multiple connectors - split on each
                current_pos = 0
                for i, match in enumerate(all_matches):
                    # Text before this connector
                    if match.start() > current_pos:
                        text_segment = text[current_pos : match.start()].strip()
                        if text_segment and len(text_segment) > 3:
                            extracted_blocks.append(
                                TextBlock(
                                    page=block.page,
                                    line_number=block.line_number + (i * 0.1),
                                    text=text_segment,
                                    raw_text=text_segment,
                                    bbox=block.bbox,
                                    centroid=block.centroid,
                                )
                            )

                    # The connector itself
                    extracted_blocks.append(
                        TextBlock(
                            page=block.page,
                            line_number=block.line_number + (i * 0.1) + 0.05,
                            text=match.group(1).lower(),
                            raw_text=match.group(1).lower(),
                            bbox=block.bbox,
                            centroid=block.centroid,
                        )
                    )
                    current_pos = match.end()

                # Text after last connector
                if current_pos < len(text):
                    text_segment = text[current_pos:].strip()
                    if text_segment and len(text_segment) > 3:
                        extracted_blocks.append(
                            TextBlock(
                                page=block.page,
                                line_number=block.line_number
                                + (len(all_matches) * 0.1),
                                text=text_segment,
                                raw_text=text_segment,
                                bbox=block.bbox,
                                centroid=block.centroid,
                            )
                        )
                continue

            # No branch label found - keep original
            extracted_blocks.append(block)

        # Sort by line number to maintain order (handles fractional line numbers)
        extracted_blocks.sort(key=lambda b: (b.page, b.line_number))

        # Renumber to fix fractional line numbers and maintain sequence
        renumbered_blocks = []
        current_line = 0
        last_page = None

        for block in extracted_blocks:
            if block.page != last_page:
                current_line = 0  # Reset line counter for new page
                last_page = block.page

            current_line += 1
            renumbered_blocks.append(
                TextBlock(
                    page=block.page,
                    line_number=current_line,
                    text=block.text.strip(),
                    raw_text=block.raw_text.strip(),
                    bbox=block.bbox,
                    centroid=block.centroid,
                )
            )

        return renumbered_blocks

    def _merge_fragmented_lines(self, text_blocks: List[TextBlock]) -> List[TextBlock]:
        """
        Merge text lines that are clearly part of the same sentence/unit.

        Merges lines when:
        - Line doesn't end with punctuation and next line doesn't start with capital/number
        - Line ends with comma, semicolon, or dash
        - Line is very short (< 20 chars) and doesn't end with punctuation
        """
        if not text_blocks:
            return text_blocks

        merged = []
        current_block = text_blocks[0]

        for i in range(1, len(text_blocks)):
            next_block = text_blocks[i]
            current_text = current_block.text.strip()
            next_text = next_block.text.strip()

            # Never merge connectors (ja/nee/single digits) with other blocks
            current_is_connector = re.match(
                r"^(ja|nee|\d+)\s*$", current_text, re.IGNORECASE
            )
            next_is_connector = re.match(r"^(ja|nee|\d+)\s*$", next_text, re.IGNORECASE)
            if current_is_connector or next_is_connector:
                # Finalize current block and start new one
                merged.append(current_block)
                current_block = next_block
                continue

            # Check if lines should be merged
            should_merge = False

            # Case 1: Current line doesn't end with sentence-ending punctuation
            if not re.search(r"[.!?]$", current_text):
                # Merge if next line doesn't start with capital letter or number (likely continuation)
                if next_text and not re.match(r"^[A-Z0-9]", next_text):
                    should_merge = True
                # Merge if current line ends with comma, semicolon, dash
                elif re.search(r"[,;—]$", current_text):
                    should_merge = True
                # Merge if current line is very short (< 20 chars) - likely fragment
                elif len(current_text) < 20:
                    should_merge = True

            # Case 2: Current line ends with question mark but next line is continuation
            # (e.g., "vervuild is?" followed by more text)
            if re.search(r"\?$", current_text) and len(current_text) < 30:
                if next_text and not re.match(r"^[A-Z0-9]", next_text):
                    should_merge = True

            # Case 3: Next line starts with lowercase and is a fragment (e.g., "van de bretslede correct?")
            # This is clearly a continuation of the previous line
            if next_text and re.match(
                r"^(van|de|het|een|is|zijn|werkt|correct)\s+", next_text, re.IGNORECASE
            ):
                if len(next_text) < 40:  # Short fragment
                    should_merge = True

            # Case 4: Don't merge across flowchart branches (detect "ja"/"nee" labels)
            # If current line ends with "ja" or "nee" or next line starts with "ja"/"nee" -> don't merge
            if re.search(r"\b(ja|nee)\s*[->]?\s*$", current_text, re.IGNORECASE):
                should_merge = False
            if next_text and re.match(
                r"^(ja|nee)\s*[->]?\s*", next_text, re.IGNORECASE
            ):
                should_merge = False

            # Case 5: Don't merge conditions with actions (they're separate flowchart elements)
            # Check if current is condition-like and next is action-like
            current_is_condition = bool(
                re.search(r"\?$|^(Is|Zijn|Werkt|Voldoet)", current_text, re.IGNORECASE)
            )
            next_is_action = bool(
                re.search(
                    r"^(Controleer|Vervang|Stel|Maak|Reinig|Wissel)",
                    next_text,
                    re.IGNORECASE,
                )
            )
            if current_is_condition and next_is_action:
                should_merge = False

            # Case 6: Don't merge if next line starts with numbered action prefix (e.g., "2a)", "3b)")
            # These indicate separate flowchart steps
            if next_text and re.match(r"^\d+[a-z]\)\s*", next_text, re.IGNORECASE):
                should_merge = False

            # Case 7: Don't merge if next line starts with action code (e.g., "A2)", "B3)")
            if next_text and re.match(r"^[A-Z]\d+\)\s*", next_text):
                should_merge = False

            # Case 8: Don't merge if current line ends with numbered reference (e.g., "4-1", "4-2")
            # These are connector references, next line is likely a new step
            if re.search(r"\d+-\d+\s*$", current_text):
                should_merge = False

            if should_merge and current_block.page == next_block.page:
                # Merge: combine text, keep first line number
                merged_text = f"{current_text} {next_text}"
                current_block = TextBlock(
                    page=current_block.page,
                    line_number=current_block.line_number,
                    text=merged_text,
                    raw_text=merged_text,
                    bbox=current_block.bbox,
                    centroid=current_block.centroid,
                )
            else:
                # Finalize current block and start new one
                merged.append(current_block)
                current_block = next_block

        # Add last block
        merged.append(current_block)

        return merged

    def segment_procedural_units(
        self, text_blocks: List[TextBlock]
    ) -> List[ProceduralUnit]:
        """
        Step 2: Segment text into procedural units using NLP.

        Classifies each line as Condition, Action, Observation, or Connector.

        Args:
            text_blocks: List of extracted text blocks

        Returns:
            List of ProceduralUnit objects
        """
        units = []
        nlp = self._get_nlp_model()

        # Action verbs to detect (even if not at start of line)
        action_verbs = [
            "controleer",
            "vervang",
            "stel",
            "maak",
            "reinig",
            "wissel",
            "afstellen",
            "kalibreer",
            "check",
            "smeren",
            "vastzetten",
            "proefdraaien",
            "teach",
            "positioneren",
        ]

        # Patterns for classification
        condition_patterns = [
            r"\?$",  # Ends with question mark
            r"^(Is|Zijn|Werkt|Gaat|Voldoet|Klopt)",  # Interrogative start
            r"(correct\?|werkt\?|defect\?|versleten\?|vervuild\?)",
            r"(ja|nee)\s*[->]",
            r"^Als\s+",
            r"^Wanneer\s+",
        ]

        action_patterns = [
            # Allow text before numbered prefix, but action verb must be within first 80 chars: "Ned 2a) Controleer", "PS 3a) Controleer", etc.
            r"^.{0,80}?\d+[a-z]?\)\s*(Controleer|Vervang|Stel|Maak|Reinig|Wissel|Afstellen|Kalibreer|Check|Smeren|Vastzetten|Proefdraaien|Teach|Positioneren)",
            # Direct action verbs at start of line (without prefix)
            r"^(Controleer|Vervang|Stel|Maak|Reinig|Wissel|Afstellen|Kalibreer|Check|Smeren|Vastzetten|Proefdraaien|Teach|Positioneren)\b",
            # Action codes (but only if followed immediately by action verb)
            r"^[A-Z]\d+\)\s*(Controleer|Vervang|Stel|Maak|Reinig|Wissel|Afstellen|Kalibreer|Check|Smeren|Vastzetten|Proefdraaien|Teach|Positioneren)",
        ]

        connector_patterns = [
            r"^\d+-\d+",  # Step references like "4-1", "4-0"
            r"^[A-Z]\d+\)(?!\s*(Controleer|Vervang|Stel|Maak|Reinig|Wissel|Afstellen|Kalibreer|Check|Smeren|Vastzetten|Proefdraaien|Teach|Positioneren))",  # Codes like "A3)" but NOT if followed by action verb
            r"^TOCAP\s+\d+",
            r"^OCAP\s+\d+",
            r"^(ja|nee)\s*$",  # Single word "ja" or "nee"
            r"^\d+\s*$",  # Single digit (page connector)
            r"^FO-\d+-\d+-\d+",  # Document reference codes like "FO-00-000-4"
        ]

        # Get page dimensions for column inference
        # Use stored page widths from PDF extraction if available, otherwise estimate from bboxes
        page_widths = getattr(self, "_page_widths", {})
        if not page_widths:
            # Fallback: estimate from bboxes
            for block in text_blocks:
                if block.bbox and block.page not in page_widths:
                    # Approximate page width from bbox (add margin)
                    page_widths[block.page] = (
                        block.bbox[2] * 1.1
                    )  # x1 * 1.1 as approximation

        for block in text_blocks:
            text = block.text
            unit_type = "Observation"  # Default
            confidence = 0.5
            text_lower = text.lower()
            inferred_shape = None

            # Skip OCR garbage before classification
            # Check for repeated character patterns (OCR error)
            if re.search(r"([A-Za-z])\1{2,}", text) and len(text) > 15:
                # Likely OCR garbage with repeated characters
                continue
            # Check for all caps long strings (likely OCR garbage)
            if re.match(r"^[A-Z]{10,}$", text):
                # Likely OCR garbage (all caps, very long)
                continue
            # Check for "Inhoudsopgave" OCR errors specifically
            if re.search(
                r"[Ii]{3,}[Nn]{2,}[Hh]{2,}[Oo]{3,}[Uu]{2,}[Dd]{3,}[Ss]{3,}[Oo]{3,}[Pp]{2,}[Gg]{2,}[Aa]{2,}[Vv]{2,}[Ee]{3,}",
                text,
                re.IGNORECASE,
            ):
                # This is clearly "Inhoudsopgave" OCR garbage - skip it
                continue

            # PRIORITY 1: Check for questions first (before connector patterns)
            # Questions should be classified as Conditions, not Connectors
            is_question = False
            for pattern in condition_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    is_question = True
                    break

            # Also check for question words in numbered references (e.g., "4-0 Voldoet...?")
            if not is_question and re.match(r"^\d+-\d+\s+", text):
                # Check if text after the number contains a question
                text_after_ref = re.sub(r"^\d+-\d+\s+", "", text)
                if re.search(
                    r"\?|^(Is|Zijn|Werkt|Gaat|Voldoet|Klopt)",
                    text_after_ref,
                    re.IGNORECASE,
                ):
                    is_question = True

            # If it's a question, skip connector checks and continue to classification
            if is_question:
                # Will be classified as Condition below
                pass
            else:
                # Filter single connectors early (only if not a question)
                # Page connectors (single digits 1-9) are handled specially - mark them
                if re.match(r"^\d+\s*$", text):
                    connector_num = text.strip()
                    # Single digit (1-9) = page connector, multi-digit = regular connector
                    if len(connector_num) == 1 and connector_num.isdigit():
                        # This is a page connector - mark it specially
                        unit_type = "PageConnector"
                        confidence = 0.95
                        unit_id = f"page_connector_p{block.page}_{connector_num}"
                    else:
                        # Multi-digit number = regular connector
                        unit_type = "Connector"
                        confidence = 0.9
                        unit_id = f"p{block.page}_l{block.line_number}"

                    units.append(
                        ProceduralUnit(
                            id=unit_id,
                            type=unit_type,
                            text=text,
                            page=block.page,
                            line_number=block.line_number,
                            confidence=confidence,
                            bbox=block.bbox,
                            centroid=block.centroid,
                            inferred_shape="connector",
                        )
                    )
                    continue

            # Single "ja" or "nee" = branch connector
            if re.match(r"^(ja|nee)\s*$", text, re.IGNORECASE):
                unit_type = "Connector"
                confidence = 0.9
                unit_id = f"p{block.page}_l{block.line_number}"
                units.append(
                    ProceduralUnit(
                        id=unit_id,
                        type=unit_type,
                        text=text.lower(),
                        page=block.page,
                        line_number=block.line_number,
                        confidence=confidence,
                        bbox=block.bbox,
                        centroid=block.centroid,
                        inferred_shape="connector",
                    )
                )
                continue

            # HYBRID APPROACH: Use spatial position to infer shape type
            # Column-based heuristics (left=condition, center=decision, right=action)
            spatial_type = None
            if block.centroid and block.page in page_widths:
                page_width = page_widths[block.page]
                x_norm = block.centroid[0] / page_width if page_width > 0 else 0.5

                if x_norm < 0.33:
                    # Left column -> likely condition box
                    spatial_type = "Condition"
                    inferred_shape = "box"
                elif 0.33 <= x_norm <= 0.66:
                    # Center column -> likely decision diamond
                    spatial_type = "Condition"  # Decisions are conditions
                    inferred_shape = "diamond"
                elif x_norm > 0.66:
                    # Right column -> likely action box
                    spatial_type = "Action"
                    inferred_shape = "box"

            # PRIORITY 1: Semantic classification (content-based)
            # If already identified as question, use that
            if is_question:
                semantic_type = "Condition"
                semantic_confidence = 0.85
            else:
                # Check if action verb appears in first 80 chars
                text_start = text_lower[:80]
                has_action_verb_at_start = any(
                    verb in text_start for verb in action_verbs
                )
                matches_action_pattern = any(
                    re.search(pattern, text, re.IGNORECASE)
                    for pattern in action_patterns
                )

                semantic_type = None
                semantic_confidence = 0.5

                if (has_action_verb_at_start or matches_action_pattern) and len(
                    text
                ) < 200:
                    semantic_type = "Action"
                    semantic_confidence = 0.85
                elif has_action_verb_at_start and len(text) >= 200:
                    semantic_type = "Action"
                    semantic_confidence = 0.85
                elif matches_action_pattern and len(text) < 150:
                    semantic_type = "Action"
                    semantic_confidence = 0.85
                elif any(
                    re.search(pattern, text, re.IGNORECASE)
                    for pattern in condition_patterns
                ):
                    semantic_type = "Condition"
                    semantic_confidence = 0.85
                elif any(
                    re.search(pattern, text, re.IGNORECASE)
                    for pattern in connector_patterns
                ):
                    semantic_type = "Connector"
                    semantic_confidence = 0.9
                elif nlp:
                    doc = nlp(text)
                    has_state_verbs = any(
                        token.pos_ == "VERB"
                        and token.lemma_ in ["zijn", "worden", "hebben"]
                        for token in doc
                    )
                    if has_state_verbs and len(doc) > 5:
                        semantic_type = "Observation"
                        semantic_confidence = 0.7

            # HYBRID DECISION: Combine semantic and spatial
            if semantic_type and spatial_type:
                # Both agree -> high confidence
                if semantic_type == spatial_type:
                    unit_type = semantic_type
                    confidence = min(0.95, semantic_confidence + 0.1)
                # Semantic overrides spatial for Actions (more reliable)
                elif semantic_type == "Action":
                    unit_type = "Action"
                    confidence = semantic_confidence
                # Spatial helps for Conditions in left column
                elif spatial_type == "Condition" and semantic_type == "Condition":
                    unit_type = "Condition"
                    confidence = max(semantic_confidence, 0.8)
                else:
                    # Prefer semantic, but adjust confidence based on spatial agreement
                    unit_type = semantic_type
                    confidence = semantic_confidence
            elif semantic_type:
                # Only semantic available
                unit_type = semantic_type
                confidence = semantic_confidence
            elif spatial_type:
                # Only spatial available (use it, but lower confidence)
                unit_type = spatial_type
                confidence = 0.65
            # else: keep default "Observation" with confidence 0.5

            unit_id = f"p{block.page}_l{block.line_number}"
            units.append(
                ProceduralUnit(
                    id=unit_id,
                    type=unit_type,
                    text=text,
                    page=block.page,
                    line_number=block.line_number,
                    confidence=confidence,
                    bbox=block.bbox,
                    centroid=block.centroid,
                    inferred_shape=inferred_shape,
                )
            )

        # Save intermediate output
        if self.debug:
            # Use page number for filename if available
            page_num = units[0].page if units else 1
            output_file = self.output_dir / f"procedural_units_p{page_num}.json"
            output_data = [asdict(unit) for unit in units]
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            print(f"Procedural units saved to: {output_file}")

        return units

    def extract_entities(self, text_blocks: List[TextBlock]) -> List[Entity]:
        """
        Step 3: Extract entities (components, parts) from text.

        Uses domain glossary and noun phrase extraction.

        Args:
            text_blocks: List of extracted text blocks

        Returns:
            List of Entity objects
        """
        entities_dict = {}
        nlp = self._get_nlp_model()

        # Common words to exclude from entity extraction
        common_words = {
            "de",
            "het",
            "een",
            "van",
            "op",
            "in",
            "voor",
            "met",
            "aan",
            "bij",
            "naar",
            "is",
            "zijn",
            "werkt",
            "wordt",
            "hebben",
            "kan",
            "moet",
            "zou",
            "correct",
            "goed",
            "klaar",
            "ok",
            "nee",
            "ja",
            "periode",
            "auteur",
            "versiedatum",
            "opmerkingen",
            "laatste",
            "wijzigingen",
            "wissel",
            "vervang",
            "stel",
            "maak",  # Verbs, not components
        }

        # Adjectives and states to exclude (not components)
        adjectives_to_exclude = {
            "statisch",
            "vervuild",
            "versleten",
            "defect",
            "correct",
            "goed",
            "vervangen",
            "schoon",
            "groot",
            "klein",
            "lang",
            "kort",
        }

        # OCR garbage patterns to filter
        ocr_garbage_patterns = [
            r"^[A-Z]{2,}\s+[A-Z]{2,}",  # "LL me", "PS 3a", etc. (but keep if followed by meaningful text)
            r"^[A-Z]{3,}$",  # All caps short words
            r"ghressto|benieetehen|lijn er",  # Known OCR errors
        ]

        for block in text_blocks:
            text = block.text.lower()
            text_words = set(
                re.findall(r"\b\w+\b", text)
            )  # Extract words with boundaries

            # Check glossary matches with exact word boundaries
            for component in self.component_glossary:
                component_lower = component.lower()

                # Skip if component is an adjective/state (not a real component)
                if component_lower in adjectives_to_exclude:
                    continue

                component_words = set(re.findall(r"\b\w+\b", component_lower))

                # Use word boundary matching to avoid partial matches
                # Check if all words of component are in text
                if component_words.issubset(text_words) or re.search(
                    r"\b" + re.escape(component_lower) + r"\b", text
                ):
                    # Skip if it's a common word
                    if component_lower not in common_words:
                        # Clean component name: remove leading articles
                        clean_component = component
                        if component_lower.startswith(("de ", "het ", "een ")):
                            # Extract main noun (everything after article)
                            clean_component = re.sub(
                                r"^(de|het|een)\s+", "", component, flags=re.IGNORECASE
                            ).strip()

                        if clean_component not in entities_dict:
                            entities_dict[clean_component] = Entity(
                                id=f"entity_{len(entities_dict)}",
                                text=clean_component,
                                type="Component",
                                mentions=[],
                                confidence=0.9,
                            )
                        entities_dict[clean_component].mentions.append(
                            (block.page, block.line_number)
                        )

            # Use NLP for noun phrase extraction
            if nlp:
                doc = nlp(block.text)
                for chunk in doc.noun_chunks:
                    chunk_text = chunk.text.strip()
                    chunk_lower = chunk_text.lower()

                    # Filter criteria:
                    # 1. Must be longer than 4 characters
                    # 2. Not a common word
                    # 3. Not an adjective/state
                    # 4. Not OCR garbage
                    # 5. Not a verb (check POS tag)
                    # 6. Not already extracted from glossary
                    # 7. Contains at least one noun
                    if len(chunk_text) > 4:
                        # Check if it's a common word
                        if chunk_lower in common_words:
                            continue

                        # Check if it's an adjective/state (not a component)
                        if chunk_lower in adjectives_to_exclude:
                            continue

                        # Check if it's OCR garbage
                        if any(
                            re.search(pattern, chunk_text, re.IGNORECASE)
                            for pattern in ocr_garbage_patterns
                        ):
                            # But allow if it's part of a longer meaningful phrase
                            if len(chunk_text.split()) <= 2:
                                continue

                        # Check if it contains verbs (likely not a component)
                        has_verb = any(token.pos_ == "VERB" for token in chunk)
                        if has_verb:
                            continue

                        # Filter coordinate references (e.g., "x/y/z")
                        if re.match(r"^[xyz][/\\][xyz]", chunk_text, re.IGNORECASE):
                            continue

                        # Filter document reference codes (e.g., "FO-00-000-4")
                        if re.match(r"^[A-Z]{2}-\d+-\d+-\d+", chunk_text):
                            continue

                        # Check if it's already in entities_dict (from glossary)
                        if chunk_text in entities_dict or any(
                            chunk_lower == e.lower() for e in entities_dict.keys()
                        ):
                            continue

                        # Must contain at least one noun
                        has_noun = any(
                            token.pos_ in ["NOUN", "PROPN"] for token in chunk
                        )

                        # Filter out action descriptions (phrases that start with verbs)
                        # Check if first word is a verb or action word
                        first_token = chunk[0] if len(chunk) > 0 else None
                        is_action_phrase = False
                        if first_token:
                            if first_token.pos_ == "VERB" or first_token.lemma_ in [
                                "oppakken",
                                "plaatsen",
                                "wissel",
                                "vervang",
                                "stel",
                                "maak",
                                "reinig",
                            ]:
                                is_action_phrase = True

                        # Filter out phrases starting with articles that are incomplete
                        # (e.g., "de geleiding van" should be just "geleiding")
                        if (
                            chunk_text.startswith(("de ", "het ", "een "))
                            and len(chunk_text.split()) <= 3
                        ):
                            # Extract the main noun instead
                            main_noun = None
                            for token in chunk:
                                if token.pos_ in ["NOUN", "PROPN"]:
                                    main_noun = token.text
                                    break
                            if main_noun:
                                chunk_text = main_noun
                                chunk_lower = chunk_text.lower()
                            else:
                                continue  # Skip if no main noun found

                        if has_noun and not is_action_phrase:
                            entities_dict[chunk_text] = Entity(
                                id=f"entity_{len(entities_dict)}",
                                text=chunk_text,
                                type="Component",
                                mentions=[],
                                confidence=0.6,
                            )
                            entities_dict[chunk_text].mentions.append(
                                (block.page, block.line_number)
                            )

        # Deduplicate similar entities (e.g., "bretslede" vs "de bretslede")
        entities_dict = self._deduplicate_entities(entities_dict)

        entities = list(entities_dict.values())

        # Save intermediate output
        if self.debug:
            output_file = self.output_dir / "entities.json"
            output_data = [asdict(entity) for entity in entities]
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            print(f"Entities saved to: {output_file}")

        return entities

    def _deduplicate_entities(
        self, entities_dict: Dict[str, Entity]
    ) -> Dict[str, Entity]:
        """
        Deduplicate entities that are similar (e.g., "bretslede" vs "de bretslede").
        Keeps the version without articles/determiners.
        """
        if not entities_dict:
            return entities_dict

        # Articles and determiners to remove
        articles = {"de", "het", "een", "van", "op", "in"}

        # Normalize entity names (remove articles, lowercase)
        normalized_map = {}
        for entity_name, entity in entities_dict.items():
            # Remove leading articles
            words = entity_name.lower().split()
            while words and words[0] in articles:
                words = words[1:]
            normalized = " ".join(words) if words else entity_name.lower()
            # Normalize hyphens and spaces: treat hyphen and space as equivalent
            normalized = re.sub(
                r"[-\s]+", " ", normalized
            )  # Replace hyphens and multiple spaces with single space
            normalized = normalized.strip()

            if normalized not in normalized_map:
                normalized_map[normalized] = []
            normalized_map[normalized].append((entity_name, entity))

        # Keep the shortest version for each normalized form
        deduplicated = {}
        for normalized, variants in normalized_map.items():
            if len(variants) == 1:
                # Only one variant, keep it
                name, entity = variants[0]
                deduplicated[name] = entity
            else:
                # Multiple variants - keep the shortest (usually without articles)
                variants.sort(key=lambda x: len(x[0]))
                name, entity = variants[0]

                # Merge mentions from all variants
                for other_name, other_entity in variants[1:]:
                    entity.mentions.extend(other_entity.mentions)

                # Remove duplicates from mentions (mentions are already tuples)
                entity.mentions = list(set(entity.mentions))

                deduplicated[name] = entity

        return deduplicated

    def infer_relations(
        self,
        units: List[ProceduralUnit],
        entities: List[Entity],
    ) -> List[Relation]:
        """
        Step 4: Infer relations using weak supervision rules.

        Args:
            units: List of procedural units
            entities: List of extracted entities

        Returns:
            List of Relation objects
        """
        relations = []

        # HYBRID: Combine semantic and spatial relation inference

        # Rule 1: Condition -> Action (semantic + spatial)
        # Skip metadata units (they shouldn't have relationships)
        metadata_indicators = [
            "naam",
            "periode",
            "unit",
            "inhoudsopgave",
            "opmerkingen",
            "auteur",
            "versiedatum",
        ]

        for i, unit in enumerate(units):
            # Skip metadata units
            unit_text_lower = unit.text.lower()
            if any(indicator in unit_text_lower for indicator in metadata_indicators):
                continue

            if unit.type == "Condition":
                # Look ahead up to 8 units (more permissive to connect more nodes)
                for j in range(i + 1, min(i + 9, len(units))):
                    next_unit = units[j]
                    # Skip if next unit is metadata
                    next_text_lower = next_unit.text.lower()
                    if any(
                        indicator in next_text_lower
                        for indicator in metadata_indicators
                    ):
                        continue

                    if next_unit.type == "Action" and next_unit.page == unit.page:
                        # Check line proximity
                        line_proximity = (
                            abs(next_unit.line_number - unit.line_number) <= 5
                        )

                        # Check spatial proximity (if bboxes available)
                        spatial_proximity = False
                        confidence = 0.8
                        distance = None
                        if unit.centroid and next_unit.centroid:
                            # Calculate distance
                            dx = next_unit.centroid[0] - unit.centroid[0]
                            dy = next_unit.centroid[1] - unit.centroid[1]
                            distance = (dx**2 + dy**2) ** 0.5

                            # Filter out relationships with excessive distance (>500pt - more permissive)
                            if distance > 500:
                                continue

                            # Flow direction: left to right, top to bottom
                            # Left condition should flow right (to center/right action)
                            # Center decision should flow right (to right action)
                            is_flow_direction = (
                                dx > -100
                            )  # More permissive - allow more left movement
                            is_downward = (
                                dy > -50
                            )  # More permissive - allow more upward movement

                            # Normalize distance (rough threshold: ~500 points - more permissive)
                            spatial_proximity = distance < 500 and (
                                is_flow_direction or is_downward
                            )

                            if spatial_proximity:
                                confidence = (
                                    0.85  # Higher confidence with spatial agreement
                                )

                        if line_proximity or spatial_proximity:
                            evidence_text = f"Adjacent condition-action (lines {unit.line_number}-{next_unit.line_number})"
                            if distance is not None:
                                evidence_text += f", spatial distance: {distance:.0f}pt"
                            relations.append(
                                Relation(
                                    source=unit.id,
                                    target=next_unit.id,
                                    relation_type="leads_to",
                                    confidence=confidence,
                                    evidence=evidence_text,
                                )
                            )
                        break

                # Also check for "ja/nee" branches: look for connector units with "ja" or "nee"
                for j in range(i + 1, min(i + 4, len(units))):
                    next_unit = units[j]
                    if next_unit.page == unit.page and next_unit.type == "Connector":
                        # Look for action after connector
                        for k in range(j + 1, min(j + 3, len(units))):
                            action_unit = units[k]
                            if (
                                action_unit.type == "Action"
                                and action_unit.page == unit.page
                            ):
                                # Validate distance
                                if unit.centroid and action_unit.centroid:
                                    dx = action_unit.centroid[0] - unit.centroid[0]
                                    dy = action_unit.centroid[1] - unit.centroid[1]
                                    distance = (dx**2 + dy**2) ** 0.5
                                    if distance > 300:
                                        continue

                                relations.append(
                                    Relation(
                                        source=unit.id,
                                        target=action_unit.id,
                                        relation_type="leads_to",
                                        confidence=0.75,
                                        evidence=f"Condition with ja/nee branch to action (lines {unit.line_number}-{action_unit.line_number})",
                                    )
                                )
                                break
                        break

        # Rule 2: Action -> Action (sequential actions on same page)
        for i, unit in enumerate(units):
            if unit.type == "Action":
                unit_text_lower = unit.text.lower()
                # Skip metadata
                if any(
                    indicator in unit_text_lower for indicator in metadata_indicators
                ):
                    continue
                # Look for next action on same page (more permissive - up to 8 units ahead)
                for j in range(i + 1, min(i + 9, len(units))):
                    next_unit = units[j]
                    next_text_lower = next_unit.text.lower()
                    # Skip metadata
                    if any(
                        indicator in next_text_lower
                        for indicator in metadata_indicators
                    ):
                        continue
                    if next_unit.type == "Action" and next_unit.page == unit.page:
                        # Check proximity (more permissive)
                        should_connect = True
                        if unit.centroid and next_unit.centroid:
                            dx = next_unit.centroid[0] - unit.centroid[0]
                            dy = next_unit.centroid[1] - unit.centroid[1]
                            distance = (dx**2 + dy**2) ** 0.5
                            # More permissive distance threshold
                            if distance > 600:
                                should_connect = False
                            # Prefer downward flow
                            elif dy < -100:  # Too far upward
                                should_connect = False

                        if should_connect:
                            relations.append(
                                Relation(
                                    source=unit.id,
                                    target=next_unit.id,
                                    relation_type="leads_to",
                                    confidence=0.7,
                                    evidence=f"Sequential actions (lines {unit.line_number}-{next_unit.line_number})",
                                )
                            )
                            break

        # Rule 3: Action -> Condition (action followed by check)
        for i, unit in enumerate(units):
            if unit.type == "Action":
                unit_text_lower = unit.text.lower()
                # Skip metadata
                if any(
                    indicator in unit_text_lower for indicator in metadata_indicators
                ):
                    continue
                # Look for condition after action
                for j in range(i + 1, min(i + 6, len(units))):
                    next_unit = units[j]
                    next_text_lower = next_unit.text.lower()
                    # Skip metadata
                    if any(
                        indicator in next_text_lower
                        for indicator in metadata_indicators
                    ):
                        continue
                    if next_unit.type == "Condition" and next_unit.page == unit.page:
                        # Check proximity
                        if unit.centroid and next_unit.centroid:
                            dx = next_unit.centroid[0] - unit.centroid[0]
                            dy = next_unit.centroid[1] - unit.centroid[1]
                            distance = (dx**2 + dy**2) ** 0.5
                            if distance > 400:
                                continue
                        relations.append(
                            Relation(
                                source=unit.id,
                                target=next_unit.id,
                                relation_type="requires_check",
                                confidence=0.7,
                                evidence=f"Action followed by condition check (lines {unit.line_number}-{next_unit.line_number})",
                            )
                        )
                        break

        # Rule 4: Connector -> Action (connector followed by action)
        for i, unit in enumerate(units):
            if unit.type == "Connector":
                unit_text_lower = unit.text.lower()
                # Skip metadata
                if any(
                    indicator in unit_text_lower for indicator in metadata_indicators
                ):
                    continue
                # Look for action after connector (same page, within 4 units)
                for j in range(i + 1, min(i + 5, len(units))):
                    next_unit = units[j]
                    next_text_lower = next_unit.text.lower()
                    # Skip metadata
                    if any(
                        indicator in next_text_lower
                        for indicator in metadata_indicators
                    ):
                        continue
                    if next_unit.type == "Action" and next_unit.page == unit.page:
                        # Check proximity
                        should_connect = True
                        if unit.centroid and next_unit.centroid:
                            dx = next_unit.centroid[0] - unit.centroid[0]
                            dy = next_unit.centroid[1] - unit.centroid[1]
                            distance = (dx**2 + dy**2) ** 0.5
                            if distance > 400:
                                should_connect = False

                        if should_connect:
                            relations.append(
                                Relation(
                                    source=unit.id,
                                    target=next_unit.id,
                                    relation_type="leads_to",
                                    confidence=0.7,
                                    evidence=f"Connector to action (lines {unit.line_number}-{next_unit.line_number})",
                                )
                            )
                            break

        # Rule 5: Action -> Component (if action mentions entity with word boundaries)
        for unit in units:
            if unit.type == "Action":
                unit_text_lower = unit.text.lower()
                # Skip metadata
                if any(
                    indicator in unit_text_lower for indicator in metadata_indicators
                ):
                    continue
                for entity in entities:
                    entity_lower = entity.text.lower()
                    # Use word boundary matching to avoid partial matches
                    if re.search(
                        r"\b" + re.escape(entity_lower) + r"\b", unit_text_lower
                    ):
                        relations.append(
                            Relation(
                                source=unit.id,
                                target=entity.id,
                                relation_type="affects",
                                confidence=0.7,
                                evidence=f"Action mentions component: {entity.text}",
                            )
                        )

        # Rule 6: Connect sequential units on same page (general flow)
        # This helps connect units that are close together even if they don't match specific patterns
        for i, unit in enumerate(units):
            unit_text_lower = unit.text.lower()
            # Skip metadata
            if any(indicator in unit_text_lower for indicator in metadata_indicators):
                continue

            # Look for next unit on same page (within 3 units, consecutive lines)
            for j in range(i + 1, min(i + 4, len(units))):
                next_unit = units[j]
                next_text_lower = next_unit.text.lower()
                # Skip metadata
                if any(
                    indicator in next_text_lower for indicator in metadata_indicators
                ):
                    continue

                # Same page and consecutive or very close lines
                if next_unit.page == unit.page:
                    line_gap = abs(next_unit.line_number - unit.line_number)
                    if line_gap <= 2:  # Very close lines
                        # Check if relation already exists
                        existing = any(
                            r.source == unit.id and r.target == next_unit.id
                            for r in relations
                        )
                        if not existing:
                            # Only connect if they're different types (avoid redundant connections)
                            if unit.type != next_unit.type:
                                relations.append(
                                    Relation(
                                        source=unit.id,
                                        target=next_unit.id,
                                        relation_type="leads_to",
                                        confidence=0.6,
                                        evidence=f"Sequential units on same page (lines {unit.line_number}-{next_unit.line_number})",
                                    )
                                )
                                break

        # Rule 3: Condition -> Result (REMOVED - creates non-existent nodes)
        # Instead, we'll skip this rule to avoid creating edges to non-existent nodes
        # If result nodes are needed, they should be created explicitly in build_ontology_graph

        # Rule 4: Connector -> Next step (sequential flow) - REMOVED
        # Connectors like "ja"/"nee" are already handled in Rule 1
        # Single digit connectors are page connectors, handled separately if needed

        # Remove duplicate relations (same source and target)
        seen_relations = set()
        unique_relations = []
        for rel in relations:
            rel_key = (rel.source, rel.target, rel.relation_type)
            if rel_key not in seen_relations:
                seen_relations.add(rel_key)
                unique_relations.append(rel)
            # If duplicate found, keep the one with higher confidence
            else:
                # Find existing relation and update if this one has higher confidence
                for existing_rel in unique_relations:
                    if (
                        existing_rel.source,
                        existing_rel.target,
                        existing_rel.relation_type,
                    ) == rel_key:
                        if rel.confidence > existing_rel.confidence:
                            unique_relations.remove(existing_rel)
                            unique_relations.append(rel)
                        break

        # Save intermediate output
        if self.debug:
            output_file = self.output_dir / "relations.json"
            output_data = [asdict(rel) for rel in unique_relations]
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            print(f"Relations saved to: {output_file}")

        return unique_relations

    def build_ontology_graph(
        self,
        units: List[ProceduralUnit],
        entities: List[Entity],
        relations: List[Relation],
        ontology_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Step 5: Build ontology graph and export to RDF/TTL format.

        Args:
            units: List of procedural units
            entities: List of entities
            relations: List of relations
            ontology_path: Path to ontology YAML schema

        Returns:
            Dictionary with graph structure
        """
        # Load ontology schema if provided
        ontology_schema = self._load_ontology_schema(ontology_path)

        # Build graph structure
        graph = {
            "nodes": [],
            "edges": [],
            "metadata": {
                "ontology_version": ontology_schema.get("version", "1.0"),
                "total_units": len(units),
                "total_entities": len(entities),
                "total_relations": len(relations),
            },
        }

        # Add procedural units as nodes
        for unit in units:
            node_data = {
                "id": unit.id,
                "type": unit.type,
                "label": unit.text,
                "page": unit.page,
                "line": unit.line_number,
                "confidence": unit.confidence,
            }
            # Include spatial information if available
            if unit.bbox:
                node_data["bbox"] = unit.bbox
            if unit.centroid:
                node_data["centroid"] = unit.centroid
            if unit.inferred_shape:
                node_data["inferred_shape"] = unit.inferred_shape
            graph["nodes"].append(node_data)

        # Add entities as nodes
        for entity in entities:
            graph["nodes"].append(
                {
                    "id": entity.id,
                    "type": entity.type,
                    "label": entity.text,
                    "mentions": len(entity.mentions),
                    "confidence": entity.confidence,
                }
            )

        # Add relations as edges
        for rel in relations:
            graph["edges"].append(
                {
                    "source": rel.source,
                    "target": rel.target,
                    "relation": rel.relation_type,
                    "confidence": rel.confidence,
                    "evidence": rel.evidence,
                }
            )

        # Export to TTL format
        ttl_content = self._export_to_ttl(graph, ontology_schema)
        ttl_file = self.output_dir / "ontology.ttl"
        with open(ttl_file, "w", encoding="utf-8") as f:
            f.write(ttl_content)
        print(f"Ontology exported to: {ttl_file}")

        return graph

    def _load_ontology_schema(self, ontology_path: Optional[str]) -> Dict[str, Any]:
        """Load ontology schema from YAML file."""
        if ontology_path and os.path.exists(ontology_path):
            try:
                import yaml

                with open(ontology_path, "r", encoding="utf-8") as f:
                    return yaml.safe_load(f)
            except Exception as e:
                print(f"Warning: Could not load ontology schema: {e}")
                return self._default_ontology_schema()
        return self._default_ontology_schema()

    def _default_ontology_schema(self) -> Dict[str, Any]:
        """Return default ontology schema."""
        return {
            "version": "1.0",
            "classes": [
                {"name": "Condition"},
                {"name": "Action"},
                {"name": "Observation"},
                {"name": "Component"},
                {"name": "Result"},
            ],
            "relations": [
                {"name": "leads_to"},
                {"name": "requires_check"},
                {"name": "affects"},
                {"name": "equivalent_to"},
            ],
        }

    def _export_to_ttl(self, graph: Dict[str, Any], schema: Dict[str, Any]) -> str:
        """Export graph to Turtle (TTL) RDF format."""
        ttl_lines = [
            "@prefix proc: <http://example.org/procedural#> .",
            "@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .",
            "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .",
            "",
        ]

        # Add class definitions
        classes = schema.get("classes", [])
        for class_def in classes:
            if isinstance(class_def, dict):
                class_name = class_def.get("name", class_def)
            else:
                class_name = class_def
            ttl_lines.append(f"proc:{class_name} rdf:type rdfs:Class .")

        ttl_lines.append("")

        # Add nodes
        for node in graph["nodes"]:
            node_id = node["id"].replace(" ", "_")
            node_type = node["type"]
            label = node["label"].replace('"', '\\"')
            ttl_lines.append(f"proc:{node_id} rdf:type proc:{node_type} ;")
            ttl_lines.append(f'    rdfs:label "{label}" .')
            ttl_lines.append("")

        # Add relations
        for edge in graph["edges"]:
            source_id = edge["source"].replace(" ", "_")
            target_id = edge["target"].replace(" ", "_")
            relation = edge["relation"]
            ttl_lines.append(f"proc:{source_id} proc:{relation} proc:{target_id} .")

        return "\n".join(ttl_lines)

    def extract_full_pipeline(
        self, pdf_path: str, ontology_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run the complete semantic extraction pipeline.

        Args:
            pdf_path: Path to PDF file
            ontology_path: Optional path to ontology YAML schema

        Returns:
            Complete knowledge graph structure
        """
        print("Step 1: Extracting text blocks...")
        text_blocks = self.extract_text_blocks(pdf_path)

        print("Step 2: Segmenting procedural units...")
        units = self.segment_procedural_units(text_blocks)

        print("Step 3: Extracting entities...")
        entities = self.extract_entities(text_blocks)

        print("Step 4: Inferring relations...")
        relations = self.infer_relations(units, entities)

        print("Step 5: Building ontology graph...")
        graph = self.build_ontology_graph(units, entities, relations, ontology_path)

        return graph
