"""
Data loader module for loading PDFs, converting pages to images, and loading gold JSON annotations.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import fitz  # PyMuPDF
from PIL import Image
import io


def load_pdf_pages(pdf_path: str) -> List[Image.Image]:
    """
    Load a PDF and convert each page to a PIL Image.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        List of PIL Images, one per page
    """
    doc = fitz.open(pdf_path)
    pages = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        # Render page to a pixmap (image)
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better quality
        # Convert to PIL Image
        img_data = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_data))
        pages.append(img)

    doc.close()
    return pages


def extract_text_from_pdf(pdf_path: str) -> List[Dict[str, any]]:
    """
    Extract text from PDF pages using PyMuPDF.
    Returns text content per page for text-only baseline.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        List of dicts with 'page' number and 'text' content
    """
    doc = fitz.open(pdf_path)
    pages_text = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        pages_text.append({"page": page_num + 1, "text": text})

    doc.close()
    return pages_text


def load_annotations(json_path: str) -> Dict:
    """
    Load gold standard annotations from a JSON file.

    Expected format:
    {
        "entities": [
            { "id": "E1", "type": "Condition", "text": "...", "bbox": [x1,y1,x2,y2], "page": 1 }
        ],
        "relations": [
            { "source": "E1", "target": "E2", "type": "leads_to" }
        ]
    }

    Args:
        json_path: Path to the annotation JSON file

    Returns:
        Dictionary with 'entities' and 'relations' keys
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Ensure required keys exist
    if "entities" not in data:
        data["entities"] = []
    if "relations" not in data:
        data["relations"] = []

    return data


def load_all_guides(data_dir: str) -> List[Dict]:
    """
    Load all guides from the data directory.
    Each guide should have a PDF in data/pdf/ and annotation in data/annotations/.

    Args:
        data_dir: Root data directory containing 'pdf' and 'annotations' subdirectories

    Returns:
        List of guide dictionaries, each containing:
        - 'name': guide identifier (filename without extension)
        - 'pdf_path': path to PDF file
        - 'pages': list of PIL Images
        - 'text_pages': list of extracted text per page
        - 'annotations': gold standard annotations dict
    """
    pdf_dir = os.path.join(data_dir, "pdf")
    annotations_dir = os.path.join(data_dir, "annotations")

    guides = []

    # Find all PDF files
    pdf_files = sorted([f for f in os.listdir(pdf_dir) if f.endswith(".pdf")])

    for pdf_file in pdf_files:
        guide_name = Path(pdf_file).stem  # e.g., "DP17-TOCAP4"
        pdf_path = os.path.join(pdf_dir, pdf_file)
        annotation_path = os.path.join(annotations_dir, f"{guide_name}.json")

        # Load pages as images
        pages = load_pdf_pages(pdf_path)

        # Extract text for baseline
        text_pages = extract_text_from_pdf(pdf_path)

        # Load annotations if they exist
        annotations = None
        if os.path.exists(annotation_path):
            annotations = load_annotations(annotation_path)
        else:
            # Create empty annotation structure if missing
            annotations = {"entities": [], "relations": []}

        guides.append(
            {
                "name": guide_name,
                "pdf_path": pdf_path,
                "pages": pages,
                "text_pages": text_pages,
                "annotations": annotations,
            }
        )

    return guides
