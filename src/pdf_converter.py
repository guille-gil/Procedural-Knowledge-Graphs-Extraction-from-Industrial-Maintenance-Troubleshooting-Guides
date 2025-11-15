"""
PDF to PNG converter for troubleshooting guides.
Converts PDF pages to high-quality PNG images suitable for VLM processing.
"""

import io
from pathlib import Path
from typing import List

import fitz  # PyMuPDF
from PIL import Image


def pdf_to_images(pdf_path: str, dpi: int = 300) -> List[Image.Image]:
    """
    Convert PDF pages to PIL Images.
    
    Args:
        pdf_path: Path to PDF file
        dpi: Resolution in DPI (default: 300 for high quality)
        
    Returns:
        List of PIL Images, one per page
    """
    doc = fitz.open(pdf_path)
    images = []
    
    zoom = dpi / 72  # PyMuPDF uses 72 DPI as base
    mat = fitz.Matrix(zoom, zoom)
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        # Render page to pixmap
        pix = page.get_pixmap(matrix=mat)
        # Convert to PIL Image
        img_data = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_data))
        images.append(img)
    
    doc.close()
    return images


def save_images(images: List[Image.Image], output_dir: str, base_name: str):
    """
    Save images to disk as PNG files.
    
    Args:
        images: List of PIL Images
        output_dir: Directory to save images
        base_name: Base name for files (e.g., "DP17-TOCAP4")
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for page_num, img in enumerate(images, start=1):
        filename = f"{base_name}_page_{page_num:02d}.png"
        filepath = output_path / filename
        img.save(filepath, "PNG", optimize=True)
        print(f"  Saved: {filename}")


def convert_pdf_to_png(pdf_path: str, output_dir: str, dpi: int = 300):
    """
    Convert a PDF file to PNG images.
    
    Args:
        pdf_path: Path to PDF file
        output_dir: Directory to save PNG images
        dpi: Resolution in DPI
    """
    pdf_file = Path(pdf_path)
    base_name = pdf_file.stem
    
    print(f"Converting {pdf_file.name}...")
    images = pdf_to_images(pdf_path, dpi=dpi)
    save_images(images, output_dir, base_name)
    print(f"  Converted {len(images)} pages\n")

