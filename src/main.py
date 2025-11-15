"""
Main script for processing PDFs: convert to PNG.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.pdf_converter import convert_pdf_to_png


def process_all_pdfs(pdf_dir: str, output_dir: str, dpi: int = 300):
    """
    Process all PDFs in directory: convert to PNG.

    Args:
        pdf_dir: Directory containing PDF files
        output_dir: Directory to save PNG images
        dpi: Resolution for PNG conversion
    """
    pdf_path = Path(pdf_dir)
    output_path = Path(output_dir)

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    # Find all PDF files
    pdf_files = sorted(pdf_path.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in {pdf_dir}")
        return

    print(f"Found {len(pdf_files)} PDF files\n")
    print("=" * 80)
    print("Converting PDFs to PNG")
    print("=" * 80)

    # Convert PDFs to PNG
    for pdf_file in pdf_files:
        convert_pdf_to_png(str(pdf_file), str(output_path), dpi=dpi)

    # Count total images
    png_files = sorted(output_path.glob("*.png"))
    print("\n✓ Processing complete!")
    print(f"  Output directory: {output_path}")
    print(f"  Total images: {len(png_files)}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Convert PDFs to PNG")
    parser.add_argument(
        "--pdf_dir",
        type=str,
        default="data/pdf",
        help="Directory containing PDF files (default: data/pdf)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed",
        help="Output directory for PNG images (default: data/processed)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for PNG conversion (default: 300)",
    )

    args = parser.parse_args()

    process_all_pdfs(
        pdf_dir=args.pdf_dir,
        output_dir=args.output_dir,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
