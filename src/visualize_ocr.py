"""
OCR Bounding Box Visualization

Visualizes OCR bounding boxes on PDF pages for debugging.
"""

import numpy as np
from typing import List, Tuple, Optional
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

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


def visualize_ocr_bboxes_pdfplumber(
    pdf_path: str,
    text_blocks: List,
    output_dir: Path,
    page_widths: dict,
) -> None:
    """
    Visualize bounding boxes from pdfplumber extraction.
    
    Args:
        pdf_path: Path to PDF file
        text_blocks: List of TextBlock objects with bboxes
        output_dir: Directory to save visualization images
        page_widths: Dict mapping page_num -> page_width
    """
    if not PIL_AVAILABLE:
        return  # Silently skip if PIL not available
    
    try:
        from pdf2image import convert_from_path
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            images = convert_from_path(pdf_path, dpi=200)
    except ImportError:
        return  # Silently skip if pdf2image not available
    except Exception as e:
        print(f"Warning: Could not convert PDF to images: {e}")
        return
    
    # Group text blocks by page
    blocks_by_page = {}
    for block in text_blocks:
        if block.bbox:  # Only visualize blocks with bboxes
            if block.page not in blocks_by_page:
                blocks_by_page[block.page] = []
            blocks_by_page[block.page].append(block)
    
    # Draw bboxes on each page
    for page_num, image in enumerate(images, 1):
        if page_num not in blocks_by_page:
            continue
        
        # Convert PIL image to numpy array for drawing
        img_array = np.array(image)
        img_pil = Image.fromarray(img_array)
        draw = ImageDraw.Draw(img_pil)
        
        # Scale factor: PDF points to pixels (assuming 200 DPI)
        # PDF point = 1/72 inch, 200 DPI = 200 pixels per inch
        # So 1 PDF point = 200/72 pixels ≈ 2.78 pixels
        scale_factor = 200 / 72
        
        page_blocks = blocks_by_page[page_num]
        for i, block in enumerate(page_blocks):
            if not block.bbox:
                continue
            
            x0, y0, x1, y1 = block.bbox
            
            # Scale from PDF points to image pixels
            x0_px = int(x0 * scale_factor)
            y0_px = int(y0 * scale_factor)
            x1_px = int(x1 * scale_factor)
            y1_px = int(y1 * scale_factor)
            
            # Draw rectangle (use different colors for different types)
            color = (255, 0, 0)  # Red for pdfplumber boxes
            draw.rectangle([x0_px, y0_px, x1_px, y1_px], outline=color, width=2)
            
            # Add text label (truncate if too long)
            label = block.text[:30] if len(block.text) > 30 else block.text
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 10)
            except:
                font = ImageFont.load_default()
            
            # Draw text above box
            draw.text((x0_px, max(0, y0_px - 15)), label, fill=color, font=font)
        
        # Save visualization
        output_file = output_dir / f"ocr_bboxes_pdfplumber_page_{page_num}.png"
        img_pil.save(output_file)
        print(f"  Saved pdfplumber bbox visualization: {output_file}")


def visualize_ocr_bboxes_tesseract(
    pdf_path: str,
    output_dir: Path,
    language: str = "nld",
) -> None:
    """
    Visualize bounding boxes from Tesseract OCR.
    
    Args:
        pdf_path: Path to PDF file
        output_dir: Directory to save visualization images
        language: OCR language code
    """
    if not PIL_AVAILABLE or not TESSERACT_AVAILABLE:
        return  # Silently skip if dependencies not available
    
    try:
        from pdf2image import convert_from_path
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            images = convert_from_path(pdf_path, dpi=200)
    except ImportError:
        return  # Silently skip if pdf2image not available
    except Exception as e:
        print(f"Warning: Could not convert PDF to images: {e}")
        return
    
    for page_num, image in enumerate(images, 1):
        try:
            # Get detailed OCR data with bounding boxes
            ocr_data = pytesseract.image_to_data(
                image, lang=language, output_type=pytesseract.Output.DICT
            )
            
            # Convert PIL image to numpy array for drawing
            img_array = np.array(image)
            img_pil = Image.fromarray(img_array)
            draw = ImageDraw.Draw(img_pil)
            
            # Draw bounding boxes for each detected word
            n_boxes = len(ocr_data['text'])
            for i in range(n_boxes):
                text = ocr_data['text'][i].strip()
                conf = int(ocr_data['conf'][i])
                
                # Skip empty text or low confidence
                if not text or conf < 30:
                    continue
                
                # Get bounding box coordinates
                x = ocr_data['left'][i]
                y = ocr_data['top'][i]
                w = ocr_data['width'][i]
                h = ocr_data['height'][i]
                
                # Draw rectangle
                color = (0, 255, 0)  # Green for Tesseract boxes
                draw.rectangle([x, y, x + w, y + h], outline=color, width=1)
                
                # Add text label for high-confidence detections
                if conf > 60 and len(text) < 20:
                    try:
                        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 8)
                    except:
                        font = ImageFont.load_default()
                    draw.text((x, max(0, y - 12)), text, fill=color, font=font)
            
            # Save visualization
            output_file = output_dir / f"ocr_bboxes_tesseract_page_{page_num}.png"
            img_pil.save(output_file)
            print(f"  Saved Tesseract bbox visualization: {output_file}")
            
        except Exception as e:
            print(f"Warning: Could not visualize Tesseract bboxes for page {page_num}: {e}")


def visualize_ocr_bboxes_easyocr(
    pdf_path: str,
    output_dir: Path,
    ocr_reader: Optional[object] = None,
) -> None:
    """
    Visualize bounding boxes from EasyOCR.
    
    Args:
        pdf_path: Path to PDF file
        output_dir: Directory to save visualization images
        ocr_reader: EasyOCR Reader instance (optional, will create if not provided)
    """
    if not PIL_AVAILABLE or not EASYOCR_AVAILABLE:
        return  # Silently skip if dependencies not available
    
    try:
        from pdf2image import convert_from_path
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            images = convert_from_path(pdf_path, dpi=200)
    except ImportError:
        return  # Silently skip if pdf2image not available
    except Exception as e:
        print(f"Warning: Could not convert PDF to images: {e}")
        return
    
    # Initialize EasyOCR reader if not provided
    if ocr_reader is None:
        try:
            import ssl
            ssl._create_default_https_context = ssl._create_unverified_context
            ocr_reader = easyocr.Reader(['nl'], gpu=False)
        except Exception as e:
            print(f"Warning: Could not initialize EasyOCR reader: {e}")
            return
    
    for page_num, image in enumerate(images, 1):
        try:
            # Run OCR to get bounding boxes
            results = ocr_reader.readtext(np.array(image))
            
            # Convert PIL image to numpy array for drawing
            img_array = np.array(image)
            img_pil = Image.fromarray(img_array)
            draw = ImageDraw.Draw(img_pil)
            
            # Draw bounding boxes for each detected text
            for result in results:
                bbox = result[0]  # List of 4 corner points
                text = result[1]
                conf = result[2]
                
                # Skip low confidence
                if conf < 0.3:
                    continue
                
                # Convert bbox to rectangle (x0, y0, x1, y1)
                x_coords = [point[0] for point in bbox]
                y_coords = [point[1] for point in bbox]
                x0 = int(min(x_coords))
                y0 = int(min(y_coords))
                x1 = int(max(x_coords))
                y1 = int(max(y_coords))
                
                # Draw rectangle
                color = (0, 0, 255)  # Blue for EasyOCR boxes
                draw.rectangle([x0, y0, x1, y1], outline=color, width=2)
                
                # Add text label
                if conf > 0.6 and len(text) < 30:
                    try:
                        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 10)
                    except:
                        font = ImageFont.load_default()
                    draw.text((x0, max(0, y0 - 15)), text[:20], fill=color, font=font)
            
            # Save visualization
            output_file = output_dir / f"ocr_bboxes_easyocr_page_{page_num}.png"
            img_pil.save(output_file)
            print(f"  Saved EasyOCR bbox visualization: {output_file}")
            
        except Exception as e:
            print(f"Warning: Could not visualize EasyOCR bboxes for page {page_num}: {e}")


def visualize_ocr_bboxes_paddleocr(
    pdf_path: str,
    output_dir: Path,
    ocr_reader: Optional[object] = None,
    language: str = "nl",
) -> None:
    """
    Visualize bounding boxes from PaddleOCR.
    
    Args:
        pdf_path: Path to PDF file
        output_dir: Directory to save visualization images
        ocr_reader: PaddleOCR instance (optional, will create if not provided)
        language: OCR language code (default: "nl" for Dutch)
    """
    if not PIL_AVAILABLE or not PADDLEOCR_AVAILABLE:
        return  # Silently skip if dependencies not available
    
    try:
        from pdf2image import convert_from_path
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            images = convert_from_path(pdf_path, dpi=200)
    except ImportError:
        return  # Silently skip if pdf2image not available
    except Exception as e:
        print(f"Warning: Could not convert PDF to images: {e}")
        return
    
    # Initialize PaddleOCR reader if not provided
    if ocr_reader is None:
        try:
            ocr_reader = PaddleOCR(lang=language)
        except Exception as e:
            print(f"Warning: Could not initialize PaddleOCR reader: {e}")
            return
    
    for page_num, image in enumerate(images, 1):
        try:
            # Run OCR to get bounding boxes
            results = ocr_reader.ocr(np.array(image))
            
            # Convert PIL image to numpy array for drawing
            img_array = np.array(image)
            img_pil = Image.fromarray(img_array)
            draw = ImageDraw.Draw(img_pil)
            
            # PaddleOCR returns OCRResult objects in newer versions
            # Try to extract the actual OCR data
            ocr_data = None
            if results and len(results) > 0:
                result_obj = results[0]
                # Try to get OCR data from OCRResult object
                if hasattr(result_obj, 'ocr_res'):
                    ocr_data = result_obj.ocr_res
                elif isinstance(result_obj, dict) and 'ocr_res' in result_obj:
                    ocr_data = result_obj['ocr_res']
                elif hasattr(result_obj, '__getitem__'):
                    try:
                        ocr_data = result_obj['ocr_res'] if 'ocr_res' in result_obj else result_obj
                    except:
                        ocr_data = result_obj
                else:
                    ocr_data = result_obj
                
                # Handle OCRResult - new PaddleOCR format has dt_polys, rec_texts, rec_scores
                dt_polys = None
                rec_texts = None
                rec_scores = None
                
                # Try to extract from OCRResult object
                if ocr_data:
                    if isinstance(ocr_data, dict):
                        dt_polys = ocr_data.get("dt_polys")
                        rec_texts = ocr_data.get("rec_texts")
                        rec_scores = ocr_data.get("rec_scores")
                    elif hasattr(ocr_data, "__getitem__"):
                        try:
                            dt_polys = ocr_data.get("dt_polys") if hasattr(ocr_data, "get") else (ocr_data["dt_polys"] if "dt_polys" in ocr_data else None)
                            rec_texts = ocr_data.get("rec_texts") if hasattr(ocr_data, "get") else (ocr_data["rec_texts"] if "rec_texts" in ocr_data else None)
                            rec_scores = ocr_data.get("rec_scores") if hasattr(ocr_data, "get") else (ocr_data["rec_scores"] if "rec_scores" in ocr_data else None)
                        except:
                            pass
                    # Fallback: try old format
                    elif isinstance(ocr_data, list):
                        # Old format: [[[[x,y], ...], (text, conf)], ...]
                        pass
                
                # Process new format (dt_polys, rec_texts, rec_scores)
                if dt_polys is not None and rec_texts is not None:
                    if not isinstance(dt_polys, list):
                        dt_polys = [dt_polys]
                    if not isinstance(rec_texts, list):
                        rec_texts = [rec_texts]
                    if rec_scores is None or not isinstance(rec_scores, list):
                        rec_scores = [1.0] * len(rec_texts)
                    
                    # Draw bounding boxes for each detected text
                    for i, (bbox_poly, text) in enumerate(zip(dt_polys, rec_texts)):
                        if not text or not text.strip():
                            continue
                        
                        try:
                            # Get confidence
                            conf = rec_scores[i] if i < len(rec_scores) else 1.0
                            
                            # Skip low confidence
                            if conf < 0.3:
                                continue
                            
                            # Convert numpy array to list if needed
                            if hasattr(bbox_poly, 'tolist'):
                                bbox_coords = bbox_poly.tolist()
                            elif isinstance(bbox_poly, list):
                                bbox_coords = bbox_poly
                            else:
                                continue
                            
                            # Convert bbox to rectangle (x0, y0, x1, y1)
                            # PaddleOCR returns coordinates in image pixels, use directly
                            x_coords = []
                            y_coords = []
                            for pt in bbox_coords:
                                if hasattr(pt, '__getitem__') and len(pt) >= 2:
                                    x_coords.append(float(pt[0]))
                                    y_coords.append(float(pt[1]))
                                elif isinstance(pt, (list, tuple)) and len(pt) >= 2:
                                    x_coords.append(float(pt[0]))
                                    y_coords.append(float(pt[1]))
                                else:
                                    continue
                            
                            if not x_coords or not y_coords:
                                continue
                                
                            x0 = int(min(x_coords))
                            y0 = int(min(y_coords))
                            x1 = int(max(x_coords))
                            y1 = int(max(y_coords))
                            
                            # Ensure coordinates are within image bounds
                            x0 = max(0, min(x0, image.width))
                            y0 = max(0, min(y0, image.height))
                            x1 = max(0, min(x1, image.width))
                            y1 = max(0, min(y1, image.height))
                            
                            # Draw rectangle
                            color = (255, 165, 0)  # Orange for PaddleOCR boxes
                            draw.rectangle([x0, y0, x1, y1], outline=color, width=2)
                            
                            # Add text label
                            if conf > 0.6 and len(text) < 30:
                                try:
                                    font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 10)
                                except:
                                    font = ImageFont.load_default()
                                draw.text((x0, max(0, y0 - 15)), text[:20], fill=color, font=font)
                        except Exception as e:
                            if debug:
                                print(f"Warning: Error drawing bbox for item {i}: {e}")
                            continue
                # Fallback: try old format
                elif isinstance(ocr_data, list):
                    # Draw bounding boxes for each detected text
                    for line_result in ocr_data:
                        if not line_result:
                            continue
                        try:
                            # Try different formats
                            if isinstance(line_result, list) and len(line_result) >= 2:
                                bbox_coords = line_result[0]
                                text_conf = line_result[1]
                                if isinstance(text_conf, tuple) and len(text_conf) >= 2:
                                    text, conf = text_conf[0], text_conf[1]
                                elif isinstance(text_conf, str):
                                    text, conf = text_conf, 1.0
                                else:
                                    continue
                            else:
                                continue
                            
                            # Skip low confidence
                            if conf < 0.3:
                                continue
                            
                            # Convert bbox to rectangle (x0, y0, x1, y1)
                            # Handle both list and numpy array formats
                            x_coords = []
                            y_coords = []
                            for point in bbox_coords:
                                if hasattr(point, '__getitem__') and len(point) >= 2:
                                    x_coords.append(float(point[0]))
                                    y_coords.append(float(point[1]))
                                elif isinstance(point, (list, tuple)) and len(point) >= 2:
                                    x_coords.append(float(point[0]))
                                    y_coords.append(float(point[1]))
                                else:
                                    continue
                            
                            if not x_coords or not y_coords:
                                continue
                                
                            x0 = int(min(x_coords))
                            y0 = int(min(y_coords))
                            x1 = int(max(x_coords))
                            y1 = int(max(y_coords))
                            
                            # Ensure coordinates are within image bounds
                            x0 = max(0, min(x0, image.width))
                            y0 = max(0, min(y0, image.height))
                            x1 = max(0, min(x1, image.width))
                            y1 = max(0, min(y1, image.height))
                            
                            # Draw rectangle
                            color = (255, 165, 0)  # Orange for PaddleOCR boxes
                            draw.rectangle([x0, y0, x1, y1], outline=color, width=2)
                            
                            # Add text label
                            if conf > 0.6 and len(text) < 30:
                                try:
                                    font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 10)
                                except:
                                    font = ImageFont.load_default()
                                draw.text((x0, max(0, y0 - 15)), text[:20], fill=color, font=font)
                        except Exception as e:
                            if debug:
                                print(f"Warning: Error drawing bbox for line_result: {e}")
                            continue
            
            # Save visualization
            output_file = output_dir / f"ocr_bboxes_paddleocr_page_{page_num}.png"
            img_pil.save(output_file)
            print(f"  Saved PaddleOCR bbox visualization: {output_file}")
            
        except Exception as e:
            print(f"Warning: Could not visualize PaddleOCR bboxes for page {page_num}: {e}")

