#!/usr/bin/env python3
"""
Pipeline Orchestrator for Hybrid Procedural Knowledge Extraction

This script orchestrates both spatial and semantic extractors to build
a complete knowledge graph from PDF troubleshooting guides.

Usage: python pipeline_orchestrator.py [PDF_PATH] [OPTIONS]
"""

import sys
import json
import argparse
import shutil
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.spatial_extractor import SpatialExtractor, TextBlock
from src.semantic_extractor import SemanticExtractor
from src.visualize_graph import visualize_graph


def clear_output_directories(output_dirs: list):
    """
    Clear output directories before running pipeline.

    Args:
        output_dirs: List of directory paths to clear
    """
    for output_dir in output_dirs:
        if output_dir.exists():
            # Remove all files and subdirectories
            for item in output_dir.iterdir():
                try:
                    if item.is_file():
                        item.unlink()
                    elif item.is_dir():
                        shutil.rmtree(item)
                except Exception as e:
                    print(f"    Warning: Could not remove {item}: {e}")
            print(f"  Cleared: {output_dir}")
        else:
            # Create directory if it doesn't exist
            output_dir.mkdir(parents=True, exist_ok=True)


def combine_results(
    semantic_units: list,
    semantic_entities: list,
    semantic_relations: list,
    spatial_info: dict,
    spatial_relations: list,
) -> dict:
    """
    Combine semantic and spatial extraction results into unified graph.

    Args:
        semantic_units: List of ProceduralUnit objects from semantic extractor
        semantic_entities: List of Entity objects from semantic extractor
        semantic_relations: List of Relation objects from semantic extractor
        spatial_info: Dict mapping unit_id to SpatialInfo
        spatial_relations: List of spatial relation dicts

    Returns:
        Unified graph dictionary
    """
    from dataclasses import asdict

    # Build unified nodes
    nodes = []

    # Add procedural units with spatial info (skip PageConnectors - they're handled separately)
    for unit in semantic_units:
        # Skip PageConnectors - they're only used for linking, not as nodes
        if unit.type == "PageConnector":
            continue

        unit_dict = {
            "id": unit.id,
            "type": unit.type,
            "label": unit.text,
            "page": unit.page,
            "line": unit.line_number,
            "confidence": unit.confidence,
        }

        # Add spatial information if available
        if unit.id in spatial_info:
            spatial = spatial_info[unit.id]
            if spatial.bbox:
                unit_dict["bbox"] = spatial.bbox
            if spatial.centroid:
                unit_dict["centroid"] = spatial.centroid
            if spatial.inferred_shape:
                unit_dict["inferred_shape"] = spatial.inferred_shape
            if spatial.column_position:
                unit_dict["column_position"] = spatial.column_position

        nodes.append(unit_dict)

    # Add entities
    for entity in semantic_entities:
        nodes.append(
            {
                "id": entity.id,
                "type": entity.type,
                "label": entity.text,
                "mentions": len(entity.mentions),
                "confidence": entity.confidence,
            }
        )

    # Build unified edges
    edges = []

    # Add semantic relations
    for rel in semantic_relations:
        edge_dict = {
            "source": rel.source,
            "target": rel.target,
            "relation": rel.relation_type,
            "confidence": rel.confidence,
            "evidence": rel.evidence or "",
        }
        edges.append(edge_dict)

    # Add spatial relations (merge with semantic if duplicate)
    for spatial_rel in spatial_relations:
        # Check if relation already exists
        existing = next(
            (
                e
                for e in edges
                if e["source"] == spatial_rel["source"]
                and e["target"] == spatial_rel["target"]
            ),
            None,
        )

        if existing:
            # Merge: increase confidence if spatial agrees
            existing["confidence"] = min(0.95, existing["confidence"] + 0.05)
            if "spatial_distance" in spatial_rel:
                existing["evidence"] += (
                    f", spatial: {spatial_rel['spatial_distance']:.0f}pt"
                )
        else:
            # Add new spatial relation
            edges.append(spatial_rel)

    # Handle page connectors: Link pages via connector numbers (1, 2, etc.)
    # Page connectors are NOT regular nodes - they're just for linking pages
    page_connectors = {}  # {connector_num: {page: node_id}}

    # Find all page connectors
    for unit in semantic_units:
        if unit.type == "PageConnector":
            connector_num = unit.text.strip()
            if connector_num not in page_connectors:
                page_connectors[connector_num] = {}
            page_connectors[connector_num][unit.page] = unit.id

    # Create page continuity edges
    for connector_num, page_dict in page_connectors.items():
        pages = sorted(page_dict.keys())
        for i in range(len(pages) - 1):
            current_page = pages[i]
            next_page = pages[i + 1]

            # Find the last non-connector node on current page
            last_node_on_page = None
            for node in nodes:
                if node.get("page") == current_page and node.get("type") not in [
                    "PageConnector",
                    "Connector",
                ]:
                    if not last_node_on_page or node.get(
                        "line", 0
                    ) > last_node_on_page.get("line", 0):
                        last_node_on_page = node

            # Find the first non-connector node on next page
            first_node_on_next_page = None
            for node in nodes:
                if node.get("page") == next_page and node.get("type") not in [
                    "PageConnector",
                    "Connector",
                ]:
                    if not first_node_on_next_page or node.get(
                        "line", 0
                    ) < first_node_on_next_page.get("line", 0):
                        first_node_on_next_page = node

            # Create edge: last node on page N → first node on page N+1
            if last_node_on_page and first_node_on_next_page:
                edges.append(
                    {
                        "source": last_node_on_page["id"],
                        "target": first_node_on_next_page["id"],
                        "relation": "leads_to",
                        "confidence": 0.9,
                        "evidence": f"Page continuity via connector {connector_num} (page {current_page} → {next_page})",
                    }
                )

    # Remove PageConnector nodes from final graph (they're just for linking)
    # Also filter out metadata nodes
    nodes = [
        n
        for n in nodes
        if n.get("type") != "PageConnector"
        and not any(
            meta in n.get("label", "").lower()
            for meta in [
                "naam:",
                "periode:",
                "unit:",
                "inhoudsopgave",
                "opmerkingen",
                "auteur:",
                "versiedatum:",
                "laatste wijzigingen",
            ]
        )
    ]

    return {
        "nodes": nodes,
        "edges": edges,
        "metadata": {
            "ontology_version": "1.0",
            "total_units": len(semantic_units),
            "total_entities": len(semantic_entities),
            "total_relations": len(edges),
        },
    }


def main():
    """Run the complete hybrid extraction pipeline."""
    parser = argparse.ArgumentParser(
        description="Extract procedural knowledge from PDF troubleshooting guides using hybrid spatial+semantic approach"
    )
    parser.add_argument(
        "pdf_path",
        nargs="?",
        type=str,
        help="Path to PDF file (default: data/raw/TOCAPs/DP17-TOCAP4.pdf)",
    )
    parser.add_argument(
        "--ontology",
        type=str,
        default=None,
        help="Path to ontology YAML schema (default: ontology.yaml)",
    )
    parser.add_argument(
        "--ocr-engine",
        choices=["paddleocr"],
        default="paddleocr",
        help="OCR engine (only PaddleOCR is supported)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for intermediate outputs (default: data/intermediate)",
    )

    args = parser.parse_args()

    # Default PDF path
    if args.pdf_path:
        pdf_path = Path(args.pdf_path)
    else:
        pdf_path = project_root / "data" / "raw" / "TOCAPs" / "DP17-TOCAP4.pdf"

    # Check if PDF exists
    if not pdf_path.exists():
        print(f"ERROR: PDF file not found: {pdf_path}")
        print(f"\nUsage: python pipeline_orchestrator.py [PDF_PATH] [OPTIONS]")
        print(
            f"Example: python pipeline_orchestrator.py data/raw/TOCAPs/DP17-TOCAP4.pdf"
        )
        sys.exit(1)

    # Default ontology path
    if args.ontology:
        ontology_path = Path(args.ontology)
    else:
        ontology_path = project_root / "ontology.yaml"

    if not ontology_path.exists():
        print(f"Warning: Ontology file not found: {ontology_path}")
        print("Using default ontology schema.")
        ontology_path = None

    print("=" * 60)
    print("Hybrid Procedural Knowledge Extraction Pipeline")
    print("=" * 60)
    print(f"Processing: {pdf_path.name}")
    print("-" * 60)

    # Clear output directories
    output_dir = project_root / "data" / "processed"
    intermediate_dir = project_root / "data" / "intermediate"

    print("\nClearing previous outputs...")
    clear_output_directories([output_dir, intermediate_dir])
    output_dir.mkdir(parents=True, exist_ok=True)
    intermediate_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Step 1: Text extraction using PaddleOCR (semantic extractor handles this)
        print("\nStep 1: Text extraction with PaddleOCR...")
        semantic_extractor = SemanticExtractor(
            ocr_engine=args.ocr_engine,
            language="nld",
            debug=True,  # Always enable debug mode
            output_dir=args.output_dir if args.output_dir else str(intermediate_dir),
        )

        # Extract text blocks using PaddleOCR (this is where the actual extraction happens)
        text_blocks = semantic_extractor.extract_text_blocks(str(pdf_path))
        print(f"  Extracted {len(text_blocks)} text blocks")

        # Step 2: Spatial analysis (layout inference, shape detection)
        print("\nStep 2: Spatial analysis (layout inference)...")
        spatial_extractor = SpatialExtractor(
            debug=True,  # Always enable debug mode
            output_dir=args.output_dir if args.output_dir else str(intermediate_dir),
        )

        # Get page widths from semantic extractor (stored during extraction)
        page_widths = getattr(semantic_extractor, "_page_widths", {})
        if not page_widths:
            # Fallback: estimate from text blocks
            for block in text_blocks:
                if block.bbox:
                    # Estimate page width from bbox x1 coordinates
                    if block.page not in page_widths:
                        page_widths[block.page] = block.bbox[2] * 1.1  # Add 10% margin
                    else:
                        page_widths[block.page] = max(
                            page_widths[block.page], block.bbox[2] * 1.1
                        )

        # Infer spatial information
        spatial_info = spatial_extractor.infer_spatial_info(text_blocks, page_widths)
        print(f"  Inferred spatial info for {len(spatial_info)} blocks")

        # Step 3: Semantic extraction (NLP classification, entity extraction, semantic relations)
        print("\nStep 3: Semantic extraction (NLP analysis)...")

        # Use text blocks from PaddleOCR extraction
        semantic_text_blocks = text_blocks

        # Segment into procedural units (semantic classification)
        units = semantic_extractor.segment_procedural_units(semantic_text_blocks)

        # Enhance units with spatial info
        for unit in units:
            if unit.id in spatial_info:
                spatial = spatial_info[unit.id]
                unit.bbox = spatial.bbox
                unit.centroid = spatial.centroid
                unit.inferred_shape = spatial.inferred_shape

        print(f"  Classified {len(units)} procedural units")

        # Extract entities
        entities = semantic_extractor.extract_entities(semantic_text_blocks)
        print(f"  Extracted {len(entities)} entities")

        # Visualize PaddleOCR bounding boxes (only PaddleOCR is used)
        print("\n  Generating PaddleOCR bounding box visualizations...")
        try:
            from src.visualize_ocr import visualize_ocr_bboxes_paddleocr

            # Always use PaddleOCR - no other options
            try:
                # Use the initialized PaddleOCR reader from semantic extractor
                if (
                    not hasattr(semantic_extractor, "_paddleocr_reader")
                    or semantic_extractor._paddleocr_reader is None
                ):
                    from paddleocr import PaddleOCR

                    ocr_reader = PaddleOCR(lang="nl")
                else:
                    ocr_reader = semantic_extractor._paddleocr_reader

                visualize_ocr_bboxes_paddleocr(
                    str(pdf_path),
                    intermediate_dir,
                    ocr_reader=ocr_reader,
                    language="nl",
                )
            except Exception as e:
                print(f"    PaddleOCR visualization failed: {e}")
        except ImportError as e:
            print(f"    Visualization skipped (missing dependencies): {e}")

        # Infer semantic relations
        semantic_relations = semantic_extractor.infer_relations(units, entities)
        print(f"  Inferred {len(semantic_relations)} semantic relations")

        # Step 4: Spatial relation inference
        print("\nStep 4: Spatial relation inference...")
        # Convert units to dict format for spatial extractor
        units_dict = [
            {
                "id": unit.id,
                "type": unit.type,
                "centroid": unit.centroid,
                "bbox": unit.bbox,
                "page": unit.page,
            }
            for unit in units
        ]
        spatial_relations = spatial_extractor.infer_spatial_relations(units_dict)
        print(f"  Inferred {len(spatial_relations)} spatial relations")

        # Step 5: Combine results
        print("\nStep 5: Combining results...")
        graph = combine_results(
            units, entities, semantic_relations, spatial_info, spatial_relations
        )

        # Step 6: Build ontology graph and export
        print("\nStep 6: Building ontology graph...")
        final_graph = semantic_extractor.build_ontology_graph(
            units,
            entities,
            semantic_relations,
            str(ontology_path) if ontology_path else None,
        )

        # Merge spatial info into final graph
        for node in final_graph["nodes"]:
            node_id = node["id"]
            if node_id in spatial_info:
                spatial = spatial_info[node_id]
                if spatial.bbox:
                    node["bbox"] = spatial.bbox
                if spatial.centroid:
                    node["centroid"] = spatial.centroid
                if spatial.inferred_shape:
                    node["inferred_shape"] = spatial.inferred_shape

        # Save final output
        output_file = output_dir / f"semantic_graph_{pdf_path.stem}.json"

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(final_graph, f, indent=2, ensure_ascii=False)

        print(f"\nFinal graph saved to: {output_file}")
        print(f"\nSummary:")
        print(f"  - Procedural Units: {final_graph['metadata']['total_units']}")
        print(f"  - Entities: {final_graph['metadata']['total_entities']}")
        print(f"  - Relations: {final_graph['metadata']['total_relations']}")
        print(f"  - Total Nodes: {len(final_graph['nodes'])}")
        print(f"  - Total Edges: {len(final_graph['edges'])}")

        # Create visualization
        try:
            viz_path = output_dir / f"graph_visualization_{pdf_path.stem}.html"
            visualize_graph(final_graph, str(viz_path))
            print(f"\nVisualization saved to: {viz_path}")
            print(f"Open in browser to view the interactive graph.")
        except ImportError as viz_error:
            print(f"\nNote: Visualization skipped ({viz_error})")
            print("Install pyvis for visualization: pip install pyvis")
        except Exception as viz_error:
            print(f"\nNote: Visualization failed: {viz_error}")

        print("\n" + "=" * 60)
        print("Pipeline complete!")
        print("=" * 60)

    except ImportError as e:
        print(f"\nERROR: Missing Python dependency - {e}")
        print("\nPlease install dependencies:")
        print("  pip install -r requirements.txt")
        print("\nFor SpaCy Dutch model:")
        print("  python -m spacy download nl_core_news_sm")
        sys.exit(1)
    except Exception as e:
        error_msg = str(e)
        if "poppler" in error_msg.lower() or "pdfinfo" in error_msg.lower():
            print(f"\nERROR: Poppler is not installed or not in PATH")
            print("\nPoppler is a system dependency required by pdf2image.")
            print("\nInstall it:")
            print("  macOS:  brew install poppler")
            print("  Linux:  sudo apt-get install poppler-utils  (Ubuntu/Debian)")
            print("          sudo dnf install poppler-utils       (Fedora/RHEL)")
            print(
                "  Windows: Download from https://github.com/oschwartz10612/poppler-windows/releases/"
            )
            sys.exit(1)
        elif "tesseract" in error_msg.lower():
            print(f"\nERROR: Tesseract is not installed or not in PATH")
            print("\nInstall Tesseract:")
            print("  macOS:  brew install tesseract tesseract-lang")
            print("  Linux:  sudo apt-get install tesseract-ocr tesseract-ocr-nld")
            print(
                "  Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki"
            )
            sys.exit(1)
        else:
            print(f"\nERROR: {e}")
            import traceback

            traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    main()
