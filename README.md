# Procedural Knowledge Graphs Extraction from Industrial Maintenance Troubleshooting Guides

This project extracts **procedural knowledge graphs** from industrial troubleshooting PDFs (TOCAPs/OCAPs) using a **hybrid approach** combining **spatial/layout analysis** and **semantic NLP techniques**. The pipeline constructs a procedural knowledge ontology capturing conditions, actions, components, and their relationships.

## Overview

The extraction pipeline uses a **two-stage hybrid approach**:

1. **Spatial Extractor**: Handles PDF text extraction with bounding boxes, spatial grouping, and layout-based inference (column positions, shape types)
2. **Semantic Extractor**: Performs NLP-based classification, entity extraction, and semantic relation inference

The pipeline identifies:
- **Conditions**: Questions and checks that must be evaluated
- **Actions**: Procedures and steps to be performed
- **Observations**: Descriptive states and measurements
- **Components**: Physical parts, systems, and entities
- **Relations**: Semantic and spatial links between procedural units (leads_to, affects, requires_check)

## Project Structure

```
.
├── pipeline_orchestrator.py     # Main entry point (orchestrates spatial + semantic)
├── ontology.yaml                # Ontology schema definition
├── src/
│   ├── __init__.py
│   ├── spatial_extractor.py    # Spatial/layout extraction module
│   ├── semantic_extractor.py   # Semantic NLP extraction module
│   ├── visualize_graph.py      # Graph visualization
│   └── visualize_ocr.py        # OCR bounding box visualization (debug)
├── data/
│   ├── raw/
│   │   └── TOCAPs/              # PDF files (gitignored)
│   ├── processed/               # Extracted graphs (gitignored)
│   ├── intermediate/            # Intermediate outputs (gitignored)
│   └── glossary/
│       └── components.json     # Domain component glossary
├── docs/
│   └── PRIVACY.md              # Privacy and security documentation
├── requirements.txt            # Python dependencies
└── README.md
```

## Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd Procedural-Knowledge-Graphs-Extraction-from-Industrial-Maintenance-Troubleshooting-Guides
```

### 2. Install system dependencies

**Poppler** (required for PDF to image conversion):
- **macOS**: `brew install poppler`
- **Linux (Ubuntu/Debian)**: `sudo apt-get install poppler-utils`
- **Linux (Fedora/RHEL)**: `sudo dnf install poppler-utils`
- **Windows**: Download from [poppler-windows](https://github.com/oschwartz10612/poppler-windows/releases/)

**Tesseract OCR** (optional, if using tesseract engine):
- **macOS**: `brew install tesseract tesseract-lang`
- **Linux**: `sudo apt-get install tesseract-ocr tesseract-ocr-nld`
- **Windows**: Download from [UB-Mannheim/tesseract](https://github.com/UB-Mannheim/tesseract/wiki)

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 4. Download SpaCy Dutch model

```bash
python -m spacy download nl_core_news_sm
```

## Quick Start

### Basic Usage

```bash
# Extract procedural knowledge from a PDF
python pipeline_orchestrator.py data/raw/TOCAPs/DP17-TOCAP4.pdf

# The pipeline automatically generates debug outputs including:
# - OCR bounding box visualizations in data/intermediate/
# - Files: ocr_bboxes_pdfplumber_page_N.png, ocr_bboxes_tesseract_page_N.png, etc.
# - Intermediate JSON files for debugging

# Using EasyOCR instead of Tesseract (for OCR fallback)
python pipeline_orchestrator.py data/raw/TOCAPs/DP17-TOCAP4.pdf --ocr-engine easyocr

# Specify custom ontology schema
python pipeline_orchestrator.py data/raw/TOCAPs/DP17-TOCAP4.pdf --ontology custom_ontology.yaml
```

**Note**: The orchestrator automatically clears previous outputs from `data/processed/` and `data/intermediate/` (including debug visualizations) before each run.

### Python API

```python
from src.spatial_extractor import SpatialExtractor
from src.semantic_extractor import SemanticExtractor

# Initialize extractors
spatial_extractor = SpatialExtractor(debug=True)
semantic_extractor = SemanticExtractor(
    ocr_engine="tesseract",  # or "easyocr"
    language="nld",
    debug=True
)

# Step 1: Extract text with spatial information
text_blocks, page_widths = spatial_extractor.extract_text_blocks("path/to/file.pdf")
spatial_info = spatial_extractor.infer_spatial_info(text_blocks, page_widths)

# Step 2: Semantic classification and entity extraction
units = semantic_extractor.segment_procedural_units(text_blocks)
entities = semantic_extractor.extract_entities(text_blocks)
relations = semantic_extractor.infer_relations(units, entities)

# Step 3: Build final graph
graph = semantic_extractor.build_ontology_graph(units, entities, relations)
```

Or use the orchestrator:

```python
# Run full hybrid pipeline
graph = extractor.extract_full_pipeline(
    pdf_path="data/raw/TOCAPs/DP17-TOCAP4.pdf",
    ontology_path="ontology.yaml"
)

# Access results
print(f"Procedural Units: {graph['metadata']['total_units']}")
print(f"Entities: {graph['metadata']['total_entities']}")
print(f"Relations: {graph['metadata']['total_relations']}")
```

## Pipeline Steps

The semantic extraction pipeline consists of 5 steps:

### Step 1: Semantic OCR Extraction
- Extracts text from PDF pages using OCR (Tesseract or EasyOCR)
- Preserves reading order
- Output: JSON with text lines per page

### Step 2: Step Segmentation
- Classifies each line using regex patterns and NLP (SpaCy)
- Types: Condition, Action, Observation, Connector
- Output: Structured procedural units with confidence scores

### Step 3: Entity & Component Extraction
- Extracts components using domain glossary
- Uses noun phrase extraction with SpaCy
- Output: Entity nodes with mention locations

### Step 4: Relation Inference (Weak Supervision)
- Applies heuristic rules to infer relations:
  - Condition → Action (adjacent within 3 lines)
  - Action → Component (action mentions component)
  - Condition → Result (contains result keywords)
  - Connector → Next step (sequential flow)
- Output: Relations with confidence scores and evidence

### Step 5: Ontology Construction
- Builds knowledge graph from units, entities, and relations
- Exports to RDF/Turtle (TTL) format
- Output: JSON graph + TTL ontology file

## Output Format

### JSON Graph Structure

```json
{
  "nodes": [
    {
      "id": "p1_l23",
      "type": "Condition",
      "label": "Is de vacuümdruk correct?",
      "page": 1,
      "line": 23,
      "confidence": 0.85
    },
    {
      "id": "entity_0",
      "type": "Component",
      "label": "vacuumfilter",
      "mentions": 3,
      "confidence": 0.9
    }
  ],
  "edges": [
    {
      "source": "p1_l23",
      "target": "p1_l24",
      "relation": "leads_to",
      "confidence": 0.8,
      "evidence": "Adjacent condition-action"
    }
  ],
  "metadata": {
    "ontology_version": "1.0",
    "total_units": 150,
    "total_entities": 45,
    "total_relations": 120
  }
}
```

### RDF/Turtle Output

The pipeline also generates `data/intermediate/ontology.ttl` in RDF/Turtle format for integration with knowledge graph systems (Neo4j, RDF stores, etc.).

### Interactive Graph Visualization

The pipeline automatically generates an interactive HTML visualization (`data/processed/graph_visualization_<pdf_name>.html`) that you can open in any web browser. The visualization shows:

- **Color-coded nodes** by type:
  - Red: Conditions
  - Teal: Actions
  - Light teal: Observations
  - Purple: Connectors
  - Pink: Components
- **Interactive features**: Click and drag nodes, zoom, pan, hover for details
- **Edge labels**: Shows relation types (leads_to, affects, etc.)

You can also generate visualizations manually:
```bash
python src/visualize_graph.py data/processed/semantic_graph_DP17-TOCAP4.json
```

## Ontology Schema

The ontology is defined in `ontology.yaml` with:
- **Classes**: Condition, Action, Observation, Component, Result, Connector
- **Relations**: leads_to, requires_check, affects, equivalent_to

Customize the schema by editing `ontology.yaml`.

## Component Glossary

Domain-specific components are defined in `data/glossary/components.json`. Add or modify entries to improve entity extraction for your domain.

## Privacy & Security

All processing is performed **100% locally and offline**. No data is sent to external services. The only network activity is:
- Initial download of SpaCy Dutch model (one-time, cached locally)
- EasyOCR model download (one-time, if using EasyOCR)

## Troubleshooting

### Tesseract not found
```bash
# macOS
brew install tesseract tesseract-lang

# Linux
sudo apt-get install tesseract-ocr tesseract-ocr-nld
```

### SpaCy Dutch model not found
```bash
python -m spacy download nl_core_news_sm
```

### Poppler not found
```bash
# macOS
brew install poppler

# Linux
sudo apt-get install poppler-utils
```

## Migration from Old Pipeline

The previous visual layout parsing approach (`flowchart_extractor.py`) has been **deprecated**. The new semantic extraction pipeline provides:
- Hybrid approach combining spatial layout analysis with semantic NLP
- Better understanding of procedural content through both layout and semantics
- More robust extraction across different PDF formats
- Structured ontology output for knowledge graph systems

## License

See LICENSE file for details.
