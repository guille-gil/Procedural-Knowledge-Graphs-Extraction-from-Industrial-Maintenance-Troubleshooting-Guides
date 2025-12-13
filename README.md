# Procedural Knowledge Extraction from Industrial Troubleshooting Guides

A minimal, modular pipeline for extracting procedural knowledge from industrial troubleshooting guides using vision language models (VLMs) and evaluating performance against manual annotations.

## Overview

This project processes annotated troubleshooting guides, runs several vision language models in zero-shot or few-shot mode, extracts procedural entities and relations according to a fixed label set, and evaluates performance against manual annotations and a text-only baseline model.

## Project Structure

```
.
├── data/
│   ├── pdf/              # Input PDF files (TOCAPs)
│   └── annotations/      # Gold standard JSON annotations
├── src/
│   ├── data/
│   │   └── loader.py     # PDF and annotation loading
│   ├── labels/
│   │   └── labels.py      # Label definitions (loads from schemas/labels.yaml)
│   ├── inference/
│   │   ├── vlm_inference.py        # VLM inference (Pixtral, InternVL2, Llava)
│   │   └── baseline_text_model.py  # Text-only baseline
│   ├── evaluation/
│   │   ├── metrics.py     # Evaluation metrics (precision, recall, F1)
│   │   └── run_evaluation.py  # Evaluation runner
│   ├── utils/
│   │   └── visualization.py  # Graph visualization utilities
│   └── main.py           # Main pipeline orchestrator
├── output/
│   ├── graphs/           # Model predictions (JSON)
│   └── evaluations/      # Evaluation results
├── schemas/
│   ├── labels.yaml       # Entity and relation type definitions
│   └── prompts.yaml      # Zero-shot and few-shot prompts
└── requirements.txt      # Python dependencies
```

## Entity and Relation Types

### Entity Types
- **Condition**: A condition or state that must be checked
- **Action**: An action or step to be performed
- **Decision**: A decision point with multiple possible outcomes
- **Observation**: An observation or measurement result
- **Component**: A physical component or part of the system

### Relation Types
- **leads_to**: Indicates that one entity leads to or causes another
- **applies_to_component**: Indicates that an action or condition applies to a specific component
- **has_outcome**: Indicates that a decision or condition has a specific outcome

## Annotation Format

Gold standard annotations are stored as JSON files with the following structure:

```json
{
  "entities": [
    {
      "id": "E1",
      "type": "Condition",
      "text": "...",
      "bbox": [x1, y1, x2, y2],
      "page": 1
    }
  ],
  "relations": [
    {
      "source": "E1",
      "target": "E2",
      "type": "leads_to"
    }
  ]
}
```

## Models

### Vision Language Models (VLMs)
- **Pixtral 12B Vision Instruct**: Large vision-language model
- **InternVL2 8B**: Efficient vision-language model
- **Llava One Vision**: Lightweight vision-language model

### Baseline
- **Mistral 7B Instruct** (text-only): Processes extracted PDF text without images

## Installation

1. Clone the repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Run the full pipeline with all models:

```bash
python src/main.py --data_dir data --output_dir output
```

### Run Specific Models

```bash
python src/main.py --models pixtral internvl --mode zero_shot
```

### Run Only Evaluation

If predictions already exist:

```bash
python src/main.py --skip_inference
```

### Command Line Options

- `--data_dir`: Data directory (default: `data`)
- `--output_dir`: Output directory (default: `output`)
- `--models`: Models to run: `pixtral`, `internvl`, `llava`, `text_baseline`, or `all` (default: `all`)
- `--mode`: Inference mode: `zero_shot`, `few_shot`, or `both` (default: `both`)
- `--device`: Device: `auto`, `cuda`, or `cpu` (default: `auto`)
- `--skip_inference`: Skip inference and only run evaluation

## Running on Habrok HPC

The code is designed to run on both local machines and the Habrok HPC cluster at University of Groningen.

### Example Slurm Script

```bash
#!/bin/bash
#SBATCH --job-name=vlm_extraction
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=32G

module load Python/3.10.4-GCCcore-11.3.0
source venv/bin/activate

python src/main.py --data_dir data --output_dir output --models pixtral --mode zero_shot
```

## Output

### Predictions
Model predictions are saved in `output/graphs/` as JSON files:
- `{guide_name}_{model_name}_zero_shot.json`
- `{guide_name}_{model_name}_few_shot.json`

### Evaluations
Evaluation results are saved in `output/evaluations/all_results.json` and include:
- Entity precision, recall, F1
- Relation precision, recall, F1
- Overall graph reconstruction F1

## Evaluation Metrics

The pipeline computes:
1. **Entity Metrics**: Precision, recall, F1 for entity extraction (matching by type and text similarity > 0.7)
2. **Relation Metrics**: Precision, recall, F1 for relation extraction (matching by type and aligned entity IDs)
3. **Graph Reconstruction Score**: Overall F1 combining entity and relation metrics

## License

MIT License - see LICENSE file for details.

