"""
Main pipeline for procedural knowledge extraction from industrial troubleshooting guides.
"""

import argparse
import json
import os
import sys
from typing import Dict, List

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.loader import load_all_guides
from src.inference.vlm_inference import load_pixtral, load_internvl, load_llava
from src.inference.baseline_text_model import load_text_model
from src.evaluation.run_evaluation import evaluate_all_models, print_summary_table


def run_zero_shot_inference(model, guide: Dict, model_name: str) -> Dict:
    """
    Run zero-shot inference on all pages of a guide.

    Args:
        model: Model instance (VLM or text model)
        guide: Guide dictionary
        model_name: Name of the model

    Returns:
        Dictionary with page-level predictions
    """
    predictions = {}

    if model_name == "text_baseline":
        # Text-only model
        for page_idx, text_page in enumerate(guide["text_pages"]):
            page_num = page_idx + 1
            text_content = text_page.get("text", "")

            print(f"    Processing page {page_num} (text-only)...")
            result = model.run_text_only_zero_shot(text_content)

            # Add page number to entities
            for entity in result.get("entities", []):
                entity["page"] = page_num

            predictions[str(page_num)] = result
    else:
        # VLM models
        for page_idx, page_image in enumerate(guide["pages"]):
            page_num = page_idx + 1

            print(f"    Processing page {page_num}...")
            result = model.run_zero_shot(page_image)

            # Add page number to entities
            for entity in result.get("entities", []):
                entity["page"] = page_num

            predictions[str(page_num)] = result

    return predictions


def run_few_shot_inference(
    model, guide: Dict, model_name: str, num_examples: int = 2
) -> Dict:
    """
    Run few-shot inference on all pages of a guide.

    Args:
        model: Model instance (VLM or text model)
        guide: Guide dictionary
        model_name: Name of the model
        num_examples: Number of example annotations to use

    Returns:
        Dictionary with page-level predictions
    """
    predictions = {}

    # Get example annotations from gold standard (if available)
    gold_entities = guide["annotations"].get("entities", [])
    gold_relations = guide["annotations"].get("relations", [])

    # Create example structure (simplified - use first few entities/relations)
    examples = []
    if gold_entities and gold_relations:
        # Create a simple example from gold data
        example_entities = gold_entities[: min(3, len(gold_entities))]
        example_relations = gold_relations[: min(2, len(gold_relations))]

        example = {
            "entities": [
                {
                    "text": e.get("text", ""),
                    "type": e.get("type", ""),
                    "bbox": e.get("bbox", []),
                }
                for e in example_entities
            ],
            "relations": [
                {
                    "source_text": r.get("source", ""),
                    "target_text": r.get("target", ""),
                    "type": r.get("type", ""),
                }
                for r in example_relations
            ],
        }
        examples.append(example)

    if model_name == "text_baseline":
        # Text-only model (few-shot not as applicable, but we can try)
        for page_idx, text_page in enumerate(guide["text_pages"]):
            page_num = page_idx + 1
            text_content = text_page.get("text", "")

            print(f"    Processing page {page_num} (text-only, few-shot)...")
            # For text model, few-shot is less applicable, use zero-shot
            result = model.run_text_only_zero_shot(text_content)

            for entity in result.get("entities", []):
                entity["page"] = page_num

            predictions[str(page_num)] = result
    else:
        # VLM models
        for page_idx, page_image in enumerate(guide["pages"]):
            page_num = page_idx + 1

            print(f"    Processing page {page_num} (few-shot)...")
            result = model.run_few_shot(page_image, examples)

            for entity in result.get("entities", []):
                entity["page"] = page_num

            predictions[str(page_num)] = result

    return predictions


def main():
    """Main pipeline execution."""
    parser = argparse.ArgumentParser(
        description="Procedural Knowledge Extraction Pipeline"
    )
    parser.add_argument("--data_dir", type=str, default="data", help="Data directory")
    parser.add_argument(
        "--output_dir", type=str, default="output", help="Output directory"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        help="Models to run: pixtral, internvl, llava, text_baseline, or all",
    )
    parser.add_argument(
        "--mode",
        choices=["zero_shot", "few_shot", "both"],
        default="both",
        help="Inference mode",
    )
    parser.add_argument(
        "--device", type=str, default="auto", help="Device: auto, cuda, or cpu"
    )
    parser.add_argument(
        "--skip_inference",
        action="store_true",
        help="Skip inference and only run evaluation",
    )

    args = parser.parse_args()

    # Create output directories
    os.makedirs(os.path.join(args.output_dir, "graphs"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "evaluations"), exist_ok=True)

    # Load all guides
    print("Loading guides...")
    guides = load_all_guides(args.data_dir)
    print(f"Loaded {len(guides)} guides")

    if not args.skip_inference:
        # Initialize models
        models = {}
        model_instances = {}

        if "all" in args.models or "pixtral" in args.models:
            print("\nLoading Pixtral 12B...")
            try:
                pixtral = load_pixtral(device=args.device)
                models["pixtral"] = pixtral
                model_instances["pixtral"] = pixtral
            except Exception as e:
                print(f"Failed to load Pixtral: {e}")

        if "all" in args.models or "internvl" in args.models:
            print("\nLoading InternVL2 8B...")
            try:
                internvl = load_internvl(device=args.device)
                models["internvl"] = internvl
                model_instances["internvl"] = internvl
            except Exception as e:
                print(f"Failed to load InternVL2: {e}")

        if "all" in args.models or "llava" in args.models:
            print("\nLoading Llava One Vision...")
            try:
                llava = load_llava(device=args.device)
                models["llava"] = llava
                model_instances["llava"] = llava
            except Exception as e:
                print(f"Failed to load Llava: {e}")

        if "all" in args.models or "text_baseline" in args.models:
            print("\nLoading Text Baseline (Mistral)...")
            try:
                text_model = load_text_model(device=args.device)
                models["text_baseline"] = text_model
                model_instances["text_baseline"] = text_model
            except Exception as e:
                print(f"Failed to load text baseline: {e}")

        # Run inference
        print("\n" + "=" * 80)
        print("RUNNING INFERENCE")
        print("=" * 80)

        for model_name, model in models.items():
            print(f"\n{'=' * 80}")
            print(f"Model: {model_name}")
            print(f"{'=' * 80}")

            for guide in guides:
                print(f"\nProcessing guide: {guide['name']}")

                # Zero-shot
                if args.mode in ["zero_shot", "both"]:
                    print("  Zero-shot inference...")
                    zero_shot_preds = run_zero_shot_inference(model, guide, model_name)

                    # Save predictions
                    output_file = os.path.join(
                        args.output_dir,
                        "graphs",
                        f"{guide['name']}_{model_name}_zero_shot.json",
                    )
                    with open(output_file, "w") as f:
                        json.dump(zero_shot_preds, f, indent=2)

                # Few-shot
                if args.mode in ["few_shot", "both"]:
                    print("  Few-shot inference...")
                    few_shot_preds = run_few_shot_inference(model, guide, model_name)

                    # Save predictions
                    output_file = os.path.join(
                        args.output_dir,
                        "graphs",
                        f"{guide['name']}_{model_name}_few_shot.json",
                    )
                    with open(output_file, "w") as f:
                        json.dump(few_shot_preds, f, indent=2)

    # Run evaluation
    print("\n" + "=" * 80)
    print("RUNNING EVALUATION")
    print("=" * 80)

    # Load model instances for evaluation (if inference was skipped, models need to be loaded)
    if args.skip_inference:
        # For evaluation, we just need the model names, not the actual instances
        model_instances = {name: None for name in args.models if name != "all"}
        if "all" in args.models:
            model_instances = {
                "pixtral": None,
                "internvl": None,
                "llava": None,
                "text_baseline": None,
            }

    # Evaluate each model
    all_results = []
    for model_name in model_instances.keys():
        print(f"\nEvaluating {model_name}...")

        for guide in guides:
            # Try both zero-shot and few-shot if both modes were run
            for mode in (
                ["zero_shot", "few_shot"] if args.mode == "both" else [args.mode]
            ):
                pred_file = os.path.join(
                    args.output_dir,
                    "graphs",
                    f"{guide['name']}_{model_name}_{mode}.json",
                )

                if not os.path.exists(pred_file):
                    continue

                with open(pred_file, "r") as f:
                    predictions = json.load(f)

                # Evaluate
                from src.evaluation.run_evaluation import evaluate_single_guide

                result = evaluate_single_guide(
                    f"{model_name}_{mode}", guide, predictions
                )
                all_results.append(result)

    # Print summary
    if all_results:
        print_summary_table(all_results)

        # Save results
        results_file = os.path.join(args.output_dir, "evaluations", "all_results.json")
        with open(results_file, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {results_file}")
    else:
        print("No results to evaluate. Make sure predictions exist in output/graphs/")

    print("\nPipeline completed!")


if __name__ == "__main__":
    main()
