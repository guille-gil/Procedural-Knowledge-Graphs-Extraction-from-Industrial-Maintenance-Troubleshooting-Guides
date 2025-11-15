"""
Evaluation runner module for evaluating all models on all guides.
"""

import json
import os
from typing import Dict, List

from src.data.loader import load_all_guides
from src.evaluation.metrics import graph_reconstruction_score


def evaluate_single_guide(model_name: str, guide: Dict, predictions: Dict) -> Dict:
    """
    Evaluate a single guide's predictions against gold annotations.

    Args:
        model_name: Name of the model being evaluated
        guide: Guide dictionary with annotations
        predictions: Dictionary with page-level predictions

    Returns:
        Dictionary with evaluation metrics
    """
    gold_annotations = guide["annotations"]

    # Aggregate predictions from all pages
    all_pred_entities = []
    all_pred_relations = []

    for page_num, page_pred in predictions.items():
        entities = page_pred.get("entities", [])
        relations = page_pred.get("relations", [])

        # Add page number to entities if not present
        for entity in entities:
            if "page" not in entity:
                entity["page"] = int(page_num)

        all_pred_entities.extend(entities)
        all_pred_relations.extend(relations)

    # Prepare gold entities and relations
    gold_entities = gold_annotations.get("entities", [])
    gold_relations = gold_annotations.get("relations", [])

    # Compute metrics
    pred_graph = {"entities": all_pred_entities, "relations": all_pred_relations}

    gold_graph = {"entities": gold_entities, "relations": gold_relations}

    metrics = graph_reconstruction_score(pred_graph, gold_graph)

    return {
        "model": model_name,
        "guide": guide["name"],
        **metrics,
        "num_pred_entities": len(all_pred_entities),
        "num_gold_entities": len(gold_entities),
        "num_pred_relations": len(all_pred_relations),
        "num_gold_relations": len(gold_relations),
    }


def evaluate_all_models(
    models: Dict[str, any], guides: List[Dict], output_dir: str = "output/evaluations"
):
    """
    Evaluate all models on all guides.

    Args:
        models: Dictionary mapping model names to model instances
        guides: List of guide dictionaries
        output_dir: Directory to save evaluation results

    Returns:
        Dictionary with all evaluation results
    """
    os.makedirs(output_dir, exist_ok=True)

    all_results = []

    for model_name, model in models.items():
        print(f"\nEvaluating {model_name}...")

        for guide in guides:
            print(f"  Processing guide: {guide['name']}")

            # Load predictions for this guide (should be saved by main pipeline)
            pred_file = os.path.join(
                "output/graphs", f"{guide['name']}_{model_name}.json"
            )

            if not os.path.exists(pred_file):
                print(
                    f"    Warning: No predictions found for {guide['name']} with {model_name}"
                )
                continue

            with open(pred_file, "r") as f:
                predictions = json.load(f)

            # Evaluate
            result = evaluate_single_guide(model_name, guide, predictions)
            all_results.append(result)

            print(
                f"    Entity F1: {result['entity_f1']:.3f}, Relation F1: {result['relation_f1']:.3f}"
            )

    # Save all results
    results_file = os.path.join(output_dir, "all_results.json")
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)

    return all_results


def print_summary_table(results: List[Dict]):
    """
    Print a summary table of evaluation results.

    Args:
        results: List of evaluation result dictionaries
    """
    if not results:
        print("No results to display.")
        return

    # Group by model
    model_results = {}
    for result in results:
        model_name = result["model"]
        if model_name not in model_results:
            model_results[model_name] = []
        model_results[model_name].append(result)

    # Compute averages per model
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(f"{'Model':<25} {'Entity F1':<12} {'Relation F1':<12} {'Overall F1':<12}")
    print("-" * 80)

    for model_name in sorted(model_results.keys()):
        model_res = model_results[model_name]

        avg_entity_f1 = sum(r["entity_f1"] for r in model_res) / len(model_res)
        avg_relation_f1 = sum(r["relation_f1"] for r in model_res) / len(model_res)
        avg_overall_f1 = sum(r["overall_f1"] for r in model_res) / len(model_res)

        print(
            f"{model_name:<25} {avg_entity_f1:<12.3f} {avg_relation_f1:<12.3f} {avg_overall_f1:<12.3f}"
        )

    print("=" * 80)

    # Per-guide breakdown
    print("\n" + "=" * 80)
    print("PER-GUIDE BREAKDOWN")
    print("=" * 80)

    guides = sorted(set(r["guide"] for r in results))
    models = sorted(set(r["model"] for r in results))

    print(f"{'Guide':<20}", end="")
    for model in models:
        print(f"{model:<15}", end="")
    print()
    print("-" * 80)

    for guide in guides:
        print(f"{guide:<20}", end="")
        for model in models:
            guide_results = [
                r for r in results if r["guide"] == guide and r["model"] == model
            ]
            if guide_results:
                f1 = guide_results[0]["overall_f1"]
                print(f"{f1:<15.3f}", end="")
            else:
                print(f"{'N/A':<15}", end="")
        print()

    print("=" * 80)
