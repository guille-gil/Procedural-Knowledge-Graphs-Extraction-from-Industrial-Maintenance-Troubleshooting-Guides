"""
Evaluation metrics module for comparing model outputs with gold annotations.
Computes precision, recall, F1 for nodes and relations, and graph reconstruction accuracy.
"""

from typing import Dict, List, Tuple
from difflib import SequenceMatcher


def text_similarity(text1: str, text2: str) -> float:
    """Compute string similarity between two texts (0-1 scale)."""
    return SequenceMatcher(None, text1.lower().strip(), text2.lower().strip()).ratio()


def match_entities(
    pred_entity: Dict, gold_entities: List[Dict], threshold: float = 0.7
) -> Tuple[bool, Optional[Dict]]:
    """
    Match a predicted entity to a gold entity.

    Args:
        pred_entity: Predicted entity dict with 'text' and 'type'
        gold_entities: List of gold entity dicts
        threshold: Text similarity threshold for matching

    Returns:
        Tuple of (matched: bool, matched_gold_entity: Dict or None)
    """
    pred_text = pred_entity.get("text", "").strip()
    pred_type = pred_entity.get("type", "")

    best_match = None
    best_similarity = 0.0

    for gold_entity in gold_entities:
        gold_text = gold_entity.get("text", "").strip()
        gold_type = gold_entity.get("type", "")

        # Type must match
        if pred_type != gold_type:
            continue

        # Text similarity must exceed threshold
        similarity = text_similarity(pred_text, gold_text)
        if similarity >= threshold and similarity > best_similarity:
            best_similarity = similarity
            best_match = gold_entity

    if best_match is not None:
        return True, best_match
    return False, None


def create_entity_mapping(
    pred_entities: List[Dict], gold_entities: List[Dict], threshold: float = 0.7
) -> Dict[str, str]:
    """
    Create a mapping from predicted entity IDs to gold entity IDs.

    Args:
        pred_entities: List of predicted entities
        gold_entities: List of gold entities
        threshold: Text similarity threshold

    Returns:
        Dictionary mapping pred_entity_id -> gold_entity_id
    """
    mapping = {}
    used_gold_ids = set()

    # Sort by confidence if available, otherwise by order
    sorted_pred = sorted(
        pred_entities, key=lambda x: x.get("confidence", 1.0), reverse=True
    )

    for pred_entity in sorted_pred:
        pred_id = pred_entity.get("id", f"pred_{len(mapping)}")

        # Try to find best match
        for gold_entity in gold_entities:
            gold_id = gold_entity.get("id", "")
            if gold_id in used_gold_ids:
                continue

            matched, matched_entity = match_entities(
                pred_entity, [gold_entity], threshold
            )
            if matched and matched_entity == gold_entity:
                mapping[pred_id] = gold_id
                used_gold_ids.add(gold_id)
                break

    return mapping


def entity_precision_recall_f1(
    pred_entities: List[Dict], gold_entities: List[Dict], threshold: float = 0.7
) -> Dict[str, float]:
    """
    Compute precision, recall, and F1 for entity extraction.

    Args:
        pred_entities: List of predicted entities
        gold_entities: List of gold standard entities
        threshold: Text similarity threshold for matching

    Returns:
        Dictionary with 'precision', 'recall', 'f1' keys
    """
    if len(pred_entities) == 0 and len(gold_entities) == 0:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}

    if len(pred_entities) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    if len(gold_entities) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    # Match predicted entities to gold entities
    matched_count = 0
    used_gold_indices = set()

    for pred_entity in pred_entities:
        matched, matched_entity = match_entities(pred_entity, gold_entities, threshold)
        if matched:
            # Find index of matched entity
            for i, gold_entity in enumerate(gold_entities):
                if gold_entity == matched_entity and i not in used_gold_indices:
                    matched_count += 1
                    used_gold_indices.add(i)
                    break

    precision = matched_count / len(pred_entities) if len(pred_entities) > 0 else 0.0
    recall = matched_count / len(gold_entities) if len(gold_entities) > 0 else 0.0

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "matched": matched_count,
        "total_pred": len(pred_entities),
        "total_gold": len(gold_entities),
    }


def relation_precision_recall_f1(
    pred_relations: List[Dict],
    gold_relations: List[Dict],
    pred_entities: List[Dict],
    gold_entities: List[Dict],
    threshold: float = 0.7,
) -> Dict[str, float]:
    """
    Compute precision, recall, and F1 for relation extraction.

    Args:
        pred_relations: List of predicted relations
        gold_relations: List of gold standard relations
        pred_entities: List of predicted entities (for text-to-ID mapping)
        gold_entities: List of gold entities (for text-to-ID mapping)
        threshold: Text similarity threshold for entity matching

    Returns:
        Dictionary with 'precision', 'recall', 'f1' keys
    """
    if len(pred_relations) == 0 and len(gold_relations) == 0:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}

    if len(pred_relations) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    if len(gold_relations) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    # Build mapping from pred entity text to pred entity ID
    pred_text_to_id = {}
    for entity in pred_entities:
        entity_id = entity.get("id", entity.get("text", ""))
        entity_text = entity.get("text", "").strip()
        if entity_text:
            pred_text_to_id[entity_text.lower()] = entity_id

    # Build mapping from gold entity text to gold entity ID
    gold_text_to_id = {}
    for entity in gold_entities:
        entity_id = entity.get("id", "")
        entity_text = entity.get("text", "").strip()
        if entity_text and entity_id:
            gold_text_to_id[entity_text.lower()] = entity_id

    # Build entity alignment: pred entity ID -> gold entity ID
    entity_alignment = {}
    for pred_entity in pred_entities:
        pred_id = pred_entity.get("id", pred_entity.get("text", ""))
        matched, matched_gold = match_entities(pred_entity, gold_entities, threshold)
        if matched:
            gold_id = matched_gold.get("id", "")
            if gold_id:
                entity_alignment[pred_id] = gold_id

    # Match relations
    matched_count = 0
    used_gold_indices = set()

    for pred_rel in pred_relations:
        pred_source_text = pred_rel.get("source_text", "").strip()
        pred_target_text = pred_rel.get("target_text", "").strip()
        pred_type = pred_rel.get("type", "")

        # Map pred entity texts to IDs, then to gold IDs
        pred_source_id = pred_text_to_id.get(pred_source_text.lower(), "")
        pred_target_id = pred_text_to_id.get(pred_target_text.lower(), "")

        gold_source_id = entity_alignment.get(pred_source_id, "")
        gold_target_id = entity_alignment.get(pred_target_id, "")

        # Find matching gold relation
        for i, gold_rel in enumerate(gold_relations):
            if i in used_gold_indices:
                continue

            gold_source = gold_rel.get("source", "")
            gold_target = gold_rel.get("target", "")
            gold_type = gold_rel.get("type", "")

            # Type must match
            if pred_type != gold_type:
                continue

            # Both source and target must match
            if gold_source_id and gold_target_id:
                if gold_source == gold_source_id and gold_target == gold_target_id:
                    matched_count += 1
                    used_gold_indices.add(i)
                    break
            else:
                # Fallback: match by text similarity if IDs not available
                gold_source_text = ""
                gold_target_text = ""

                # Find gold entity texts by ID
                for gold_entity in gold_entities:
                    if gold_entity.get("id", "") == gold_source:
                        gold_source_text = gold_entity.get("text", "").strip().lower()
                    if gold_entity.get("id", "") == gold_target:
                        gold_target_text = gold_entity.get("text", "").strip().lower()

                if (
                    text_similarity(pred_source_text.lower(), gold_source_text)
                    >= threshold
                    and text_similarity(pred_target_text.lower(), gold_target_text)
                    >= threshold
                ):
                    matched_count += 1
                    used_gold_indices.add(i)
                    break

    precision = matched_count / len(pred_relations) if len(pred_relations) > 0 else 0.0
    recall = matched_count / len(gold_relations) if len(gold_relations) > 0 else 0.0

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "matched": matched_count,
        "total_pred": len(pred_relations),
        "total_gold": len(gold_relations),
    }


def graph_reconstruction_score(
    pred_graph: Dict, gold_graph: Dict, threshold: float = 0.7
) -> Dict[str, float]:
    """
    Compute graph reconstruction accuracy.

    Args:
        pred_graph: Predicted graph with 'entities' and 'relations'
        gold_graph: Gold graph with 'entities' and 'relations'
        threshold: Text similarity threshold

    Returns:
        Dictionary with various graph-level metrics
    """
    pred_entities = pred_graph.get("entities", [])
    gold_entities = gold_graph.get("entities", [])
    pred_relations = pred_graph.get("relations", [])
    gold_relations = gold_graph.get("relations", [])

    # Compute entity metrics
    entity_metrics = entity_precision_recall_f1(pred_entities, gold_entities, threshold)

    # Compute relation metrics (now takes entities directly)
    relation_metrics = relation_precision_recall_f1(
        pred_relations, gold_relations, pred_entities, gold_entities, threshold
    )

    # Overall graph score (average of entity and relation F1)
    overall_f1 = (entity_metrics["f1"] + relation_metrics["f1"]) / 2.0

    return {
        "entity_precision": entity_metrics["precision"],
        "entity_recall": entity_metrics["recall"],
        "entity_f1": entity_metrics["f1"],
        "relation_precision": relation_metrics["precision"],
        "relation_recall": relation_metrics["recall"],
        "relation_f1": relation_metrics["f1"],
        "overall_f1": overall_f1,
    }
