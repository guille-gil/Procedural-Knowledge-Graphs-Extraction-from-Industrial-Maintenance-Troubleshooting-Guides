"""
Label definitions for procedural knowledge extraction.
Loads entity types and relation types from YAML configuration.
"""

import yaml
from pathlib import Path


def _load_labels():
    """Load labels from YAML file."""
    # Get project root (parent of src)
    project_root = Path(__file__).parent.parent.parent
    labels_file = project_root / "schemas" / "labels.yaml"

    with open(labels_file, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return data


# Load labels on import
_labels_data = _load_labels()

ENTITY_TYPES = _labels_data["entity_types"]
RELATION_TYPES = _labels_data["relation_types"]


def is_valid_entity_type(entity_type: str) -> bool:
    """Check if an entity type is valid according to the label set."""
    return entity_type in ENTITY_TYPES


def is_valid_relation_type(relation_type: str) -> bool:
    """Check if a relation type is valid according to the label set."""
    return relation_type in RELATION_TYPES


def get_entity_type_description(entity_type: str) -> str:
    """Get a description of what an entity type represents."""
    descriptions = _labels_data.get("entity_descriptions", {})
    return descriptions.get(entity_type, "Unknown entity type")


def get_relation_type_description(relation_type: str) -> str:
    """Get a description of what a relation type represents."""
    descriptions = _labels_data.get("relation_descriptions", {})
    return descriptions.get(relation_type, "Unknown relation type")
