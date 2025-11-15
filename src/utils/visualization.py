"""
Visualization utilities for drawing entities and relations on pages and visualizing graphs.
"""

import json
from typing import Dict, List, Optional

import networkx as nx
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def draw_entities_on_page(
    image: Image.Image, entities: List[Dict], color: str = "red"
) -> Image.Image:
    """
    Draw entity bounding boxes on a page image.

    Args:
        image: PIL Image of the page
        entities: List of entity dictionaries with 'bbox' and 'text' keys
        color: Color for bounding boxes (default: "red")

    Returns:
        PIL Image with bounding boxes drawn
    """
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)

    # Try to load a font, fallback to default if not available
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
    except:
        font = ImageFont.load_default()

    for entity in entities:
        bbox = entity.get("bbox")
        if bbox is None or len(bbox) != 4:
            continue

        x1, y1, x2, y2 = bbox
        entity_type = entity.get("type", "Unknown")
        text = entity.get("text", "")[:30]  # Truncate long text

        # Draw bounding box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

        # Draw label above box
        label = f"{entity_type}: {text}"
        # Get text size
        bbox_text = draw.textbbox((0, 0), label, font=font)
        text_width = bbox_text[2] - bbox_text[0]
        text_height = bbox_text[3] - bbox_text[1]

        # Draw background for text
        draw.rectangle(
            [x1, y1 - text_height - 4, x1 + text_width + 4, y1],
            fill="white",
            outline=color,
        )
        draw.text((x1 + 2, y1 - text_height - 2), label, fill=color, font=font)

    return img_copy


def draw_relations(
    image: Image.Image,
    entities: List[Dict],
    relations: List[Dict],
    entity_positions: Optional[Dict] = None,
) -> Image.Image:
    """
    Draw relations as arrows between entities on a page image.

    Args:
        image: PIL Image of the page
        entities: List of entity dictionaries
        relations: List of relation dictionaries with 'source_text' and 'target_text'
        entity_positions: Optional dict mapping entity text to center positions

    Returns:
        PIL Image with relations drawn as arrows
    """
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)

    # Build entity text to bbox mapping
    entity_map = {}
    for entity in entities:
        text = entity.get("text", "").strip()
        bbox = entity.get("bbox")
        if bbox and len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            entity_map[text] = (center_x, center_y)

    # Draw arrows for relations
    for relation in relations:
        source_text = relation.get("source_text", "").strip()
        target_text = relation.get("target_text", "").strip()
        rel_type = relation.get("type", "")

        if source_text in entity_map and target_text in entity_map:
            x1, y1 = entity_map[source_text]
            x2, y2 = entity_map[target_text]

            # Draw arrow
            draw.line([(x1, y1), (x2, y2)], fill="blue", width=2)

            # Draw arrowhead (simplified)
            # Calculate arrow direction
            dx = x2 - x1
            dy = y2 - y1
            length = (dx**2 + dy**2) ** 0.5
            if length > 0:
                # Normalize
                dx /= length
                dy /= length
                # Arrowhead size
                arrow_size = 10
                # Arrowhead points
                arrow_x = x2 - dx * arrow_size
                arrow_y = y2 - dy * arrow_size
                # Perpendicular for arrowhead
                perp_x = -dy
                perp_y = dx
                # Draw arrowhead triangle
                p1 = (x2, y2)
                p2 = (arrow_x + perp_x * 5, arrow_y + perp_y * 5)
                p3 = (arrow_x - perp_x * 5, arrow_y - perp_y * 5)
                draw.polygon([p1, p2, p3], fill="blue")

    return img_copy


def visualize_graph(graph_data: Dict, output_path: Optional[str] = None) -> nx.DiGraph:
    """
    Create a networkx graph visualization from graph data.

    Args:
        graph_data: Dictionary with 'entities' and 'relations' keys
        output_path: Optional path to save visualization

    Returns:
        NetworkX DiGraph object
    """
    G = nx.DiGraph()

    entities = graph_data.get("entities", [])
    relations = graph_data.get("relations", [])

    # Add nodes
    for entity in entities:
        entity_id = entity.get("id", entity.get("text", ""))
        entity_type = entity.get("type", "Unknown")
        entity_text = entity.get("text", "")[:50]  # Truncate for display

        G.add_node(
            entity_id,
            label=f"{entity_type}\n{entity_text}",
            type=entity_type,
            text=entity_text,
        )

    # Add edges
    for relation in relations:
        source_text = relation.get("source_text", "")
        target_text = relation.get("target_text", "")
        rel_type = relation.get("type", "")

        # Find entity IDs by text
        source_id = None
        target_id = None

        for entity in entities:
            if entity.get("text", "").strip() == source_text.strip():
                source_id = entity.get("id", entity.get("text", ""))
            if entity.get("text", "").strip() == target_text.strip():
                target_id = entity.get("id", entity.get("text", ""))

        if source_id and target_id:
            G.add_edge(source_id, target_id, label=rel_type, type=rel_type)

    # Visualize if output path provided
    if output_path:
        plt.figure(figsize=(16, 12))
        pos = nx.spring_layout(G, k=2, iterations=50)

        # Color nodes by type
        node_colors = []
        for node in G.nodes():
            node_type = G.nodes[node].get("type", "Unknown")
            color_map = {
                "Condition": "lightblue",
                "Action": "lightgreen",
                "Decision": "lightyellow",
                "Observation": "lightcoral",
                "Component": "lightpink",
            }
            node_colors.append(color_map.get(node_type, "lightgray"))

        nx.draw_networkx_nodes(
            G, pos, node_color=node_colors, node_size=2000, alpha=0.9
        )
        nx.draw_networkx_labels(G, pos, font_size=8)
        nx.draw_networkx_edges(G, pos, edge_color="gray", arrows=True, arrowsize=20)

        # Draw edge labels
        edge_labels = nx.get_edge_attributes(G, "label")
        nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=6)

        plt.title("Procedural Knowledge Graph")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

    return G
