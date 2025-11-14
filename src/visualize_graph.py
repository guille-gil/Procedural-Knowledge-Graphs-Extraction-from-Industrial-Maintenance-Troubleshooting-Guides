"""
Visualization module for procedural knowledge graphs.

Creates interactive HTML visualizations of the extracted ontology.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

try:
    from pyvis.network import Network
    PYVIS_AVAILABLE = True
except ImportError:
    PYVIS_AVAILABLE = False


def visualize_graph(
    graph_json: Dict[str, Any],
    output_path: Optional[str] = None,
    height: str = "800px",
    width: str = "100%",
    physics: bool = True,
) -> str:
    """
    Create an interactive HTML visualization of the knowledge graph.

    Args:
        graph_json: Graph dictionary with nodes and edges
        output_path: Path to save HTML file (default: data/processed/graph_visualization.html)
        height: Height of visualization canvas
        width: Width of visualization canvas
        physics: Enable physics simulation for layout

    Returns:
        Path to saved HTML file
    """
    if not NETWORKX_AVAILABLE:
        raise ImportError(
            "NetworkX not available. Install with: pip install networkx"
        )
    if not PYVIS_AVAILABLE:
        raise ImportError(
            "Pyvis not available. Install with: pip install pyvis"
        )

    # Create NetworkX graph
    G = nx.DiGraph()

    # Color mapping for node types
    node_colors = {
        "Condition": "#FF6B6B",      # Red
        "Action": "#4ECDC4",         # Teal
        "Observation": "#95E1D3",    # Light teal
        "Component": "#F38181",      # Light red
        "Connector": "#AA96DA",      # Purple
        "Result": "#FCBAD3",         # Pink
    }

    # Shape mapping for node types
    node_shapes = {
        "Condition": "diamond",
        "Action": "box",
        "Observation": "ellipse",
        "Component": "triangle",
        "Connector": "star",
        "Result": "dot",
    }

    # Add nodes with attributes
    for node in graph_json["nodes"]:
        node_id = node["id"]
        node_type = node.get("type", "Observation")
        label = node.get("label", node_id)
        
        # Truncate long labels for display
        display_label = label[:50] + "..." if len(label) > 50 else label
        
        G.add_node(
            node_id,
            label=display_label,
            title=f"{node_type}: {label}",
            color=node_colors.get(node_type, "#CCCCCC"),
            shape=node_shapes.get(node_type, "ellipse"),
            size=20 if node_type == "Component" else 15,
        )

    # Add edges with labels
    for edge in graph_json["edges"]:
        source = edge["source"]
        target = edge["target"]
        relation = edge.get("relation", "")
        confidence = edge.get("confidence", 0.5)
        
        # Only add edge if both nodes exist
        if source in G and target in G:
            G.add_edge(
                source,
                target,
                label=relation,
                title=f"{relation} (confidence: {confidence:.2f})",
                value=confidence,
            )

    # Create pyvis network
    net = Network(
        height=height,
        width=width,
        directed=True,
        notebook=False,
    )

    # Configure physics
    if physics:
        net.set_options("""
        {
          "physics": {
            "enabled": true,
            "stabilization": {"iterations": 200},
            "barnesHut": {
              "gravitationalConstant": -2000,
              "centralGravity": 0.1,
              "springLength": 200,
              "springConstant": 0.05
            }
          }
        }
        """)
    else:
        net.set_options("""
        {
          "physics": {
            "enabled": false
          }
        }
        """)

    # Add nodes and edges from NetworkX graph
    net.from_nx(G)

    # Determine output path
    if output_path is None:
        # Try to infer from graph metadata or use default
        output_dir = Path("data/processed")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(output_dir / "graph_visualization.html")
    else:
        output_path = str(output_path)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Save HTML
    net.save_graph(output_path)

    return output_path


def visualize_graph_from_file(
    json_path: str,
    output_path: Optional[str] = None,
    **kwargs
) -> str:
    """
    Load graph from JSON file and create visualization.

    Args:
        json_path: Path to JSON graph file
        output_path: Path to save HTML file
        **kwargs: Additional arguments for visualize_graph

    Returns:
        Path to saved HTML file
    """
    with open(json_path, "r", encoding="utf-8") as f:
        graph_json = json.load(f)

    # If output_path not specified, derive from input filename
    if output_path is None:
        input_path = Path(json_path)
        output_path = str(
            input_path.parent / f"{input_path.stem}_visualization.html"
        )

    return visualize_graph(graph_json, output_path, **kwargs)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python visualize_graph.py <graph_json_path> [output_html_path]")
        sys.exit(1)

    json_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    try:
        html_path = visualize_graph_from_file(json_path, output_path)
        print(f"Visualization saved to: {html_path}")
        print(f"Open in browser to view the interactive graph.")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

