"""
Graph visualization API routes.

Provides endpoints to fetch graph data for visualization.
"""

import io
import zipfile
from datetime import datetime
from pathlib import Path

import csv
import re

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from typing import Optional, Any, Dict, List

from src.neo4j_client import get_graphdb
from src.config import get_neo4j_import_dir

router = APIRouter(prefix="/api/graph", tags=["graph"])


def get_display_name(props: Dict[str, Any], fallback: str = "", label: str = "") -> str:
    """
    Extract display name from node properties.
    Tries multiple common property patterns based on node label.
    Priority: human-readable name > system ID
    """
    # Label-specific property priorities (human-readable first)
    label_priority = {
        "Respondent": ["respondent_id"],  # Show system ID for Respondent
        "Aspect": ["name"],  # Show human-readable name like "service_experience"
        "Brand": ["name", "brand_name"],  # Human-readable name first
        "Model": ["name", "model_name"],
        "Store": ["name", "store_name"],
        "Feature": ["name", "feature_name"],  # Text extraction entity
        "Component": ["name", "component_name"],
        "Experience": ["name", "experience_name"],
        "Issue": ["name", "issue_name"],
        "Quality": ["name", "quality_name"],
    }

    # If we have a label, try label-specific properties first
    if label and label in label_priority:
        for prop in label_priority[label]:
            if prop in props and props[prop]:
                return str(props[prop])

    # Generic name properties as fallback (human-readable first)
    name_props = [
        "name",  # Human-readable name first
        "respondent_id",
        "aspect_name",
        "brand_name",
        "model_name",
        "store_name",
        "brand_powertrain",
        "attribute_name",
        "entity_name",
        "Unnamed: 0",
    ]

    for prop in name_props:
        if prop in props and props[prop]:
            return str(props[prop])

    # Fall back to first non-empty value
    for v in props.values():
        if v:
            return str(v)[:30]

    return fallback[:8] if fallback else "Unknown"


def _format_cypher_value(value: Any) -> str:
    """Format a value for Cypher syntax."""
    if value is None:
        return "null"
    elif isinstance(value, bool):
        return "true" if value else "false"
    elif isinstance(value, (int, float)):
        return str(value)
    elif isinstance(value, str):
        # Escape quotes and backslashes
        escaped = value.replace("\\", "\\\\").replace("'", "\\'").replace("\n", "\\n")
        return f"'{escaped}'"
    elif isinstance(value, list):
        items = ", ".join(_format_cypher_value(v) for v in value)
        return f"[{items}]"
    else:
        escaped = str(value).replace("\\", "\\\\").replace("'", "\\'")
        return f"'{escaped}'"


def _format_cypher_props(props: Dict[str, Any]) -> str:
    """Format properties dict as Cypher property string."""
    if not props:
        return ""
    pairs = []
    for k, v in props.items():
        if v is not None:
            pairs.append(f"{k}: {_format_cypher_value(v)}")
    return ", ".join(pairs)


def _generate_create_cypher(graphdb) -> str:
    """
    Generate Cypher CREATE statements for all nodes and relationships.
    Returns a self-contained Cypher script that can recreate the entire graph.
    """
    statements = [
        "// ============================================",
        "// Neo4j Graph Export - CREATE Statements",
        "// Generated: " + datetime.now().isoformat(),
        "// ============================================",
        "",
        "// Clear existing data (uncomment if needed)",
        "// MATCH (n) DETACH DELETE n;",
        "",
    ]

    # Get all labels
    labels_result = graphdb.send_query("CALL db.labels()")
    if labels_result.get("status") != "success":
        return "// Error: Could not fetch labels"
    labels = [r["label"] for r in labels_result.get("query_result", [])]

    # Export nodes by label
    for label in labels:
        nodes_result = graphdb.send_query(f"MATCH (n:`{label}`) RETURN properties(n) as props")
        if nodes_result.get("status") == "success":
            nodes = nodes_result.get("query_result", [])
            if nodes:
                statements.append(f"// ----- {label} nodes ({len(nodes)}) -----")
                for node in nodes:
                    props = node.get("props", {})
                    props_str = _format_cypher_props(props)
                    statements.append(f"CREATE (:`{label}` {{{props_str}}});")
                statements.append("")

    # Export relationships
    rel_types_result = graphdb.send_query("CALL db.relationshipTypes()")
    if rel_types_result.get("status") == "success":
        rel_types = [r["relationshipType"] for r in rel_types_result.get("query_result", [])]

        for rel_type in rel_types:
            # Get relationships with their endpoint properties for matching
            rel_query = f"""
            MATCH (a)-[r:`{rel_type}`]->(b)
            RETURN labels(a) as a_labels, properties(a) as a_props,
                   properties(r) as r_props,
                   labels(b) as b_labels, properties(b) as b_props
            """
            rels_result = graphdb.send_query(rel_query)
            if rels_result.get("status") == "success":
                rels = rels_result.get("query_result", [])
                if rels:
                    statements.append(f"// ----- {rel_type} relationships ({len(rels)}) -----")
                    for rel in rels:
                        a_label = rel["a_labels"][0] if rel["a_labels"] else "Node"
                        b_label = rel["b_labels"][0] if rel["b_labels"] else "Node"
                        a_props = rel.get("a_props", {})
                        b_props = rel.get("b_props", {})
                        r_props = rel.get("r_props", {})

                        # Use first unique property for matching
                        a_key = list(a_props.keys())[0] if a_props else "id"
                        a_val = _format_cypher_value(a_props.get(a_key))
                        b_key = list(b_props.keys())[0] if b_props else "id"
                        b_val = _format_cypher_value(b_props.get(b_key))

                        r_props_str = f" {{{_format_cypher_props(r_props)}}}" if r_props else ""
                        statements.append(
                            f"MATCH (a:`{a_label}` {{{a_key}: {a_val}}}), "
                            f"(b:`{b_label}` {{{b_key}: {b_val}}}) "
                            f"CREATE (a)-[:`{rel_type}`{r_props_str}]->(b);"
                        )
                    statements.append("")

    return "\n".join(statements)


def _generate_load_csv_cypher(csv_files: List[Path]) -> str:
    """
    Generate Cypher LOAD CSV statements based on exported CSV files.
    Returns a Cypher script that can import data from the CSV files.
    """
    statements = [
        "// ============================================",
        "// Neo4j Graph Import - LOAD CSV Statements",
        "// Generated: " + datetime.now().isoformat(),
        "// ============================================",
        "",
        "// Instructions:",
        "// 1. Copy CSV files to Neo4j import directory",
        "// 2. Run these statements in Neo4j Browser",
        "",
        "// Clear existing data (uncomment if needed)",
        "// MATCH (n) DETACH DELETE n;",
        "",
    ]

    # Separate entity and relationship files
    entity_files = [f for f in csv_files if "_entities" in f.name.lower()]
    rel_files = [f for f in csv_files if "_relationships" in f.name.lower()]

    # Generate node import statements
    for csv_file in entity_files:
        # Extract label from filename: targeted_respondent_entities.csv -> Respondent
        match = re.search(r'targeted_(\w+)_entities', csv_file.name, re.IGNORECASE)
        if match:
            label = match.group(1).capitalize()

            # Read CSV headers to get property names
            try:
                with open(csv_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    headers = reader.fieldnames or []
            except Exception:
                headers = []

            if headers:
                # Build property mapping
                props_list = []
                for h in headers:
                    if h not in ['source_file', 'source_column']:  # Skip metadata
                        props_list.append(f"{h}: row.{h}")
                props_str = ", ".join(props_list)

                statements.append(f"// Load {label} nodes from {csv_file.name}")
                statements.append(f"LOAD CSV WITH HEADERS FROM 'file:///{csv_file.name}' AS row")
                statements.append(f"CREATE (:{label} {{{props_str}}});")
                statements.append("")

    # Generate relationship import statements
    for csv_file in rel_files:
        # Extract relationship type: targeted_rates_relationships.csv -> RATES
        match = re.search(r'targeted_(\w+)_relationships', csv_file.name, re.IGNORECASE)
        if match:
            rel_type = match.group(1).upper()

            # Read CSV to understand structure
            try:
                with open(csv_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    headers = reader.fieldnames or []
                    first_row = next(reader, None)
            except Exception:
                headers = []
                first_row = None

            if headers and first_row:
                # Determine from/to node types from relationship type
                from_label = "Respondent"  # Default
                to_label = "Node"

                if rel_type == "RATES":
                    to_label = "Aspect"
                elif rel_type == "EVALUATED_BRAND":
                    to_label = "Brand"
                elif rel_type == "EVALUATED_MODEL":
                    to_label = "Model"
                elif rel_type == "VISITED_STORE":
                    to_label = "Store"
                elif rel_type == "MENTIONED_FEATURE":
                    to_label = "Feature"
                elif rel_type == "BELONGS_TO":
                    from_label = "Model"
                    to_label = "Brand"

                # Build property mapping (exclude from_id, to_id, to_name)
                rel_props = []
                for h in headers:
                    if h not in ['from_id', 'to_id', 'to_name', 'relationship_type', 'source_column']:
                        if h == 'score':
                            rel_props.append(f"{h}: toFloat(row.{h})")
                        else:
                            rel_props.append(f"{h}: row.{h}")
                rel_props_str = ", ".join(rel_props) if rel_props else ""
                rel_props_clause = f" {{{rel_props_str}}}" if rel_props_str else ""

                statements.append(f"// Load {rel_type} relationships from {csv_file.name}")
                statements.append(f"LOAD CSV WITH HEADERS FROM 'file:///{csv_file.name}' AS row")
                statements.append(f"MATCH (from:{from_label} {{name: row.from_id}})")  # Assuming name matches from_id pattern
                statements.append(f"MATCH (to:{to_label} {{name: row.to_name}})")
                statements.append(f"CREATE (from)-[:{rel_type}{rel_props_clause}]->(to);")
                statements.append("")

    return "\n".join(statements)


@router.get("/schema")
async def get_graph_schema():
    """
    Get the graph schema (node labels and relationship types).
    """
    graphdb = get_graphdb()

    # Get node labels
    labels_result = graphdb.send_query("CALL db.labels()")
    labels = []
    if labels_result.get("status") == "success":
        labels = [r["label"] for r in labels_result.get("query_result", [])]

    # Get relationship types
    rel_types_result = graphdb.send_query("CALL db.relationshipTypes()")
    rel_types = []
    if rel_types_result.get("status") == "success":
        rel_types = [r["relationshipType"] for r in rel_types_result.get("query_result", [])]

    # Get node counts per label
    node_counts = {}
    for label in labels:
        count_result = graphdb.send_query(f"MATCH (n:`{label}`) RETURN count(n) as count")
        if count_result.get("status") == "success" and count_result.get("query_result"):
            node_counts[label] = count_result["query_result"][0]["count"]

    # Get relationship counts per type
    rel_counts = {}
    for rel_type in rel_types:
        count_result = graphdb.send_query(f"MATCH ()-[r:`{rel_type}`]->() RETURN count(r) as count")
        if count_result.get("status") == "success" and count_result.get("query_result"):
            rel_counts[rel_type] = count_result["query_result"][0]["count"]

    return {
        "labels": labels,
        "relationship_types": rel_types,
        "node_counts": node_counts,
        "relationship_counts": rel_counts
    }


@router.get("/nodes")
async def get_graph_nodes(
    limit: int = Query(default=100, le=500),
    label: Optional[str] = None
):
    """
    Get nodes for visualization.
    """
    graphdb = get_graphdb()

    if label:
        query = f"MATCH (n:`{label}`) RETURN n LIMIT {limit}"
    else:
        query = f"MATCH (n) RETURN n LIMIT {limit}"

    result = graphdb.send_query(query)

    if result.get("status") == "error":
        raise HTTPException(status_code=500, detail=result.get("error_message"))

    nodes = []
    for record in result.get("query_result", []):
        node = record.get("n", {})
        nodes.append({
            "id": node.get("element_id", str(id(node))),
            "labels": node.get("labels", []),
            "properties": {k: v for k, v in node.items() if k not in ["element_id", "labels"]}
        })

    return {"nodes": nodes, "count": len(nodes)}


@router.get("/sample")
async def get_graph_sample(limit: int = Query(default=5000, le=10000)):
    """
    Get a sample of the graph including nodes and relationships.
    Returns data formatted for force-graph visualization.
    Traverses from Respondent nodes to get complete connected graph.
    """
    graphdb = get_graphdb()

    # Strategy: Get ALL relationships in the graph
    # This ensures we capture all nodes including those not connected to Respondent path
    query = f"""
    // Get ALL relationships in the graph (no path restriction)
    MATCH (n)-[r]->(m)
    RETURN
        elementId(n) as n_id,
        labels(n) as n_labels,
        properties(n) as n_props,
        type(r) as r_type,
        properties(r) as r_props,
        elementId(m) as m_id,
        labels(m) as m_labels,
        properties(m) as m_props
    LIMIT {limit}
    """

    result = graphdb.send_query(query)

    if result.get("status") == "error":
        raise HTTPException(status_code=500, detail=result.get("error_message"))

    nodes_map = {}
    links = []

    for record in result.get("query_result", []):
        # Process source node
        n_id = record.get("n_id", str(hash(str(record))))
        n_labels = record.get("n_labels", ["Node"])
        n_props = record.get("n_props", {})
        n_label = n_labels[0] if n_labels else "Node"

        if n_id not in nodes_map:
            nodes_map[n_id] = {
                "id": n_id,
                "label": n_label,
                "name": get_display_name(n_props, n_id, n_label),
                "properties": n_props
            }

        # Process target node
        m_id = record.get("m_id", str(hash(str(record))))
        m_labels = record.get("m_labels", ["Node"])
        m_props = record.get("m_props", {})
        m_label = m_labels[0] if m_labels else "Node"

        if m_id not in nodes_map:
            nodes_map[m_id] = {
                "id": m_id,
                "label": m_label,
                "name": get_display_name(m_props, m_id, m_label),
                "properties": m_props
            }

        # Process relationship
        r_type = record.get("r_type", "RELATED")
        r_props = record.get("r_props", {})

        links.append({
            "source": n_id,
            "target": m_id,
            "type": r_type,
            "properties": r_props
        })

    return {
        "nodes": list(nodes_map.values()),
        "links": links,
        "node_count": len(nodes_map),
        "link_count": len(links)
    }


@router.get("/stats")
async def get_graph_stats():
    """
    Get overall graph statistics.
    """
    graphdb = get_graphdb()

    # Total nodes
    node_result = graphdb.send_query("MATCH (n) RETURN count(n) as count")
    node_count = 0
    if node_result.get("status") == "success" and node_result.get("query_result"):
        node_count = node_result["query_result"][0]["count"]

    # Total relationships
    rel_result = graphdb.send_query("MATCH ()-[r]->() RETURN count(r) as count")
    rel_count = 0
    if rel_result.get("status") == "success" and rel_result.get("query_result"):
        rel_count = rel_result["query_result"][0]["count"]

    return {
        "total_nodes": node_count,
        "total_relationships": rel_count
    }


@router.get("/brand-powertrains")
async def get_brand_powertrains():
    """
    Get all BrandPowertrain nodes for dropdown selection.
    """
    graphdb = get_graphdb()

    query = """
    MATCH (bp:BrandPowertrain)
    RETURN
        elementId(bp) as id,
        bp.brand_powertrain as name,
        bp.primary_name as primary_name,
        bp.secondary_name as secondary_name
    ORDER BY bp.brand_powertrain
    """

    result = graphdb.send_query(query)

    if result.get("status") == "error":
        raise HTTPException(status_code=500, detail=result.get("error_message"))

    brand_powertrains = []
    for record in result.get("query_result", []):
        brand_powertrains.append({
            "id": record.get("id"),
            "name": record.get("name"),
            "primary_name": record.get("primary_name"),
            "secondary_name": record.get("secondary_name")
        })

    return {"brand_powertrains": brand_powertrains, "count": len(brand_powertrains)}


@router.get("/filter-options/{label}")
async def get_filter_options(label: str):
    """
    Get all nodes of a specific label for filter dropdown.
    Returns node id and display name for each node.
    """
    graphdb = get_graphdb()

    # Get nodes with their properties to determine display name
    query = f"""
    MATCH (n:`{label}`)
    RETURN
        elementId(n) as id,
        properties(n) as props
    ORDER BY properties(n)
    LIMIT 2000
    """

    result = graphdb.send_query(query)

    if result.get("status") == "error":
        raise HTTPException(status_code=500, detail=result.get("error_message"))

    options = []
    for record in result.get("query_result", []):
        props = record.get("props", {})
        options.append({
            "id": record.get("id"),
            "name": get_display_name(props, record.get("id", ""), label),
            "properties": props
        })

    return {"label": label, "options": options, "count": len(options)}


@router.get("/by-center-node/{node_id:path}")
async def get_graph_by_center_node(node_id: str, limit: int = Query(default=500, le=1000)):
    """
    Get graph data centered on a specific node and its connected nodes.
    Works with any node type.
    """
    graphdb = get_graphdb()

    # Get all relationships where the node is either source or target
    query = f"""
    MATCH (center)
    WHERE elementId(center) = $node_id
    OPTIONAL MATCH (center)-[r1]->(out)
    OPTIONAL MATCH (in)-[r2]->(center)
    WITH center,
         collect(DISTINCT {{node: out, rel: r1, dir: 'out'}}) as outgoing,
         collect(DISTINCT {{node: in, rel: r2, dir: 'in'}}) as incoming
    RETURN
        elementId(center) as center_id,
        labels(center) as center_labels,
        properties(center) as center_props,
        [x in outgoing WHERE x.node IS NOT NULL | {{
            n_id: elementId(x.node),
            n_labels: labels(x.node),
            n_props: properties(x.node),
            r_type: type(x.rel),
            r_props: properties(x.rel),
            direction: 'out'
        }}][0..{limit}] as outgoing_rels,
        [x in incoming WHERE x.node IS NOT NULL | {{
            n_id: elementId(x.node),
            n_labels: labels(x.node),
            n_props: properties(x.node),
            r_type: type(x.rel),
            r_props: properties(x.rel),
            direction: 'in'
        }}][0..{limit}] as incoming_rels
    """

    result = graphdb.send_query(query, {"node_id": node_id})

    if result.get("status") == "error":
        raise HTTPException(status_code=500, detail=result.get("error_message"))

    nodes_map = {}
    links = []

    for record in result.get("query_result", []):
        # Add center node
        center_id = record.get("center_id")
        center_labels = record.get("center_labels", ["Node"])
        center_props = record.get("center_props", {})
        center_label = center_labels[0] if center_labels else "Node"

        nodes_map[center_id] = {
            "id": center_id,
            "label": center_label,
            "name": get_display_name(center_props, center_id),
            "properties": center_props,
            "isCenter": True
        }

        # Process outgoing relationships
        for rel in record.get("outgoing_rels", []):
            if rel:
                n_id = rel.get("n_id")
                n_labels = rel.get("n_labels", ["Node"])
                n_props = rel.get("n_props", {})
                n_label = n_labels[0] if n_labels else "Node"

                if n_id not in nodes_map:
                    nodes_map[n_id] = {
                        "id": n_id,
                        "label": n_label,
                        "name": get_display_name(n_props, n_id),
                        "properties": n_props
                    }

                links.append({
                    "source": center_id,
                    "target": n_id,
                    "type": rel.get("r_type", "RELATED"),
                    "properties": rel.get("r_props", {})
                })

        # Process incoming relationships
        for rel in record.get("incoming_rels", []):
            if rel:
                n_id = rel.get("n_id")
                n_labels = rel.get("n_labels", ["Node"])
                n_props = rel.get("n_props", {})
                n_label = n_labels[0] if n_labels else "Node"

                if n_id not in nodes_map:
                    nodes_map[n_id] = {
                        "id": n_id,
                        "label": n_label,
                        "name": get_display_name(n_props, n_id),
                        "properties": n_props
                    }

                links.append({
                    "source": n_id,
                    "target": center_id,
                    "type": rel.get("r_type", "RELATED"),
                    "properties": rel.get("r_props", {})
                })

    return {
        "nodes": list(nodes_map.values()),
        "links": links,
        "node_count": len(nodes_map),
        "link_count": len(links)
    }


@router.get("/by-brand-powertrain/{bp_id:path}")
async def get_graph_by_brand_powertrain(bp_id: str, limit: int = Query(default=300, le=500)):
    """
    Get graph data for a specific BrandPowertrain node and its connected attributes.
    """
    graphdb = get_graphdb()

    # Get the BrandPowertrain node and all connected VehicleAttribute nodes
    query = f"""
    MATCH (attr)-[r]->(bp:BrandPowertrain)
    WHERE elementId(bp) = $bp_id
    RETURN
        elementId(attr) as n_id,
        labels(attr) as n_labels,
        properties(attr) as n_props,
        type(r) as r_type,
        properties(r) as r_props,
        elementId(bp) as m_id,
        labels(bp) as m_labels,
        properties(bp) as m_props
    LIMIT {limit}
    """

    result = graphdb.send_query(query, {"bp_id": bp_id})

    if result.get("status") == "error":
        raise HTTPException(status_code=500, detail=result.get("error_message"))

    nodes_map = {}
    links = []

    for record in result.get("query_result", []):
        # Process attribute node (source)
        n_id = record.get("n_id", str(hash(str(record))))
        n_labels = record.get("n_labels", ["Node"])
        n_props = record.get("n_props", {})
        n_label = n_labels[0] if n_labels else "Node"

        if n_id not in nodes_map:
            nodes_map[n_id] = {
                "id": n_id,
                "label": n_label,
                "name": get_display_name(n_props, n_id),
                "properties": n_props
            }

        # Process BrandPowertrain node (target)
        m_id = record.get("m_id", str(hash(str(record))))
        m_labels = record.get("m_labels", ["Node"])
        m_props = record.get("m_props", {})
        m_label = m_labels[0] if m_labels else "Node"

        if m_id not in nodes_map:
            nodes_map[m_id] = {
                "id": m_id,
                "label": m_label,
                "name": get_display_name(m_props, m_id),
                "properties": m_props
            }

        # Process relationship
        r_type = record.get("r_type", "RELATED")
        r_props = record.get("r_props", {})

        links.append({
            "source": n_id,
            "target": m_id,
            "type": r_type,
            "properties": r_props
        })

    return {
        "nodes": list(nodes_map.values()),
        "links": links,
        "node_count": len(nodes_map),
        "link_count": len(links)
    }


@router.delete("/clear")
async def clear_graph_database():
    """
    Clear all nodes and relationships from the database.
    Use this before importing a new graph to ensure fresh data.
    """
    graphdb = get_graphdb()

    # Delete all relationships first, then all nodes
    rel_result = graphdb.send_query("MATCH ()-[r]->() DELETE r")
    if rel_result.get("status") == "error":
        raise HTTPException(status_code=500, detail=f"Failed to delete relationships: {rel_result.get('error_message')}")

    node_result = graphdb.send_query("MATCH (n) DELETE n")
    if node_result.get("status") == "error":
        raise HTTPException(status_code=500, detail=f"Failed to delete nodes: {node_result.get('error_message')}")

    return {
        "status": "success",
        "message": "Database cleared successfully"
    }


@router.get("/export")
async def export_graph_data():
    """
    Export all extracted graph data as a ZIP file.
    Returns a ZIP containing:
    - All CSV files from data/extracted_data/
    - import_create.cypher - Self-contained CREATE statements
    - import_load_csv.cypher - LOAD CSV commands for fast import
    """
    # Get data directory
    import_dir = get_neo4j_import_dir()
    if not import_dir:
        raise HTTPException(status_code=500, detail="NEO4J_IMPORT_DIR not configured")

    extracted_dir = Path(import_dir) / "extracted_data"
    if not extracted_dir.exists():
        raise HTTPException(status_code=404, detail="No extracted data found. Run the pipeline first.")

    # Get all CSV files
    csv_files = list(extracted_dir.glob("*.csv"))
    if not csv_files:
        raise HTTPException(status_code=404, detail="No CSV files found in extracted_data")

    # Get graph database connection for Cypher export
    graphdb = get_graphdb()

    # Create ZIP in memory
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        # Add CSV files
        for csv_file in csv_files:
            zf.write(csv_file, csv_file.name)

        # Generate and add CREATE Cypher file
        create_cypher = _generate_create_cypher(graphdb)
        zf.writestr("import_create.cypher", create_cypher)

        # Generate and add LOAD CSV Cypher file
        load_csv_cypher = _generate_load_csv_cypher(csv_files)
        zf.writestr("import_load_csv.cypher", load_csv_cypher)

    zip_buffer.seek(0)

    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"graph_data_{timestamp}.zip"

    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


@router.get("/export/files")
async def list_exportable_files():
    """
    List all files available for export.
    """
    import_dir = get_neo4j_import_dir()
    if not import_dir:
        return {"files": [], "count": 0}

    extracted_dir = Path(import_dir) / "extracted_data"
    if not extracted_dir.exists():
        return {"files": [], "count": 0}

    files = []
    for csv_file in sorted(extracted_dir.glob("*.csv")):
        stat = csv_file.stat()
        files.append({
            "name": csv_file.name,
            "size": stat.st_size,
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
        })

    return {"files": files, "count": len(files)}
