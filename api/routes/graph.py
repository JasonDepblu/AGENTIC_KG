"""
Graph visualization API routes.

Provides endpoints to fetch graph data for visualization.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional

from src.neo4j_client import get_graphdb

router = APIRouter(prefix="/api/graph", tags=["graph"])


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
async def get_graph_sample(limit: int = Query(default=50, le=200)):
    """
    Get a sample of the graph including nodes and relationships.
    Returns data formatted for force-graph visualization.
    """
    graphdb = get_graphdb()

    # Get nodes and relationships with full metadata
    query = f"""
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

        # Get display name from properties
        n_name = (
            n_props.get("brand_powertrain") or
            n_props.get("attribute_name") or
            n_props.get("name") or
            n_props.get("Unnamed: 0") or
            (list(n_props.values())[0] if n_props else n_id[:8])
        )

        if n_id not in nodes_map:
            nodes_map[n_id] = {
                "id": n_id,
                "label": n_label,
                "name": str(n_name)[:30],
                "properties": n_props
            }

        # Process target node
        m_id = record.get("m_id", str(hash(str(record))))
        m_labels = record.get("m_labels", ["Node"])
        m_props = record.get("m_props", {})
        m_label = m_labels[0] if m_labels else "Node"

        m_name = (
            m_props.get("brand_powertrain") or
            m_props.get("attribute_name") or
            m_props.get("name") or
            m_props.get("Unnamed: 0") or
            (list(m_props.values())[0] if m_props else m_id[:8])
        )

        if m_id not in nodes_map:
            nodes_map[m_id] = {
                "id": m_id,
                "label": m_label,
                "name": str(m_name)[:30],
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
    LIMIT 500
    """

    result = graphdb.send_query(query)

    if result.get("status") == "error":
        raise HTTPException(status_code=500, detail=result.get("error_message"))

    options = []
    for record in result.get("query_result", []):
        props = record.get("props", {})
        # Try to get a display name from common property names
        display_name = (
            props.get("brand_powertrain") or
            props.get("attribute_name") or
            props.get("name") or
            props.get("Unnamed: 0") or
            (list(props.values())[0] if props else record.get("id", "")[:8])
        )
        options.append({
            "id": record.get("id"),
            "name": str(display_name),
            "properties": props
        })

    return {"label": label, "options": options, "count": len(options)}


@router.get("/by-center-node/{node_id:path}")
async def get_graph_by_center_node(node_id: str, limit: int = Query(default=300, le=500)):
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

        center_name = (
            center_props.get("brand_powertrain") or
            center_props.get("attribute_name") or
            center_props.get("name") or
            center_props.get("Unnamed: 0") or
            (list(center_props.values())[0] if center_props else center_id[:8])
        )

        nodes_map[center_id] = {
            "id": center_id,
            "label": center_label,
            "name": str(center_name)[:30],
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

                n_name = (
                    n_props.get("brand_powertrain") or
                    n_props.get("attribute_name") or
                    n_props.get("name") or
                    n_props.get("Unnamed: 0") or
                    (list(n_props.values())[0] if n_props else n_id[:8])
                )

                if n_id not in nodes_map:
                    nodes_map[n_id] = {
                        "id": n_id,
                        "label": n_label,
                        "name": str(n_name)[:30],
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

                n_name = (
                    n_props.get("brand_powertrain") or
                    n_props.get("attribute_name") or
                    n_props.get("name") or
                    n_props.get("Unnamed: 0") or
                    (list(n_props.values())[0] if n_props else n_id[:8])
                )

                if n_id not in nodes_map:
                    nodes_map[n_id] = {
                        "id": n_id,
                        "label": n_label,
                        "name": str(n_name)[:30],
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

        n_name = (
            n_props.get("attribute_name") or
            n_props.get("name") or
            n_props.get("Unnamed: 0") or
            (list(n_props.values())[0] if n_props else n_id[:8])
        )

        if n_id not in nodes_map:
            nodes_map[n_id] = {
                "id": n_id,
                "label": n_label,
                "name": str(n_name)[:30],
                "properties": n_props
            }

        # Process BrandPowertrain node (target)
        m_id = record.get("m_id", str(hash(str(record))))
        m_labels = record.get("m_labels", ["Node"])
        m_props = record.get("m_props", {})
        m_label = m_labels[0] if m_labels else "Node"

        m_name = (
            m_props.get("brand_powertrain") or
            m_props.get("name") or
            (list(m_props.values())[0] if m_props else m_id[:8])
        )

        if m_id not in nodes_map:
            nodes_map[m_id] = {
                "id": m_id,
                "label": m_label,
                "name": str(m_name)[:30],
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
