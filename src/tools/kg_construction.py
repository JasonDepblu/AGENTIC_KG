"""
Knowledge graph construction tools for Agentic KG.

Tools for building and managing knowledge graph schemas and data import.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from google.adk.tools import ToolContext

from .common import tool_success, tool_error
from .file_suggestion import search_file, SEARCH_RESULTS
from ..config import get_neo4j_import_dir


def _to_relative_path(file_path: str) -> str:
    """
    Convert an absolute file path to a relative path for Neo4j LOAD CSV.

    Neo4j in Docker can only access files relative to its import directory.
    This function strips the NEO4J_IMPORT_DIR prefix to get the relative path.

    Args:
        file_path: Absolute or relative file path

    Returns:
        Relative path suitable for Neo4j LOAD CSV
    """
    import_dir = get_neo4j_import_dir()
    if not import_dir:
        # No import dir configured, return as-is
        return file_path

    file_path_obj = Path(file_path)
    import_dir_obj = Path(import_dir)

    # Check if it's an absolute path under the import directory
    try:
        relative = file_path_obj.relative_to(import_dir_obj)
        return str(relative)
    except ValueError:
        # Not under import_dir, return original
        return file_path

# State keys
PROPOSED_CONSTRUCTION_PLAN = "proposed_construction_plan"
APPROVED_CONSTRUCTION_PLAN = "approved_construction_plan"
NODE_CONSTRUCTION = "node_construction"
RELATIONSHIP_CONSTRUCTION = "relationship_construction"


def create_uniqueness_constraint(
    label: str,
    unique_property_key: str,
    graphdb=None
) -> Dict[str, Any]:
    """
    Create a uniqueness constraint for a node label and property key.

    A uniqueness constraint ensures that no two nodes with the same label
    and property key have the same value. This improves performance and
    data integrity during import and queries.

    Args:
        label: The label of the node to create a constraint for
        unique_property_key: The property key that should have a unique value
        graphdb: Neo4j client instance (optional, uses default if not provided)

    Returns:
        Dictionary with status indicating success or error
    """
    if graphdb is None:
        from ..neo4j_client import get_graphdb
        graphdb = get_graphdb()

    constraint_name = f"{label}_{unique_property_key}_constraint"
    query = f"""CREATE CONSTRAINT `{constraint_name}` IF NOT EXISTS
    FOR (n:`{label}`)
    REQUIRE n.`{unique_property_key}` IS UNIQUE"""

    return graphdb.send_query(query)


def load_nodes_from_csv(
    source_file: str,
    label: str,
    unique_column_name: str,
    properties: List[str],
    graphdb=None,
    import_all_columns: bool = True
) -> Dict[str, Any]:
    """
    Batch load nodes from a CSV file into Neo4j.

    Uses LOAD CSV with MERGE to create nodes while avoiding duplicates
    based on the unique column.

    Args:
        source_file: CSV file name in the import directory
        label: Node label to assign
        unique_column_name: Column containing unique identifier
        properties: List of column names to import as properties
        graphdb: Neo4j client instance (optional)
        import_all_columns: If True, import all columns from CSV (default True)

    Returns:
        Dictionary with status indicating success or error
    """
    import pandas as pd

    if graphdb is None:
        from ..neo4j_client import get_graphdb
        graphdb = get_graphdb()

    # Convert absolute path to relative path for Docker Neo4j
    relative_file = _to_relative_path(source_file)

    # If import_all_columns is True, read CSV headers to get all columns
    if import_all_columns:
        try:
            import_dir = get_neo4j_import_dir()
            if import_dir:
                full_path = Path(import_dir) / relative_file
            else:
                full_path = Path(source_file)

            if full_path.exists():
                df = pd.read_csv(full_path, nrows=0)
                all_columns = list(df.columns)
                # Merge with provided properties, removing duplicates
                properties = list(dict.fromkeys(properties + all_columns))
        except Exception as e:
            print(f"Warning: Could not read CSV headers: {e}")

    # Build SET clause for properties (exclude unique column to avoid redundancy)
    set_clauses = ", ".join([f"n.`{prop}` = row.`{prop}`" for prop in properties if prop != unique_column_name])
    set_statement = f"SET {set_clauses}" if set_clauses else ""

    # Neo4j doesn't support parameterized labels, so we use f-string
    # Backticks escape special characters in identifiers
    query = f"""LOAD CSV WITH HEADERS FROM 'file:///{relative_file}' AS row
    CALL (row) {{
        MERGE (n:`{label}` {{ `{unique_column_name}`: row.`{unique_column_name}` }})
        {set_statement}
    }} IN TRANSACTIONS OF 1000 ROWS
    """

    return graphdb.send_query(query)


def load_relationships_from_csv(
    source_file: str,
    relationship_type: str,
    from_node_label: str,
    from_node_column: str,
    to_node_label: str,
    to_node_column: str,
    properties: List[str],
    from_node_property: Optional[str] = None,
    to_node_property: Optional[str] = None,
    graphdb=None,
    import_all_columns: bool = True
) -> Dict[str, Any]:
    """
    Batch load relationships from a CSV file into Neo4j.

    Uses LOAD CSV with MERGE to create relationships between existing nodes.

    Args:
        source_file: CSV file name in the import directory
        relationship_type: Type of relationship to create
        from_node_label: Label of source nodes
        from_node_column: Column with source node identifier
        to_node_label: Label of target nodes
        to_node_column: Column with target node identifier
        properties: List of column names to import as properties
        from_node_property: Property to match on source node (optional)
        to_node_property: Property to match on target node (optional)
        graphdb: Neo4j client instance (optional)
        import_all_columns: If True, import all columns from CSV as properties (default True)

    Returns:
        Dictionary with status indicating success or error
    """
    import pandas as pd

    if graphdb is None:
        from ..neo4j_client import get_graphdb
        graphdb = get_graphdb()

    # Convert absolute path to relative path for Docker Neo4j
    relative_file = _to_relative_path(source_file)

    # If import_all_columns is True, read CSV headers to get all columns
    if import_all_columns:
        try:
            import_dir = get_neo4j_import_dir()
            if import_dir:
                full_path = Path(import_dir) / relative_file
            else:
                full_path = Path(source_file)

            if full_path.exists():
                df = pd.read_csv(full_path, nrows=0)
                all_columns = list(df.columns)
                # Exclude from/to columns and relationship_type metadata
                excluded_cols = {from_node_column, to_node_column, 'relationship_type'}
                property_columns = [c for c in all_columns if c not in excluded_cols]
                # Merge with provided properties, removing duplicates
                properties = list(dict.fromkeys(properties + property_columns))
        except Exception as e:
            print(f"Warning: Could not read relationship CSV headers: {e}")

    # Build SET clause for relationship properties
    set_clauses = ", ".join([f"r.`{prop}` = row.`{prop}`" for prop in properties])
    set_statement = f"SET {set_clauses}" if set_clauses else ""

    from_prop = from_node_property or from_node_column
    to_prop = to_node_property or to_node_column

    # Neo4j doesn't support parameterized labels/types, so we use f-string
    # Backticks escape special characters in identifiers
    query = f"""LOAD CSV WITH HEADERS FROM 'file:///{relative_file}' AS row
    CALL (row) {{
        MATCH (from_node:`{from_node_label}` {{ `{from_prop}`: row.`{from_node_column}` }})
        MATCH (to_node:`{to_node_label}` {{ `{to_prop}`: row.`{to_node_column}` }})
        MERGE (from_node)-[r:`{relationship_type}`]->(to_node)
        {set_statement}
    }} IN TRANSACTIONS OF 1000 ROWS
    """

    return graphdb.send_query(query)


def import_nodes(node_construction: Dict, graphdb=None) -> Dict[str, Any]:
    """
    Import nodes as defined by a node construction rule.

    Creates uniqueness constraint first, then loads nodes from CSV.

    Args:
        node_construction: Dictionary with node construction details
        graphdb: Neo4j client instance (optional)

    Returns:
        Dictionary with status indicating success or error
    """
    # Create uniqueness constraint
    uniqueness_result = create_uniqueness_constraint(
        node_construction["label"],
        node_construction["unique_column_name"],
        graphdb
    )

    if uniqueness_result.get("status") == "error":
        return uniqueness_result

    # Import nodes from CSV
    return load_nodes_from_csv(
        node_construction["source_file"],
        node_construction["label"],
        node_construction["unique_column_name"],
        node_construction["properties"],
        graphdb
    )


def import_relationships(
    relationship_construction: Dict,
    graphdb=None
) -> Dict[str, Any]:
    """
    Import relationships as defined by a relationship construction rule.

    Args:
        relationship_construction: Dictionary with relationship construction details
        graphdb: Neo4j client instance (optional)

    Returns:
        Dictionary with status indicating success or error
    """
    return load_relationships_from_csv(
        relationship_construction["source_file"],
        relationship_construction["relationship_type"],
        relationship_construction["from_node_label"],
        relationship_construction["from_node_column"],
        relationship_construction["to_node_label"],
        relationship_construction["to_node_column"],
        relationship_construction.get("properties", []),
        relationship_construction.get("from_node_property"),
        relationship_construction.get("to_node_property"),
        graphdb
    )


def construct_domain_graph(
    construction_plan: Dict,
    graphdb=None
) -> Dict[str, Any]:
    """
    Construct a domain graph according to a construction plan.

    Processes all node construction rules first, then relationship rules.
    This two-phase approach prevents relationship failures due to missing nodes.

    Args:
        construction_plan: Dictionary of construction rules
        graphdb: Neo4j client instance (optional)

    Returns:
        Dictionary with status and construction results
    """
    # Accept stringified JSON plans
    if isinstance(construction_plan, str):
        try:
            construction_plan = json.loads(construction_plan)
        except Exception:
            return tool_error("Construction plan must be a dict or JSON string representing a dict.")

    if not isinstance(construction_plan, dict):
        return tool_error(f"Construction plan must be a dict, got {type(construction_plan).__name__}")

    results = {"nodes": [], "relationships": []}

    # Normalize older construction plan format (with nodes/relationships lists)
    if "nodes" in construction_plan and isinstance(construction_plan.get("nodes"), list):
        normalized_plan: Dict[str, Dict[str, Any]] = {}
        node_prop_map: Dict[str, str] = {}
        for node in construction_plan.get("nodes", []):
            label = node.get("label")
            if not label:
                continue
            rule = {
                "construction_type": "node",
                "source_file": node.get("file"),
                "label": label,
                "unique_column_name": node.get("unique_property"),
                "properties": node.get("properties", []),
            }
            normalized_plan[f"node::{label}"] = rule
            if node.get("unique_property"):
                node_prop_map[label] = node["unique_property"]

        for rel in construction_plan.get("relationships", []):
            rel_type = rel.get("relationship_type")
            if not rel_type:
                continue
            from_label = rel.get("from_node")
            to_label = rel.get("to_node")
            normalized_plan[f"rel::{rel_type}"] = {
                "construction_type": "relationship",
                "source_file": rel.get("file"),
                "relationship_type": rel_type,
                "from_node_label": from_label,
                "to_node_label": to_label,
                "from_node_column": "from_id",
                "to_node_column": "to_id",
                "from_node_property": node_prop_map.get(from_label, "from_id"),
                "to_node_property": node_prop_map.get(to_label, "to_id"),
                "properties": rel.get("properties", []),
            }
        construction_plan = normalized_plan

    # First, import all nodes
    node_constructions = [
        value for value in construction_plan.values()
        if isinstance(value, dict) and value.get('construction_type') == 'node'
    ]
    for node_construction in node_constructions:
        result = import_nodes(node_construction, graphdb)
        results["nodes"].append({
            "label": node_construction["label"],
            "result": result
        })

    # Second, import all relationships
    relationship_constructions = [
        value for value in construction_plan.values()
        if value.get('construction_type') == 'relationship'
    ]
    for rel_construction in relationship_constructions:
        result = import_relationships(rel_construction, graphdb)
        results["relationships"].append({
            "type": rel_construction["relationship_type"],
            "result": result
        })

    return tool_success("construction_results", results)


def propose_node_construction(
    approved_file: str,
    proposed_label: str,
    unique_column_name: str,
    proposed_properties: List[str],
    tool_context: ToolContext
) -> Dict[str, Any]:
    """
    Propose a node construction for an approved file.

    Args:
        approved_file: The approved file to propose a node construction for
        proposed_label: The proposed label for constructed nodes
        unique_column_name: Column that uniquely identifies nodes
        proposed_properties: Column names to import as properties
        tool_context: ADK ToolContext for state management

    Returns:
        Dictionary with status and the node construction rule
    """
    # Sanity check - verify unique column exists
    search_results = search_file(approved_file, unique_column_name)
    if search_results.get("status") == "error":
        return search_results
    if search_results.get(SEARCH_RESULTS, {}).get("metadata", {}).get("lines_found", 0) == 0:
        return tool_error(
            f"{approved_file} does not have the column {unique_column_name}. "
            "Check the file content and try again."
        )

    # Get or create construction plan
    construction_plan = tool_context.state.get(PROPOSED_CONSTRUCTION_PLAN, {})

    node_construction_rule = {
        "construction_type": "node",
        "source_file": approved_file,
        "label": proposed_label,
        "unique_column_name": unique_column_name,
        "properties": proposed_properties
    }

    construction_plan[proposed_label] = node_construction_rule
    tool_context.state[PROPOSED_CONSTRUCTION_PLAN] = construction_plan

    return tool_success(NODE_CONSTRUCTION, node_construction_rule)


def propose_relationship_construction(
    approved_file: str,
    proposed_relationship_type: str,
    from_node_label: str,
    from_node_column: str,
    to_node_label: str,
    to_node_column: str,
    proposed_properties: List[str],
    tool_context: ToolContext
) -> Dict[str, Any]:
    """
    Propose a relationship construction for an approved file.

    Args:
        approved_file: The approved file for relationship construction
        proposed_relationship_type: The proposed relationship type
        from_node_label: Label of source nodes
        from_node_column: Column with source node identifier
        to_node_label: Label of target nodes
        to_node_column: Column with target node identifier
        proposed_properties: Column names to import as properties
        tool_context: ADK ToolContext for state management

    Returns:
        Dictionary with status and the relationship construction rule
    """
    # Verify from_node_column exists
    search_results = search_file(approved_file, from_node_column)
    if search_results.get("status") == "error":
        return search_results
    if search_results.get(SEARCH_RESULTS, {}).get("metadata", {}).get("lines_found", 0) == 0:
        return tool_error(
            f"{approved_file} does not have the from node column {from_node_column}. "
            "Check the content of the file and reconsider the relationship."
        )

    # Verify to_node_column exists
    search_results = search_file(approved_file, to_node_column)
    if search_results.get("status") == "error":
        return search_results
    if search_results.get(SEARCH_RESULTS, {}).get("metadata", {}).get("lines_found", 0) == 0:
        return tool_error(
            f"{approved_file} does not have the to node column {to_node_column}. "
            "Check the content of the file and reconsider the relationship."
        )

    construction_plan = tool_context.state.get(PROPOSED_CONSTRUCTION_PLAN, {})

    relationship_construction_rule = {
        "construction_type": "relationship",
        "source_file": approved_file,
        "relationship_type": proposed_relationship_type,
        "from_node_label": from_node_label,
        "from_node_column": from_node_column,
        "to_node_label": to_node_label,
        "to_node_column": to_node_column,
        "properties": proposed_properties
    }

    construction_plan[proposed_relationship_type] = relationship_construction_rule
    tool_context.state[PROPOSED_CONSTRUCTION_PLAN] = construction_plan

    return tool_success(RELATIONSHIP_CONSTRUCTION, relationship_construction_rule)


def remove_node_construction(
    node_label: str,
    tool_context: ToolContext
) -> Dict[str, Any]:
    """
    Remove a node construction from the proposed plan.

    Args:
        node_label: The label of the node construction to remove
        tool_context: ADK ToolContext for state management

    Returns:
        Dictionary with status indicating removal result
    """
    construction_plan = tool_context.state.get(PROPOSED_CONSTRUCTION_PLAN, {})

    if node_label not in construction_plan:
        return tool_success(
            "message",
            "Node construction rule not found. Removal not needed."
        )

    del construction_plan[node_label]
    tool_context.state[PROPOSED_CONSTRUCTION_PLAN] = construction_plan

    return tool_success("node_construction_removed", node_label)


def remove_relationship_construction(
    relationship_type: str,
    tool_context: ToolContext
) -> Dict[str, Any]:
    """
    Remove a relationship construction from the proposed plan.

    Args:
        relationship_type: The type of relationship construction to remove
        tool_context: ADK ToolContext for state management

    Returns:
        Dictionary with status indicating removal result
    """
    construction_plan = tool_context.state.get(PROPOSED_CONSTRUCTION_PLAN, {})

    if relationship_type not in construction_plan:
        return tool_success(
            "message",
            "Relationship construction rule not found. Removal not needed."
        )

    del construction_plan[relationship_type]
    tool_context.state[PROPOSED_CONSTRUCTION_PLAN] = construction_plan

    return tool_success("relationship_construction_removed", relationship_type)


def get_proposed_construction_plan(tool_context: ToolContext) -> Dict[str, Any]:
    """
    Get the proposed construction plan.

    Args:
        tool_context: ADK ToolContext for state management

    Returns:
        Dictionary with the construction plan rules
    """
    plan = tool_context.state.get(PROPOSED_CONSTRUCTION_PLAN, {})
    return tool_success(PROPOSED_CONSTRUCTION_PLAN, plan)


def approve_proposed_construction_plan(tool_context: ToolContext) -> Dict[str, Any]:
    """
    Approve the proposed construction plan.

    Args:
        tool_context: ADK ToolContext for state management

    Returns:
        Dictionary with status and the approved plan,
        or error if no plan proposed
    """
    if PROPOSED_CONSTRUCTION_PLAN not in tool_context.state:
        return tool_error("No proposed construction plan found. Propose a plan first.")

    tool_context.state[APPROVED_CONSTRUCTION_PLAN] = tool_context.state[PROPOSED_CONSTRUCTION_PLAN]
    return tool_success(
        APPROVED_CONSTRUCTION_PLAN,
        tool_context.state[APPROVED_CONSTRUCTION_PLAN]
    )


def get_approved_construction_plan(tool_context: ToolContext) -> Dict[str, Any]:
    """
    Get the approved construction plan.

    Args:
        tool_context: ADK ToolContext for state management

    Returns:
        Dictionary with the approved construction plan,
        or error if not approved yet
    """
    if APPROVED_CONSTRUCTION_PLAN not in tool_context.state:
        return tool_error(
            "No approved construction plan found. "
            "Approve a construction plan first."
        )

    return tool_success(
        APPROVED_CONSTRUCTION_PLAN,
        tool_context.state[APPROVED_CONSTRUCTION_PLAN]
    )
