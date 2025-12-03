"""
Knowledge graph construction tools for Agentic KG.

Tools for building and managing knowledge graph schemas and data import.
"""

from typing import Any, Dict, List

from google.adk.tools import ToolContext

from .common import tool_success, tool_error
from .file_suggestion import search_file, SEARCH_RESULTS

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
    graphdb=None
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

    Returns:
        Dictionary with status indicating success or error
    """
    if graphdb is None:
        from ..neo4j_client import get_graphdb
        graphdb = get_graphdb()

    # Build SET clause for properties
    set_clauses = ", ".join([f"n.`{prop}` = row.`{prop}`" for prop in properties if prop != unique_column_name])
    set_statement = f"SET {set_clauses}" if set_clauses else ""

    # Neo4j doesn't support parameterized labels, so we use f-string
    # Backticks escape special characters in identifiers
    query = f"""LOAD CSV WITH HEADERS FROM 'file:///{source_file}' AS row
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
    graphdb=None
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
        graphdb: Neo4j client instance (optional)

    Returns:
        Dictionary with status indicating success or error
    """
    if graphdb is None:
        from ..neo4j_client import get_graphdb
        graphdb = get_graphdb()

    # Build SET clause for relationship properties
    set_clauses = ", ".join([f"r.`{prop}` = row.`{prop}`" for prop in properties])
    set_statement = f"SET {set_clauses}" if set_clauses else ""

    # Neo4j doesn't support parameterized labels/types, so we use f-string
    # Backticks escape special characters in identifiers
    query = f"""LOAD CSV WITH HEADERS FROM 'file:///{source_file}' AS row
    CALL (row) {{
        MATCH (from_node:`{from_node_label}` {{ `{from_node_column}`: row.`{from_node_column}` }})
        MATCH (to_node:`{to_node_label}` {{ `{to_node_column}`: row.`{to_node_column}` }})
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
        relationship_construction["properties"],
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
    results = {"nodes": [], "relationships": []}

    # First, import all nodes
    node_constructions = [
        value for value in construction_plan.values()
        if value.get('construction_type') == 'node'
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
