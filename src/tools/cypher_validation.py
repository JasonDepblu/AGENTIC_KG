"""
Cypher Query Validation Tools for Agentic KG.

Provides tools for generating, validating, and testing Cypher queries
with iterative refinement through the Critic Loop pattern.
"""

from typing import Any, Dict, List, Optional

from google.adk.tools import ToolContext

from .common import tool_success, tool_error
from ..neo4j_client import get_graphdb


# State keys for Cypher validation workflow
PROPOSED_CYPHER = "proposed_cypher_query"
VALIDATED_CYPHER = "validated_cypher_query"
CYPHER_FEEDBACK = "cypher_feedback"
QUERY_RESULT = "query_result"


def get_graph_schema_for_cypher(tool_context: ToolContext = None) -> Dict[str, Any]:
    """
    Get the knowledge graph schema for Cypher query generation.

    Returns node labels, relationship types, and their properties
    to help generate accurate Cypher queries.

    Returns:
        Dict with graph schema including:
        - node_labels: List of node labels with their properties
        - relationship_types: List of relationship types with counts
        - node_counts: Count of nodes per label
    """
    graphdb = get_graphdb()

    # Get node labels with properties
    labels_query = """
        MATCH (n)
        WITH labels(n) AS lbls, keys(n) AS props
        UNWIND lbls AS label
        WITH label, props WHERE NOT label STARTS WITH '__'
        UNWIND props AS prop
        WITH label, collect(DISTINCT prop) AS properties
        RETURN label, properties
        ORDER BY label
    """
    labels_result = graphdb.send_query(labels_query)

    if labels_result.get("status") == "error":
        return tool_error(f"Failed to get labels: {labels_result.get('message')}")

    # Get relationship types with connection info
    rels_query = """
        MATCH (a)-[r]->(b)
        WITH type(r) AS rel_type,
             labels(a) AS from_labels,
             labels(b) AS to_labels,
             count(*) AS count
        RETURN rel_type,
               head([l IN from_labels WHERE NOT l STARTS WITH '__']) AS from_label,
               head([l IN to_labels WHERE NOT l STARTS WITH '__']) AS to_label,
               count
        ORDER BY count DESC
    """
    rels_result = graphdb.send_query(rels_query)

    if rels_result.get("status") == "error":
        return tool_error(f"Failed to get relationships: {rels_result.get('message')}")

    # Get node counts per label
    counts_query = """
        MATCH (n)
        WITH labels(n) AS lbls
        UNWIND lbls AS label
        WITH label WHERE NOT label STARTS WITH '__'
        RETURN label, count(*) AS count
        ORDER BY count DESC
    """
    counts_result = graphdb.send_query(counts_query)

    if counts_result.get("status") == "error":
        return tool_error(f"Failed to get counts: {counts_result.get('message')}")

    schema = {
        "node_labels": labels_result.get("query_result", []),
        "relationship_types": rels_result.get("query_result", []),
        "node_counts": counts_result.get("query_result", []),
    }

    return tool_success("graph_schema", schema)


def propose_cypher_query(
    query: str,
    explanation: str,
    tool_context: ToolContext = None
) -> Dict[str, Any]:
    """
    Propose a Cypher query for validation.

    Call this after generating a Cypher query to submit it for validation.
    The query will be syntax-checked and executed to verify correctness.

    Args:
        query: The Cypher query string
        explanation: Brief explanation of what the query does and why

    Returns:
        Success status with proposed query details
    """
    if not query or not query.strip():
        return tool_error("Query cannot be empty")

    if tool_context:
        tool_context.state[PROPOSED_CYPHER] = {
            "query": query.strip(),
            "explanation": explanation,
        }
        # Clear previous feedback when new query is proposed
        # Use empty string instead of del since state doesn't support __delitem__
        tool_context.state[CYPHER_FEEDBACK] = ""

    return tool_success("proposed_query", {
        "query": query.strip(),
        "explanation": explanation,
        "status": "pending_validation",
    })


def validate_cypher_syntax(tool_context: ToolContext = None) -> Dict[str, Any]:
    """
    Validate the syntax of the proposed Cypher query.

    Uses Neo4j's EXPLAIN to check syntax without executing the query.
    This catches syntax errors before attempting execution.

    Returns:
        Validation result with is_valid flag and any error messages
    """
    if not tool_context or PROPOSED_CYPHER not in tool_context.state:
        return tool_error("No proposed query to validate. Call propose_cypher_query first.")

    proposed = tool_context.state[PROPOSED_CYPHER]
    query = proposed["query"]

    graphdb = get_graphdb()

    # Use EXPLAIN to validate syntax without executing
    explain_query = f"EXPLAIN {query}"
    result = graphdb.send_query(explain_query)

    if result.get("status") == "error":
        error_msg = result.get("message", "Unknown syntax error")
        tool_context.state[CYPHER_FEEDBACK] = f"syntax_error: {error_msg}"
        return tool_success("validation_result", {
            "is_valid": False,
            "error_type": "syntax_error",
            "error": error_msg,
            "suggestion": "Check the Cypher syntax. Common issues: missing colons, incorrect relationship patterns, typos in labels.",
        })

    return tool_success("validation_result", {
        "is_valid": True,
        "message": "Cypher syntax is valid",
    })


def execute_and_validate_query(
    expected_count: Optional[int] = None,
    max_results: int = 100,
    tool_context: ToolContext = None
) -> Dict[str, Any]:
    """
    Execute the proposed query and validate the results.

    Runs the query against Neo4j and performs sanity checks on the results.
    Detects common issues like duplicate counting through relationships.

    Args:
        expected_count: Optional expected result count for validation
        max_results: Maximum results to return (default 100)

    Returns:
        Execution result with validation status and query results
    """
    if not tool_context or PROPOSED_CYPHER not in tool_context.state:
        return tool_error("No proposed query to execute. Call propose_cypher_query first.")

    proposed = tool_context.state[PROPOSED_CYPHER]
    query = proposed["query"]

    graphdb = get_graphdb()
    result = graphdb.send_query(query)

    if result.get("status") == "error":
        error_msg = result.get("message", "Query execution failed")
        tool_context.state[CYPHER_FEEDBACK] = f"execution_error: {error_msg}"
        return tool_success("execution_result", {
            "is_valid": False,
            "error_type": "execution_error",
            "error": error_msg,
            "suggestion": "Check that all node labels and relationship types exist in the graph.",
        })

    query_result = result.get("query_result", [])
    result_count = len(query_result)

    # Store result for later use
    tool_context.state[QUERY_RESULT] = query_result[:max_results]

    # Perform sanity checks
    issues = []
    suggestions = []

    # Check for empty results
    if result_count == 0:
        issues.append("Query returned no results")
        suggestions.append("Verify filter conditions and node/relationship labels exist")

    # Check against expected count if provided
    if expected_count is not None:
        if result_count != expected_count:
            issues.append(f"Expected {expected_count} results, got {result_count}")
            if result_count > expected_count:
                suggestions.append("May be counting duplicates - try using COUNT(DISTINCT node) instead of COUNT(*)")

    # Check for suspiciously high counts in aggregation results
    if result_count > 0 and isinstance(query_result[0], dict):
        for row in query_result:
            for key, value in row.items():
                if isinstance(value, (int, float)) and 'count' in key.lower():
                    if value > 1000:
                        issues.append(f"Very high count value ({value}) for '{key}' - may indicate duplicate counting")
                        suggestions.append("When counting through relationships, use COUNT(DISTINCT node) to avoid duplicates")
                        break

    # Check if query appears to traverse relationships but doesn't use DISTINCT
    query_lower = query.lower()
    if ('-[' in query_lower or ']->' in query_lower or '<-[' in query_lower):
        if 'count(*)' in query_lower and 'distinct' not in query_lower:
            issues.append("Query uses COUNT(*) while traversing relationships")
            suggestions.append("Replace COUNT(*) with COUNT(DISTINCT node) to count unique entities")

    if issues:
        feedback = "; ".join(issues)
        tool_context.state[CYPHER_FEEDBACK] = f"logic_warning: {feedback}"
        return tool_success("execution_result", {
            "is_valid": False,
            "error_type": "logic_warning",
            "warnings": issues,
            "suggestions": suggestions,
            "result_count": result_count,
            "sample_results": query_result[:5],
        })

    # All checks passed
    tool_context.state[CYPHER_FEEDBACK] = "valid"
    tool_context.state[VALIDATED_CYPHER] = query

    return tool_success("execution_result", {
        "is_valid": True,
        "result_count": result_count,
        "results": query_result[:max_results],
    })


def get_cypher_feedback(tool_context: ToolContext = None) -> Dict[str, Any]:
    """
    Get the current validation feedback for the proposed query.

    Use this to check what feedback was given by the validator
    so you can address issues in the next iteration.

    Returns:
        Current feedback status and proposed query
    """
    if not tool_context:
        return tool_error("No context available")

    feedback = tool_context.state.get(CYPHER_FEEDBACK, "")
    proposed = tool_context.state.get(PROPOSED_CYPHER, {})

    return tool_success("feedback", {
        "feedback": feedback,
        "proposed_query": proposed.get("query", ""),
        "explanation": proposed.get("explanation", ""),
    })


def approve_cypher_query(tool_context: ToolContext = None) -> Dict[str, Any]:
    """
    Approve the validated Cypher query.

    Call this when validation passes and the query is ready to use.
    This marks the query as approved and returns the final results.

    Returns:
        Approved query and its execution results
    """
    if not tool_context:
        return tool_error("No context available")

    if VALIDATED_CYPHER not in tool_context.state:
        return tool_error(
            "No validated query to approve. "
            "Run execute_and_validate_query first and ensure it passes."
        )

    validated_query = tool_context.state[VALIDATED_CYPHER]
    query_result = tool_context.state.get(QUERY_RESULT, [])

    # Mark as valid
    tool_context.state[CYPHER_FEEDBACK] = "valid"

    return tool_success("approved", {
        "query": validated_query,
        "result_count": len(query_result),
        "results": query_result,
    })


def get_sample_data(
    label: str,
    limit: int = 5,
    tool_context: ToolContext = None
) -> Dict[str, Any]:
    """
    Get sample data from a specific node label.

    Useful for understanding the data structure and property values
    before writing queries.

    Args:
        label: The node label to sample (e.g., 'Respondent', 'Store')
        limit: Maximum number of samples to return

    Returns:
        Sample nodes with their properties
    """
    graphdb = get_graphdb()

    # Validate label exists
    check_query = """
        CALL db.labels() YIELD label
        WHERE NOT label STARTS WITH '__'
        RETURN collect(label) AS labels
    """
    check_result = graphdb.send_query(check_query)

    if check_result.get("status") == "error":
        return tool_error(f"Failed to get labels: {check_result.get('message')}")

    valid_labels = check_result.get("query_result", [{}])[0].get("labels", [])
    if label not in valid_labels:
        return tool_error(f"Label '{label}' not found. Valid labels: {valid_labels}")

    # Get sample data
    sample_query = f"""
        MATCH (n:{label})
        RETURN properties(n) AS properties
        LIMIT $limit
    """
    result = graphdb.send_query(sample_query, {"limit": limit})

    if result.get("status") == "error":
        return tool_error(f"Failed to get samples: {result.get('message')}")

    return tool_success("sample_data", {
        "label": label,
        "samples": result.get("query_result", []),
        "count": len(result.get("query_result", [])),
    })
