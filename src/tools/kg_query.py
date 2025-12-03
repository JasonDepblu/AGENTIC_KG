"""
Knowledge Graph Query Tools for Agentic KG.

Provides tools for querying the knowledge graph using natural language
and retrieving relevant information from Neo4j.
"""

from typing import Any, Dict, List, Optional

from google.adk.tools import ToolContext

from .common import tool_success, tool_error
from ..neo4j_client import get_graphdb
from ..config import get_config


# State keys
QUERY_HISTORY = "query_history"


def get_graph_schema(tool_context: ToolContext = None) -> Dict[str, Any]:
    """
    Get the current knowledge graph schema.

    Returns information about node labels, relationship types,
    and their properties to help formulate queries.

    Returns:
        Dict with node labels, relationship types, and schema details
    """
    graphdb = get_graphdb()

    # Get node labels and counts
    labels_result = graphdb.send_query("""
        MATCH (n)
        WITH labels(n) AS lbls
        UNWIND lbls AS label
        WITH label WHERE NOT label STARTS WITH '__'
        RETURN label, count(*) AS count
        ORDER BY count DESC
    """)

    if labels_result.get("status") == "error":
        return tool_error(f"Failed to get labels: {labels_result.get('message')}")

    # Get relationship types and counts
    rels_result = graphdb.send_query("""
        MATCH ()-[r]->()
        RETURN type(r) AS relationship_type, count(*) AS count
        ORDER BY count DESC
    """)

    if rels_result.get("status") == "error":
        return tool_error(f"Failed to get relationships: {rels_result.get('message')}")

    # Get sample properties for each label
    properties_result = graphdb.send_query("""
        MATCH (n)
        WITH labels(n) AS lbls, keys(n) AS props
        UNWIND lbls AS label
        WITH label, props WHERE NOT label STARTS WITH '__'
        UNWIND props AS prop
        RETURN label, collect(DISTINCT prop) AS properties
        ORDER BY label
    """)

    schema = {
        "node_labels": labels_result.get("query_result", []),
        "relationship_types": rels_result.get("query_result", []),
        "label_properties": properties_result.get("query_result", [])
    }

    return tool_success("graph_schema", schema)


def query_graph_cypher(
    cypher_query: str,
    parameters: Optional[Dict[str, Any]] = None,
    tool_context: ToolContext = None
) -> Dict[str, Any]:
    """
    Execute a Cypher query against the knowledge graph.

    Use this tool to run custom Cypher queries when you need
    specific data from the graph.

    Args:
        cypher_query: The Cypher query to execute
        parameters: Optional query parameters

    Returns:
        Dict with query results
    """
    graphdb = get_graphdb()

    result = graphdb.send_query(cypher_query, parameters)

    if result.get("status") == "error":
        return tool_error(f"Query failed: {result.get('message')}")

    # Store in query history
    if tool_context:
        history = tool_context.state.get(QUERY_HISTORY, [])
        history.append({
            "query": cypher_query,
            "parameters": parameters,
            "result_count": len(result.get("query_result", []))
        })
        tool_context.state[QUERY_HISTORY] = history[-10:]  # Keep last 10

    return result


def find_best_stores(
    metric: str = "rating",
    limit: int = 5,
    tool_context: ToolContext = None
) -> Dict[str, Any]:
    """
    Find the best stores based on ratings or opinions.

    Args:
        metric: The metric to rank by ('rating', 'positive_opinions', 'review_count')
        limit: Number of top stores to return

    Returns:
        Dict with ranked stores and their scores
    """
    graphdb = get_graphdb()

    if metric == "rating":
        # Query stores with average ratings
        query = """
            MATCH (r:Respondent)-[rating:GAVE_RATING]->(s:Store)
            WITH s, avg(toFloat(rating.rating)) AS avg_rating, count(r) AS review_count
            RETURN s.name AS store_name,
                   round(avg_rating * 100) / 100 AS average_rating,
                   review_count
            ORDER BY avg_rating DESC, review_count DESC
            LIMIT $limit
        """
    elif metric == "positive_opinions":
        # Query stores with most positive opinions
        query = """
            MATCH (r:Respondent)-[op:EXPRESSED_OPINION]->(a:Aspect)
            MATCH (r)-[:REFERRED_TO_STORE]->(s:Store)
            WHERE op.sentiment = 'positive' OR toFloat(op.rating) >= 4
            WITH s, count(op) AS positive_count
            RETURN s.name AS store_name, positive_count
            ORDER BY positive_count DESC
            LIMIT $limit
        """
    else:
        # Default: count reviews
        query = """
            MATCH (r:Respondent)-[:REFERRED_TO_STORE]->(s:Store)
            WITH s, count(r) AS review_count
            RETURN s.name AS store_name, review_count
            ORDER BY review_count DESC
            LIMIT $limit
        """

    result = graphdb.send_query(query, {"limit": limit})

    if result.get("status") == "error":
        return tool_error(f"Query failed: {result.get('message')}")

    return tool_success("best_stores", {
        "metric": metric,
        "stores": result.get("query_result", [])
    })


def find_store_opinions(
    store_name: Optional[str] = None,
    aspect: Optional[str] = None,
    sentiment: Optional[str] = None,
    limit: int = 20,
    tool_context: ToolContext = None
) -> Dict[str, Any]:
    """
    Find opinions about stores, optionally filtered by store, aspect, or sentiment.

    Args:
        store_name: Filter by store name (partial match)
        aspect: Filter by aspect (e.g., 'service', 'price', 'quality')
        sentiment: Filter by sentiment ('positive', 'negative', 'neutral')
        limit: Maximum number of results

    Returns:
        Dict with matching opinions
    """
    graphdb = get_graphdb()

    # Build dynamic query based on filters
    where_clauses = []
    parameters = {"limit": limit}

    if store_name:
        where_clauses.append("toLower(s.name) CONTAINS toLower($store_name)")
        parameters["store_name"] = store_name

    if aspect:
        where_clauses.append("toLower(a.name) CONTAINS toLower($aspect)")
        parameters["aspect"] = aspect

    if sentiment:
        where_clauses.append("op.sentiment = $sentiment")
        parameters["sentiment"] = sentiment

    where_clause = " AND ".join(where_clauses) if where_clauses else "true"

    query = f"""
        MATCH (r:Respondent)-[op:EXPRESSED_OPINION]->(a:Aspect)
        OPTIONAL MATCH (r)-[:REFERRED_TO_STORE]->(s:Store)
        WHERE {where_clause}
        RETURN s.name AS store_name,
               a.name AS aspect,
               op.opinion AS opinion,
               op.sentiment AS sentiment,
               op.rating AS rating
        LIMIT $limit
    """

    result = graphdb.send_query(query, parameters)

    if result.get("status") == "error":
        return tool_error(f"Query failed: {result.get('message')}")

    return tool_success("opinions", result.get("query_result", []))


def get_graph_statistics(tool_context: ToolContext = None) -> Dict[str, Any]:
    """
    Get overall statistics about the knowledge graph.

    Returns counts of nodes, relationships, and key metrics.
    """
    graphdb = get_graphdb()

    # Get comprehensive stats
    query = """
        MATCH (n)
        WITH count(n) AS total_nodes
        MATCH ()-[r]->()
        WITH total_nodes, count(r) AS total_relationships
        MATCH (n)
        WITH total_nodes, total_relationships, labels(n) AS lbls
        UNWIND lbls AS label
        WITH total_nodes, total_relationships, label WHERE NOT label STARTS WITH '__'
        WITH total_nodes, total_relationships, collect(DISTINCT label) AS labels
        RETURN total_nodes, total_relationships, labels, size(labels) AS label_count
    """

    result = graphdb.send_query(query)

    if result.get("status") == "error":
        return tool_error(f"Failed to get statistics: {result.get('message')}")

    stats = result.get("query_result", [{}])[0] if result.get("query_result") else {}

    return tool_success("graph_statistics", stats)


def search_entities(
    search_term: str,
    label: Optional[str] = None,
    limit: int = 10,
    tool_context: ToolContext = None
) -> Dict[str, Any]:
    """
    Search for entities in the knowledge graph by name or property values.

    Args:
        search_term: The term to search for (case-insensitive partial match)
        label: Optional label to filter by (e.g., 'Store', 'Brand', 'Model')
        limit: Maximum number of results

    Returns:
        Dict with matching entities
    """
    graphdb = get_graphdb()

    parameters = {
        "search_term": search_term.lower(),
        "limit": limit
    }

    if label:
        query = f"""
            MATCH (n:{label})
            WHERE any(prop IN keys(n) WHERE toLower(toString(n[prop])) CONTAINS $search_term)
            RETURN labels(n) AS labels, properties(n) AS properties
            LIMIT $limit
        """
    else:
        query = """
            MATCH (n)
            WHERE NOT any(l IN labels(n) WHERE l STARTS WITH '__')
            AND any(prop IN keys(n) WHERE toLower(toString(n[prop])) CONTAINS $search_term)
            RETURN labels(n) AS labels, properties(n) AS properties
            LIMIT $limit
        """

    result = graphdb.send_query(query, parameters)

    if result.get("status") == "error":
        return tool_error(f"Search failed: {result.get('message')}")

    return tool_success("search_results", result.get("query_result", []))


def analyze_aspect_sentiment(
    aspect: Optional[str] = None,
    tool_context: ToolContext = None
) -> Dict[str, Any]:
    """
    Analyze sentiment distribution for aspects across all reviews.

    Args:
        aspect: Optional specific aspect to analyze (e.g., 'service', 'price')

    Returns:
        Dict with sentiment analysis results per aspect
    """
    graphdb = get_graphdb()

    parameters = {}
    where_clause = ""

    if aspect:
        where_clause = "WHERE toLower(a.name) CONTAINS toLower($aspect)"
        parameters["aspect"] = aspect

    query = f"""
        MATCH (r:Respondent)-[op:EXPRESSED_OPINION]->(a:Aspect)
        {where_clause}
        WITH a.name AS aspect,
             sum(CASE WHEN op.sentiment = 'positive' OR toFloat(op.rating) >= 4 THEN 1 ELSE 0 END) AS positive,
             sum(CASE WHEN op.sentiment = 'negative' OR toFloat(op.rating) <= 2 THEN 1 ELSE 0 END) AS negative,
             sum(CASE WHEN op.sentiment = 'neutral' OR (toFloat(op.rating) > 2 AND toFloat(op.rating) < 4) THEN 1 ELSE 0 END) AS neutral,
             count(op) AS total
        RETURN aspect, positive, negative, neutral, total,
               round(toFloat(positive) / total * 100) AS positive_pct
        ORDER BY total DESC
        LIMIT 20
    """

    result = graphdb.send_query(query, parameters)

    if result.get("status") == "error":
        return tool_error(f"Analysis failed: {result.get('message')}")

    return tool_success("aspect_sentiment", result.get("query_result", []))
