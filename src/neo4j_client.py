"""
Neo4j client module for Agentic KG.

Provides a wrapper for Neo4j database operations with ADK-friendly responses.
"""

import atexit
from typing import Any, Dict, Optional

from neo4j import GraphDatabase, Result
from neo4j.graph import Node, Relationship, Path
from neo4j import Record
import neo4j.time

from .config import get_config
from .tools.common import tool_success, tool_error


def to_python(value: Any) -> Any:
    """
    Convert Neo4j types to Python native types.

    This ensures that all values returned from Neo4j queries are
    JSON-serializable and compatible with ADK tools.

    Args:
        value: Any Neo4j value (Node, Relationship, Path, Record, etc.)

    Returns:
        Python native type equivalent
    """
    if isinstance(value, Record):
        return {k: to_python(v) for k, v in value.items()}
    elif isinstance(value, dict):
        return {k: to_python(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [to_python(v) for v in value]
    elif isinstance(value, Node):
        return {
            "id": value.id,
            "labels": list(value.labels),
            "properties": to_python(dict(value))
        }
    elif isinstance(value, Relationship):
        return {
            "id": value.id,
            "type": value.type,
            "start_node": value.start_node.id,
            "end_node": value.end_node.id,
            "properties": to_python(dict(value))
        }
    elif isinstance(value, Path):
        return {
            "nodes": [to_python(node) for node in value.nodes],
            "relationships": [to_python(rel) for rel in value.relationships]
        }
    elif isinstance(value, neo4j.time.DateTime):
        return value.iso_format()
    elif isinstance(value, (neo4j.time.Date, neo4j.time.Time, neo4j.time.Duration)):
        return str(value)
    else:
        return value


def result_to_adk(result: Result) -> Dict[str, Any]:
    """
    Convert Neo4j query result to ADK-friendly format.

    Args:
        result: Neo4j query result

    Returns:
        Dictionary with 'status' and 'query_result' keys
    """
    eager_result = result.to_eager_result()
    records = [to_python(record.data()) for record in eager_result.records]
    return tool_success("query_result", records)


class Neo4jClient:
    """
    A wrapper for querying Neo4j which returns ADK-friendly responses.

    This class implements a singleton pattern to ensure only one connection
    pool is maintained throughout the application lifecycle.
    """

    _instance: Optional["Neo4jClient"] = None
    _driver = None
    database_name: str = "neo4j"

    def __new__(cls) -> "Neo4jClient":
        """Implement singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self) -> None:
        """Initialize the Neo4j connection."""
        config = get_config()
        self.database_name = config.neo4j.database

        self._driver = GraphDatabase.driver(
            config.neo4j.uri,
            auth=(config.neo4j.username, config.neo4j.password)
        )

        # Register cleanup on exit
        atexit.register(self.close)

    def get_driver(self):
        """Get the underlying Neo4j driver."""
        return self._driver

    def close(self) -> None:
        """Close the database connection."""
        if self._driver:
            self._driver.close()
            self._driver = None

    def send_query(
        self,
        cypher_query: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute a Cypher query and return ADK-friendly results.

        Args:
            cypher_query: The Cypher query to execute
            parameters: Optional query parameters

        Returns:
            Dictionary with 'status' key ('success' or 'error')
            If success: includes 'query_result' with the data
            If error: includes 'error_message' with error details
        """
        session = self._driver.session()
        try:
            result = session.run(
                cypher_query,
                parameters or {},
                database_=self.database_name
            )
            return result_to_adk(result)
        except Exception as e:
            return tool_error(str(e))
        finally:
            session.close()

    def verify_connection(self) -> Dict[str, Any]:
        """
        Verify the Neo4j connection is working.

        Returns:
            Dictionary with status and message
        """
        return self.send_query("RETURN 'Neo4j is Ready!' as message")

    def get_import_directory(self) -> Dict[str, Any]:
        """
        Get the Neo4j import directory path.

        Returns:
            Dictionary with status and import directory path
        """
        config = get_config()
        import_dir = config.neo4j.import_dir

        if import_dir:
            return tool_success("neo4j_import_dir", import_dir)
        else:
            return tool_error(
                "NEO4J_IMPORT_DIR not configured. "
                "Set it in .env file or environment variables."
            )

    def clear_database(self) -> Dict[str, Any]:
        """
        Clear all nodes and relationships from the database.

        WARNING: This is destructive and cannot be undone!

        Returns:
            Dictionary with status
        """
        return self.send_query("MATCH (n) DETACH DELETE n")

    def get_schema(self) -> Dict[str, Any]:
        """
        Get the current database schema (labels, relationship types, properties).

        Returns:
            Dictionary with schema information
        """
        return self.send_query("""
            CALL db.schema.visualization()
            YIELD nodes, relationships
            RETURN nodes, relationships
        """)

    def get_node_count(self) -> Dict[str, Any]:
        """Get the total number of nodes in the database."""
        return self.send_query("MATCH (n) RETURN count(n) as node_count")

    def get_relationship_count(self) -> Dict[str, Any]:
        """Get the total number of relationships in the database."""
        return self.send_query("MATCH ()-[r]->() RETURN count(r) as relationship_count")


# Global instance for convenience
_graphdb: Optional[Neo4jClient] = None


def get_graphdb() -> Neo4jClient:
    """Get the global Neo4j client instance."""
    global _graphdb
    if _graphdb is None:
        _graphdb = Neo4jClient()
    return _graphdb


# Backward compatibility alias
graphdb = get_graphdb()
