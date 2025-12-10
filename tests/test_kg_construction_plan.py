"""
Unit tests for construct_domain_graph to cover plan format compatibility.

These tests use a fake graph client to avoid real Neo4j calls and
exercise:
- New plan format (construction_type keyed dict)
- Stringified JSON plans
- Legacy plan format with nodes/relationships lists
"""

import json
import sys
import types

import pytest

# Provide a lightweight stub for google.adk.tools.ToolContext if ADK isn't installed
if "google" not in sys.modules:
    google_mod = types.ModuleType("google")
    adk_mod = types.ModuleType("google.adk")
    tools_mod = types.ModuleType("google.adk.tools")

    class _StubToolContext:
        def __init__(self):
            self.state = {}

    tools_mod.ToolContext = _StubToolContext
    sys.modules["google"] = google_mod
    sys.modules["google.adk"] = adk_mod
    sys.modules["google.adk.tools"] = tools_mod

# Stub neo4j to avoid optional dependency during unit tests
if "neo4j" not in sys.modules:
    neo4j_mod = types.ModuleType("neo4j")
    neo4j_mod.__path__ = []  # treat as package
    graph_submod = types.ModuleType("neo4j.graph")
    time_submod = types.ModuleType("neo4j.time")

    class _StubResult:
        pass
    class _StubRecord:
        pass

    class _StubGraphDatabase:
        @staticmethod
        def driver(*args, **kwargs):
            raise RuntimeError("neo4j driver not available in test stub")

    class _StubNode:
        pass

    class _StubRelationship:
        pass

    class _StubPath:
        pass

    neo4j_mod.Result = _StubResult
    neo4j_mod.Record = _StubRecord
    neo4j_mod.GraphDatabase = _StubGraphDatabase
    graph_submod.Node = _StubNode
    graph_submod.Relationship = _StubRelationship
    graph_submod.Path = _StubPath
    sys.modules["neo4j"] = neo4j_mod
    sys.modules["neo4j.graph"] = graph_submod
    sys.modules["neo4j.time"] = time_submod

# Stub src.neo4j_client to prevent real driver init during import of src.tools.*
if "src.neo4j_client" not in sys.modules:
    neo4j_client_stub = types.ModuleType("src.neo4j_client")
    neo4j_client_stub.get_graphdb = lambda: None
    neo4j_client_stub.Neo4jClient = object
    sys.modules["src.neo4j_client"] = neo4j_client_stub

from src.tools.kg_construction import construct_domain_graph


class FakeGraphDB:
    """Collects Cypher queries instead of sending them to Neo4j."""

    def __init__(self) -> None:
        self.queries = []

    def send_query(self, query: str):
        self.queries.append(query)
        return {"status": "success", "query": query}


def _assert_success(result):
    assert result["status"] == "success"
    assert "construction_results" in result


def test_construct_domain_graph_with_new_format():
    """New format: dict keyed by node::/rel:: with construction_type."""
    plan = {
        "node::Respondent": {
            "construction_type": "node",
            "source_file": "targeted_respondent_entities.csv",
            "label": "Respondent",
            "unique_column_name": "respondent_id",
            "properties": ["name"],
        },
        "node::Model": {
            "construction_type": "node",
            "source_file": "targeted_model_entities.csv",
            "label": "Model",
            "unique_column_name": "model_id",
            "properties": ["name"],
        },
        "rel::EVALUATED": {
            "construction_type": "relationship",
            "source_file": "targeted_evaluated_relationships.csv",
            "relationship_type": "EVALUATED",
            "from_node_label": "Respondent",
            "to_node_label": "Model",
            "from_node_column": "from_id",
            "to_node_column": "to_id",
            "from_node_property": "respondent_id",
            "to_node_property": "model_id",
            "properties": ["source_column"],
        },
    }

    graphdb = FakeGraphDB()
    result = construct_domain_graph(plan, graphdb=graphdb)
    _assert_success(result)

    # Each node generates constraint + load, plus one relationship
    assert len(graphdb.queries) == 5
    rel_query = graphdb.queries[-1]
    assert "EVALUATED" in rel_query
    # Ensure mapping uses the unique properties instead of CSV column names
    assert "`respondent_id`: row.`from_id`" in rel_query
    assert "`model_id`: row.`to_id`" in rel_query


def test_construct_domain_graph_with_string_plan():
    """Plan provided as JSON string should be parsed and processed."""
    plan_dict = {
        "node::Aspect": {
            "construction_type": "node",
            "source_file": "targeted_aspect_entities.csv",
            "label": "Aspect",
            "unique_column_name": "aspect_id",
            "properties": ["name"],
        }
    }
    plan_str = json.dumps(plan_dict)

    graphdb = FakeGraphDB()
    result = construct_domain_graph(plan_str, graphdb=graphdb)
    _assert_success(result)
    # Constraint + load for Aspect
    assert len(graphdb.queries) == 2
    assert "Aspect" in graphdb.queries[0]


def test_construct_domain_graph_with_legacy_format():
    """Legacy plan (nodes/relationships lists) should be normalized."""
    legacy_plan = {
        "schema_name": "Legacy",
        "nodes": [
            {
                "label": "Respondent",
                "file": "targeted_respondent_entities.csv",
                "unique_property": "respondent_id",
                "properties": ["name"],
            },
            {
                "label": "Store",
                "file": "targeted_store_entities.csv",
                "unique_property": "store_id",
                "properties": ["name"],
            },
        ],
        "relationships": [
            {
                "relationship_type": "VISITED",
                "from_node": "Respondent",
                "to_node": "Store",
                "file": "targeted_visited_relationships.csv",
                "properties": ["source_column"],
            }
        ],
    }

    graphdb = FakeGraphDB()
    result = construct_domain_graph(legacy_plan, graphdb=graphdb)
    _assert_success(result)

    # 2 nodes (constraint + load each) + 1 relationship => 5 queries
    assert len(graphdb.queries) == 5
    rel_query = graphdb.queries[-1]
    # Legacy normalization should default to from_id/to_id column names
    assert "VISITED" in rel_query
    assert "`Respondent`" in rel_query
    assert "`Store`" in rel_query
