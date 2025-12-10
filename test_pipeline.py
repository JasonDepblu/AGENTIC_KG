#!/usr/bin/env python3
"""
End-to-end smoke test for the schema-first pipeline on data/序号.csv.

This script:
- Designs a schema tailored to “分析一下不同的人对于不同的车型的属性的情感需求”
- Runs targeted preprocessing (entity + relationship extraction)
- Saves extracted CSVs and generates a construction plan
- Optionally imports into Neo4j if connectivity is configured

It avoids WebSocket/UI dependencies so you can validate the core modules quickly.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict

from src.tools.schema_design import (
    approve_target_schema,
    get_target_schema,
    propose_node_type,
    propose_relationship_type,
)
from src.tools.targeted_preprocessing import (
    complete_targeted_preprocessing,
    extract_entities_for_node,
    extract_relationship_data,
    generate_construction_rules,
    get_extraction_summary,
    save_extracted_data,
)
from src.tools.kg_construction import construct_domain_graph


DATA_FILE = "序号.csv"
OUTPUT_DIR = "targeted_preprocessing_output"
USER_QUERY = "分析一下不同的人对于不同的车型的属性的情感需求"


class DummyToolContext:
    """Minimal ToolContext stand-in (only .state is used by tool functions)."""

    def __init__(self) -> None:
        self.state: Dict[str, object] = {}


def ensure_import_dir() -> Path:
    """Ensure NEO4J_IMPORT_DIR points to repo/data so tools can read/write."""
    repo_root = Path(__file__).resolve().parent
    data_dir = repo_root / "data"
    if not data_dir.exists():
        raise FileNotFoundError(f"data directory not found at {data_dir}")
    os.environ.setdefault("NEO4J_IMPORT_DIR", str(data_dir.resolve()))
    return data_dir


def design_schema(ctx: DummyToolContext) -> None:
    """Design and approve a schema matching 序号.csv for the sentiment-by-vehicle query."""
    # Seed user intent and file approval so downstream tools have context
    ctx.state["approved_user_goal"] = {"goal": USER_QUERY}
    ctx.state["approved_files"] = [DATA_FILE]

    def ensure_ok(result: Dict, step: str) -> None:
        if result.get("status") == "error":
            raise RuntimeError(f"{step} failed: {result.get('error_message')}")

    # Node definitions
    ensure_ok(
        propose_node_type(
        label="Respondent",
        unique_property="respondent_id",
        properties=["name", "source_column", "source_file"],
        entity_type="Respondent",
        extraction_hints={
            "source_type": "entity_selection",
            "column_pattern": r"^序号$",
        },
        tool_context=ctx,
        ),
        "Propose Respondent",
    )
    ensure_ok(
        propose_node_type(
        label="Brand",
        unique_property="brand_id",
        properties=["name", "source_column", "source_file"],
        entity_type="Brand",
        extraction_hints={
            "source_type": "entity_selection",
            "column_pattern": "品牌",
        },
        tool_context=ctx,
        ),
        "Propose Brand",
    )
    ensure_ok(
        propose_node_type(
        label="Model",
        unique_property="model_id",
        properties=["name", "source_column", "source_file"],
        entity_type="Model",
        extraction_hints={
            "source_type": "entity_selection",
            "column_pattern": "车型.*配置",
        },
        tool_context=ctx,
        ),
        "Propose Model",
    )
    ensure_ok(
        propose_node_type(
        label="Aspect",
        unique_property="aspect_id",
        properties=["name", "source_column", "source_file"],
        entity_type="Aspect",
        extraction_hints={
            "source_type": "column_header",
            "column_pattern": "打多少分|评分",
            "name_regex": "“([^”]+)”",
        },
        tool_context=ctx,
        ),
        "Propose Aspect",
    )

    # Relationship definitions
    ensure_ok(
        propose_relationship_type(
        relationship_type="EVALUATED_BRAND",
        from_node="Respondent",
        to_node="Brand",
        properties=["source_column"],
        extraction_hints={
            "source_type": "entity_reference",
            "column_pattern": "品牌",
            "respondent_column": r"^序号$",
        },
        tool_context=ctx,
        ),
        "Propose EVALUATED_BRAND",
    )
    ensure_ok(
        propose_relationship_type(
        relationship_type="EVALUATED_MODEL",
        from_node="Respondent",
        to_node="Model",
        properties=["source_column"],
        extraction_hints={
            "source_type": "entity_reference",
            "column_pattern": "车型.*配置",
            "respondent_column": r"^序号$",
        },
        tool_context=ctx,
        ),
        "Propose EVALUATED_MODEL",
    )
    ensure_ok(
        propose_relationship_type(
        relationship_type="RATES",
        from_node="Respondent",
        to_node="Aspect",
        properties=["score", "source_column"],
        extraction_hints={
            "source_type": "rating_column",
            "column_pattern": "打多少分|评分",
            "respondent_column": r"^序号$",
            "name_regex": "“([^”]+)”",
        },
        tool_context=ctx,
        ),
        "Propose RATES",
    )
    ensure_ok(
        propose_relationship_type(
        relationship_type="MODEL_BELONGS_TO_BRAND",
        from_node="Model",
        to_node="Brand",
        properties=["source_column"],
        extraction_hints={
            "source_type": "foreign_key",
            "from_column": "车型.*配置",
            "to_column": "品牌",
        },
        tool_context=ctx,
        ),
        "Propose MODEL_BELONGS_TO_BRAND",
    )

    # Approve schema
    ensure_ok(approve_target_schema(tool_context=ctx), "Approve schema")
    schema_summary = get_target_schema(tool_context=ctx)
    print("\n[Schema]\n", schema_summary.get("target_schema", {}).get("summary", ""))


def run_preprocessing(ctx: DummyToolContext) -> None:
    """Extract entities and relationships according to the approved schema."""
    file_path = DATA_FILE

    for node in ["Respondent", "Brand", "Model", "Aspect"]:
        result = extract_entities_for_node(
            file_path=file_path,
            node_label=node,
            tool_context=ctx,
        )
        if result.get("status") == "error":
            raise RuntimeError(f"Entity extraction failed for {node}: {result}")
        print(f"[Entity Extraction] {node}: {result['extracted_entities']['count']}")

    for rel in ["EVALUATED_BRAND", "EVALUATED_MODEL", "RATES", "MODEL_BELONGS_TO_BRAND"]:
        result = extract_relationship_data(
            file_path=file_path,
            relationship_type=rel,
            tool_context=ctx,
        )
        if result.get("status") == "error":
            raise RuntimeError(f"Relationship extraction failed for {rel}: {result}")
        print(f"[Relationship Extraction] {rel}: {result['extracted_relationships']['count']}")

    summary = get_extraction_summary(tool_context=ctx)
    print("\n[Extraction Summary]\n", json.dumps(summary["extraction_summary"], indent=2, ensure_ascii=False))


def persist_and_plan(ctx: DummyToolContext) -> Dict:
    """Save extracted data and generate a construction plan."""
    saved = save_extracted_data(
        output_dir=OUTPUT_DIR,
        tool_context=ctx,
        prefix="targeted",
    )
    if saved.get("status") == "error":
        raise RuntimeError(saved["error_message"])
    print("\n[Saved Files]")
    for f in saved["saved_files"]["files"]:
        print(f" - {f['type']}: {f['path']} ({f['count']})")

    plan = generate_construction_rules(tool_context=ctx)
    if plan.get("status") == "error":
        raise RuntimeError(plan["error_message"])
    print("\n[Construction Plan]\n", json.dumps(plan["construction_rules"], indent=2, ensure_ascii=False))

    complete_targeted_preprocessing(tool_context=ctx)
    return plan["construction_rules"]


def maybe_build_graph(construction_plan: Dict) -> None:
    """Attempt to build the graph in Neo4j if credentials are present."""
    password = os.getenv("NEO4J_PASSWORD")
    if not password:
        print("\n[Neo4j] NEO4J_PASSWORD not set, skipping graph import.")
        return

    try:
        result = construct_domain_graph(construction_plan)
        print("\n[Neo4j Import Result]\n", json.dumps(result, indent=2, ensure_ascii=False))
    except Exception as exc:
        print(f"\n[Neo4j] Import attempted but failed: {exc}")


def main() -> None:
    ensure_import_dir()
    ctx = DummyToolContext()
    ctx.state["pipeline_mode"] = "schema_first_separate"

    design_schema(ctx)
    run_preprocessing(ctx)
    plan = persist_and_plan(ctx)

    # Only attempt DB import if credentials are present
    maybe_build_graph(plan)

    print("\n✅ Pipeline smoke test finished for data/序号.csv")


if __name__ == "__main__":
    main()
