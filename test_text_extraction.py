#!/usr/bin/env python3
"""
Test script for text feedback column extraction functionality.

Tests the new text extraction tools:
1. identify_text_feedback_columns - Find text columns
2. sample_text_column - Get text samples
3. analyze_text_column_entities - LLM entity type analysis
4. analyze_text_column_relationships - LLM relationship type analysis
5. add_text_entity_to_schema - Add to schema
6. extract_entities_from_text_column - Extract entity instances
7. extract_relationships_from_text_column - Extract relationships
8. deduplicate_entities_with_llm - Merge similar entities
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict

# Ensure NEO4J_IMPORT_DIR is set
repo_root = Path(__file__).resolve().parent
data_dir = repo_root / "data"
os.environ.setdefault("NEO4J_IMPORT_DIR", str(data_dir.resolve()))

from src.tools.schema_design import (
    identify_text_feedback_columns,
    sample_text_column,
    analyze_text_column_entities,
    analyze_text_column_relationships,
    add_text_entity_to_schema,
    add_text_relationship_to_schema,
    get_text_analysis_summary,
    propose_node_type,
    approve_target_schema,
    get_target_schema,
)
from src.tools.targeted_preprocessing import (
    extract_entities_from_text_column,
    extract_relationships_from_text_column,
    deduplicate_entities_with_llm,
    get_extraction_summary,
    save_extracted_data,
)


DATA_FILE = "序号_standardized.csv"


class DummyToolContext:
    """Minimal ToolContext stand-in."""

    def __init__(self) -> None:
        self.state: Dict[str, object] = {}


def print_section(title: str) -> None:
    """Print a section header."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print('=' * 60)


def test_identify_text_columns(ctx: DummyToolContext) -> list:
    """Test identifying text feedback columns."""
    print_section("1. Identify Text Feedback Columns")

    result = identify_text_feedback_columns(
        file_path=DATA_FILE,
        tool_context=ctx
    )

    if result.get("status") == "error":
        print(f"ERROR: {result.get('error_message')}")
        return []

    text_cols = result.get("text_feedback_columns", {})
    print(f"Found {text_cols.get('count', 0)} text feedback columns:")

    for col in text_cols.get("text_columns", [])[:10]:  # Show first 10
        print(f"  - {col['column_name']}")
        print(f"    Category: {col['category']}, Fill rate: {col['fill_rate']}%")
        if col.get('sample_values'):
            print(f"    Sample: {col['sample_values'][0][:50]}...")

    if text_cols.get('count', 0) > 10:
        print(f"  ... and {text_cols['count'] - 10} more columns")

    return text_cols.get("text_columns", [])


def test_sample_text_column(ctx: DummyToolContext, column_name: str) -> list:
    """Test sampling a text column."""
    print_section("2. Sample Text Column")

    result = sample_text_column(
        file_path=DATA_FILE,
        column_name=column_name,
        tool_context=ctx,
        sample_size=10
    )

    if result.get("status") == "error":
        print(f"ERROR: {result.get('error_message')}")
        return []

    samples_data = result.get("text_samples", {})
    print(f"Column: {samples_data.get('column_name')}")
    print(f"Statistics: {json.dumps(samples_data.get('statistics', {}), ensure_ascii=False)}")

    samples = samples_data.get("samples", [])
    print(f"\nSample texts ({len(samples)}):")
    for i, s in enumerate(samples[:5]):
        print(f"  {i+1}. {s[:100]}...")

    return samples


def test_analyze_entities(ctx: DummyToolContext, column_name: str, samples: list, category: str) -> list:
    """Test LLM entity type analysis."""
    print_section("3. Analyze Text Column Entities (LLM)")

    result = analyze_text_column_entities(
        column_name=column_name,
        samples=samples,
        column_category=category,
        tool_context=ctx
    )

    if result.get("status") == "error":
        print(f"ERROR: {result.get('error_message')}")
        return []

    analysis = result.get("entity_analysis", {})
    print(f"Column: {analysis.get('column_name')}")
    print(f"Category: {analysis.get('column_category')}")
    print(f"Analysis notes: {analysis.get('analysis_notes', '')}")

    entity_types = analysis.get("entity_types", [])
    print(f"\nDiscovered entity types ({len(entity_types)}):")
    for et in entity_types:
        print(f"  - Type: {et.get('type')}")
        print(f"    Description: {et.get('description', '')}")
        print(f"    Examples: {et.get('examples', [])[:3]}")

    return [et.get("type") for et in entity_types]


def test_analyze_relationships(ctx: DummyToolContext, column_name: str, samples: list,
                               entity_types: list, category: str) -> list:
    """Test LLM relationship type analysis."""
    print_section("4. Analyze Text Column Relationships (LLM)")

    result = analyze_text_column_relationships(
        column_name=column_name,
        samples=samples,
        entity_types=entity_types,
        column_category=category,
        tool_context=ctx
    )

    if result.get("status") == "error":
        print(f"ERROR: {result.get('error_message')}")
        return []

    analysis = result.get("relationship_analysis", {})
    print(f"Entity types used: {analysis.get('entity_types_used', [])}")
    print(f"Analysis notes: {analysis.get('analysis_notes', '')}")

    rel_types = analysis.get("relationship_types", [])
    print(f"\nDiscovered relationship types ({len(rel_types)}):")
    for rt in rel_types:
        print(f"  - Type: {rt.get('type')}")
        print(f"    From: {rt.get('from_node')} -> To: {rt.get('to_node')}")
        print(f"    Properties: {rt.get('properties', [])}")

    return rel_types


def test_add_to_schema(ctx: DummyToolContext, entity_types: list, rel_types: list,
                       source_columns: list) -> None:
    """Test adding text entities and relationships to schema."""
    print_section("5. Add Text Entities/Relationships to Schema")

    # First create a base Respondent node (required for relationships)
    propose_node_type(
        label="Respondent",
        unique_property="respondent_id",
        properties=["name"],
        entity_type="Respondent",
        extraction_hints={
            "source_type": "entity_selection",
            "column_pattern": "respondent_id",
        },
        tool_context=ctx
    )

    # Add each entity type
    for et in entity_types[:2]:  # Limit to first 2 for testing
        result = add_text_entity_to_schema(
            entity_type=et,
            description=f"Entity extracted from text feedback: {et}",
            source_columns=source_columns,
            tool_context=ctx
        )

        if result.get("status") == "error":
            print(f"ERROR adding {et}: {result.get('error_message')}")
        else:
            added = result.get("text_entity_added", {})
            print(f"Added entity: {added.get('entity_type')}")
            print(f"  Unique property: {added.get('unique_property')}")

    # Add relationships
    for rt in rel_types[:2]:  # Limit to first 2
        rel_type = rt.get("type", "MENTIONS")
        from_node = rt.get("from_node", "Respondent")
        to_node = rt.get("to_node", entity_types[0] if entity_types else "Feature")

        # Skip if to_node not in schema
        if to_node not in entity_types[:2] and to_node != "Respondent":
            continue

        result = add_text_relationship_to_schema(
            relationship_type=rel_type,
            from_node=from_node,
            to_node=to_node,
            properties=rt.get("properties", ["sentiment"]),
            source_columns=source_columns,
            sentiment="positive",
            tool_context=ctx
        )

        if result.get("status") == "error":
            print(f"ERROR adding {rel_type}: {result.get('error_message')}")
        else:
            added = result.get("text_relationship_added", {})
            print(f"Added relationship: {added.get('relationship_type')}")
            print(f"  {added.get('from_node')} -> {added.get('to_node')}")

    # Show final schema
    schema = get_target_schema(tool_context=ctx)
    print(f"\nSchema summary: {schema.get('target_schema', {}).get('summary', 'N/A')}")


def test_extract_from_text(ctx: DummyToolContext, column_name: str, entity_type: str) -> None:
    """Test extracting entity instances from text."""
    print_section("6. Extract Entities from Text Column (LLM)")

    # Approve schema first
    approve_target_schema(tool_context=ctx)

    result = extract_entities_from_text_column(
        file_path=DATA_FILE,
        node_label=entity_type,
        column_name=column_name,
        tool_context=ctx,
        batch_size=5  # Small batch for testing
    )

    if result.get("status") == "error":
        print(f"ERROR: {result.get('error_message')}")
        return

    extracted = result.get("extracted_entities", {})
    print(f"Node label: {extracted.get('node_label')}")
    print(f"Column: {extracted.get('column_name')}")
    print(f"Count: {extracted.get('count')}")

    samples = extracted.get("sample", [])
    print(f"\nSample entities ({len(samples)}):")
    for e in samples[:5]:
        print(f"  - {e.get('name')} (mentions: {e.get('mention_count', 1)})")


def test_deduplication(ctx: DummyToolContext, entity_type: str) -> None:
    """Test LLM-based entity deduplication."""
    print_section("7. Deduplicate Entities (LLM)")

    result = deduplicate_entities_with_llm(
        node_label=entity_type,
        tool_context=ctx
    )

    if result.get("status") == "error":
        print(f"ERROR: {result.get('error_message')}")
        return

    dedup = result.get("deduplication", {})
    print(f"Original count: {dedup.get('original_count')}")
    print(f"Deduplicated count: {dedup.get('deduplicated_count')}")
    print(f"Merged groups: {dedup.get('merged_groups')}")

    sample_merges = dedup.get("sample_merges", [])
    if sample_merges:
        print("\nSample merges:")
        for m in sample_merges[:3]:
            print(f"  - {m.get('canonical')} <- {m.get('aliases', [])}")


def test_text_analysis_summary(ctx: DummyToolContext) -> None:
    """Test getting text analysis summary."""
    print_section("8. Text Analysis Summary")

    result = get_text_analysis_summary(tool_context=ctx)

    summary = result.get("text_analysis_summary", {})
    print(f"Text columns analyzed: {summary.get('text_columns_count', 0)}")
    print(f"Entity types discovered: {summary.get('entity_types_count', 0)}")
    print(f"Relationship types discovered: {summary.get('relationship_types_count', 0)}")


def main() -> None:
    """Run all text extraction tests."""
    print("\n" + "=" * 60)
    print("  TEXT FEEDBACK COLUMN EXTRACTION TEST")
    print("=" * 60)

    ctx = DummyToolContext()
    ctx.state["approved_files"] = [DATA_FILE]

    # 1. Identify text columns
    text_columns = test_identify_text_columns(ctx)

    if not text_columns:
        print("\nNo text columns found. Exiting.")
        return

    # Pick a positive feedback column for testing
    test_col = None
    for col in text_columns:
        if "positive" in col["column_name"].lower() and col["fill_rate"] > 10:
            test_col = col
            break

    if not test_col:
        test_col = text_columns[0]

    column_name = test_col["column_name"]
    category = test_col["category"]
    print(f"\n>>> Using column for testing: {column_name}")

    # 2. Sample the column
    samples = test_sample_text_column(ctx, column_name)

    if not samples:
        print("\nNo samples found. Exiting.")
        return

    # 3. Analyze for entity types
    entity_types = test_analyze_entities(ctx, column_name, samples, category)

    if not entity_types:
        print("\nNo entity types found. Using default 'Feature'.")
        entity_types = ["Feature"]

    # 4. Analyze for relationship types
    rel_types = test_analyze_relationships(ctx, column_name, samples, entity_types, category)

    # 5. Add to schema
    test_add_to_schema(ctx, entity_types, rel_types, [column_name])

    # 6. Extract entity instances
    test_extract_from_text(ctx, column_name, entity_types[0])

    # 7. Deduplicate
    test_deduplication(ctx, entity_types[0])

    # 8. Summary
    test_text_analysis_summary(ctx)

    print("\n" + "=" * 60)
    print("  TEST COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()
