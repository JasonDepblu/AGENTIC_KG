#!/usr/bin/env python3
"""
Agentic KG - Knowledge Graph Construction with Google ADK

A multi-agent system for building knowledge graphs from structured and unstructured data.

Usage:
    python main.py [--interactive] [--verbose]
    python main.py --demo
    python main.py --test-connection
"""

import argparse
import asyncio
import sys

from src.config import get_config
from src.llm import get_adk_llm, test_llm_connection
from src.neo4j_client import get_graphdb
from src.pipelines import KGPipeline


def test_connections():
    """Test connections to LLM and Neo4j."""
    print("Testing connections...\n")

    # Test LLM
    print("1. Testing LLM connection...")
    try:
        llm = get_adk_llm()
        if test_llm_connection(llm):
            print("   LLM connection: OK\n")
        else:
            print("   LLM connection: FAILED\n")
            return False
    except Exception as e:
        print(f"   LLM connection: FAILED ({e})\n")
        return False

    # Test Neo4j
    print("2. Testing Neo4j connection...")
    try:
        graphdb = get_graphdb()
        result = graphdb.verify_connection()
        if result.get("status") == "success":
            print("   Neo4j connection: OK")
            print(f"   Message: {result.get('query_result', [{}])[0].get('message', 'Connected')}\n")
        else:
            print(f"   Neo4j connection: FAILED ({result.get('error_message')})\n")
            return False
    except Exception as e:
        print(f"   Neo4j connection: FAILED ({e})\n")
        return False

    print("All connections successful!")
    return True


async def run_demo():
    """Run a demonstration of the pipeline."""
    print("\n" + "=" * 60)
    print("DEMO: Knowledge Graph Construction Pipeline")
    print("=" * 60)

    # Example construction plan for demo
    demo_construction_plan = {
        "Product": {
            "construction_type": "node",
            "source_file": "products.csv",
            "label": "Product",
            "unique_column_name": "product_id",
            "properties": ["product_name", "price", "description"]
        },
        "Supplier": {
            "construction_type": "node",
            "source_file": "suppliers.csv",
            "label": "Supplier",
            "unique_column_name": "supplier_id",
            "properties": ["name", "specialty", "city", "country"]
        },
    }

    print("\nDemo Construction Plan:")
    for name, rule in demo_construction_plan.items():
        print(f"  - {name}: {rule['construction_type']} from {rule['source_file']}")

    print("\nTo run the full pipeline, use --interactive mode.")
    print("Example: python main.py --interactive")


async def run_interactive(verbose: bool = False):
    """Run the pipeline in interactive mode."""
    print("\n" + "=" * 60)
    print("INTERACTIVE MODE: Knowledge Graph Construction")
    print("=" * 60)

    # Get user goal
    print("\nDescribe the knowledge graph you want to build:")
    print("(Example: 'I want a supply chain graph for bill of materials analysis')")
    user_goal = input("\n> ").strip()

    if not user_goal:
        print("No goal provided. Exiting.")
        return

    # Create and run pipeline
    pipeline = KGPipeline(verbose=verbose)

    try:
        # Run each phase with user confirmation
        print("\n--- Phase 1: Capturing User Intent ---")
        await pipeline.run_user_intent_phase(user_goal, "Approve that goal")

        print(f"\nApproved Goal: {pipeline.state.get('approved_user_goal')}")
        input("\nPress Enter to continue to file suggestion...")

        print("\n--- Phase 2: Suggesting Files ---")
        await pipeline.run_file_suggestion_phase("Yes, use those files")

        print(f"\nApproved Files: {pipeline.state.get('approved_files')}")
        input("\nPress Enter to continue to schema proposal...")

        print("\n--- Phase 3: Proposing Schema ---")
        await pipeline.run_schema_proposal_phase("Approve the schema")

        print(f"\nConstruction Plan Keys: {list(pipeline.state.get('approved_construction_plan', {}).keys())}")

        proceed = input("\nProceed with construction? (y/n): ").strip().lower()
        if proceed != 'y':
            print("Construction cancelled.")
            return

        print("\n--- Phase 4: Constructing Knowledge Graph ---")
        result = await pipeline.run_construction_phase()

        print("\n" + "=" * 60)
        print("PIPELINE COMPLETE!")
        print("=" * 60)
        print(f"\nConstruction Result: {result}")

    except Exception as e:
        print(f"\nError during pipeline execution: {e}")
        raise


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Agentic KG - Knowledge Graph Construction with Google ADK"
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run in interactive mode"
    )
    parser.add_argument(
        "--demo", "-d",
        action="store_true",
        help="Run demonstration"
    )
    parser.add_argument(
        "--test-connection", "-t",
        action="store_true",
        help="Test LLM and Neo4j connections"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )

    args = parser.parse_args()

    # Validate configuration
    try:
        config = get_config()
        config.validate()
    except ValueError as e:
        print(f"Configuration error: {e}")
        print("\nPlease ensure your .env file has the required settings:")
        print("  - DASHSCOPE_API_KEY")
        print("  - NEO4J_URI")
        print("  - NEO4J_PASSWORD")
        sys.exit(1)

    if args.test_connection:
        success = test_connections()
        sys.exit(0 if success else 1)

    if args.demo:
        await run_demo()
        return

    if args.interactive:
        await run_interactive(verbose=args.verbose)
        return

    # Default: show help
    parser.print_help()
    print("\n" + "=" * 60)
    print("Quick Start:")
    print("  1. python main.py --test-connection  # Verify setup")
    print("  2. python main.py --demo             # See demonstration")
    print("  3. python main.py --interactive      # Run full pipeline")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
