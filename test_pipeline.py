"""
Test script for Agentic KG pipeline.

Tests the full pipeline flow using data/data_process.csv as the data source.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.llm import get_adk_llm
from src.agents.base import make_agent_caller
from src.agents.user_intent_agent import create_user_intent_agent
from src.agents.file_suggestion_agent import create_file_suggestion_agent
from src.agents.schema_proposal_agent import create_schema_refinement_loop
from src.tools.user_intent import APPROVED_USER_GOAL, PERCEIVED_USER_GOAL
from src.tools.kg_construction import APPROVED_CONSTRUCTION_PLAN


async def test_user_intent_agent():
    """Test the user intent agent flow."""
    print("\n" + "=" * 60)
    print("Testing User Intent Agent")
    print("=" * 60)

    llm = get_adk_llm()
    agent = create_user_intent_agent(llm)

    # Initial state
    initial_state = {}

    caller = await make_agent_caller(agent, initial_state)

    # Step 1: User describes their goal
    print("\n[Step 1] User: 我想构建一个车企用户关注度知识图谱，分析不同车企的用户对各种属性的关注程度")
    response = await caller.call("我想构建一个车企用户关注度知识图谱，分析不同车企的用户对各种属性的关注程度")
    print(f"Agent Response: {response}")

    # Check state
    session = await caller.get_session()
    print(f"\nState after Step 1:")
    print(f"  - perceived_user_goal: {PERCEIVED_USER_GOAL in session.state}")
    print(f"  - approved_user_goal: {APPROVED_USER_GOAL in session.state}")
    if PERCEIVED_USER_GOAL in session.state:
        print(f"  - Goal content: {session.state[PERCEIVED_USER_GOAL]}")

    # Step 2: User approves
    print("\n[Step 2] User: 是的，按照这个目标构建")
    response = await caller.call("是的，按照这个目标构建")
    print(f"Agent Response: {response}")

    # Check state
    session = await caller.get_session()
    print(f"\nState after Step 2:")
    print(f"  - perceived_user_goal: {PERCEIVED_USER_GOAL in session.state}")
    print(f"  - approved_user_goal: {APPROVED_USER_GOAL in session.state}")
    if APPROVED_USER_GOAL in session.state:
        print(f"  - Approved goal: {session.state[APPROVED_USER_GOAL]}")
        return session.state
    else:
        print("\nERROR: User goal was not approved!")
        return None


async def test_file_suggestion_agent(state: dict):
    """Test the file suggestion agent flow."""
    print("\n" + "=" * 60)
    print("Testing File Suggestion Agent")
    print("=" * 60)

    llm = get_adk_llm()
    agent = create_file_suggestion_agent(llm)

    caller = await make_agent_caller(agent, state)

    # Ask for file suggestions
    print("\n[Step 1] Asking for file suggestions...")
    response = await caller.call("What files can we use for import?")
    print(f"Agent Response: {response}")

    # Check state
    session = await caller.get_session()
    print(f"\nState - approved_files: {'approved_files' in session.state}")

    # Approve files
    print("\n[Step 2] User: Approve those files")
    response = await caller.call("Approve those files")
    print(f"Agent Response: {response}")

    session = await caller.get_session()
    if "approved_files" in session.state:
        print(f"  - Approved files: {session.state['approved_files']}")
        return session.state
    else:
        print("\nERROR: Files were not approved!")
        return None


async def test_schema_proposal_agent(state: dict):
    """Test the schema proposal agent flow."""
    print("\n" + "=" * 60)
    print("Testing Schema Proposal Agent")
    print("=" * 60)

    llm = get_adk_llm()

    # Add empty feedback
    state["feedback"] = ""

    loop = create_schema_refinement_loop(llm)

    caller = await make_agent_caller(loop, state)

    # Ask for schema proposal
    print("\n[Step 1] Asking for schema proposal...")
    response = await caller.call("How can these files be imported?")
    print(f"Agent Response: {response[:500]}..." if len(response) > 500 else f"Agent Response: {response}")

    # Check state
    session = await caller.get_session()
    print(f"\nState - proposed_construction_plan: {'proposed_construction_plan' in session.state}")
    print(f"State - approved_construction_plan: {APPROVED_CONSTRUCTION_PLAN in session.state}")

    if "proposed_construction_plan" in session.state:
        return session.state
    return None


async def test_llm_basic():
    """Basic LLM connectivity test."""
    print("\n" + "=" * 60)
    print("Testing LLM Basic Connectivity")
    print("=" * 60)

    import litellm
    from src.config import get_config

    config = get_config()

    print(f"LLM Model: {config.llm.default_model}")
    print(f"API Base: {config.llm.api_base}")

    # Try a simple completion using litellm directly
    try:
        response = litellm.completion(
            model=f"openai/{config.llm.default_model}",
            messages=[{"role": "user", "content": "Say 'Hello, test successful!'"}],
            api_key=config.llm.api_key,
            api_base=config.llm.api_base,
        )
        print(f"LLM Response: {response.choices[0].message.content}")
        return True
    except Exception as e:
        print(f"LLM Error: {type(e).__name__}: {e}")
        return False


async def main():
    """Run all tests."""
    print("=" * 60)
    print("Agentic KG Pipeline Test")
    print("=" * 60)
    print(f"Data file: data/data_process.csv")

    # Test 1: Basic LLM connectivity
    llm_ok = await test_llm_basic()
    if not llm_ok:
        print("\n❌ LLM connectivity failed. Check your API credentials.")
        return
    print("\n✅ LLM connectivity OK")

    # Test 2: User Intent Agent
    try:
        state = await test_user_intent_agent()
        if state and APPROVED_USER_GOAL in state:
            print("\n✅ User Intent Agent: PASSED")
        else:
            print("\n❌ User Intent Agent: FAILED - Goal not approved")
            return
    except Exception as e:
        print(f"\n❌ User Intent Agent: ERROR - {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return

    # Test 3: File Suggestion Agent
    try:
        state = await test_file_suggestion_agent(state)
        if state and "approved_files" in state:
            print("\n✅ File Suggestion Agent: PASSED")
        else:
            print("\n❌ File Suggestion Agent: FAILED - Files not approved")
            return
    except Exception as e:
        print(f"\n❌ File Suggestion Agent: ERROR - {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return

    # Test 4: Schema Proposal Agent
    try:
        state = await test_schema_proposal_agent(state)
        if state and "proposed_construction_plan" in state:
            print("\n✅ Schema Proposal Agent: PASSED")
        else:
            print("\n❌ Schema Proposal Agent: FAILED - No proposal generated")
            return
    except Exception as e:
        print(f"\n❌ Schema Proposal Agent: ERROR - {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n" + "=" * 60)
    print("All Tests Completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
