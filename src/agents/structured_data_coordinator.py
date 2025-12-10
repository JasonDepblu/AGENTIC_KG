"""
Structured Data Coordinator Agent for Agentic KG.

This coordinator manages the complete structured data pipeline flow:
Files → Data Cleaning → Schema Proposal → Preprocessing → Construction Plan

Based on the reference architecture from images/entire_solution.png and
reference/schema_proposal_structured.ipynb.

Architecture:
```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Structured Data Coordinator (SequentialAgent)            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Phase 1: Data Cleaning                                                      │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ Data Cleaning Loop (proposal + critic + check)                        │   │
│  │ - Analyze file quality                                                │   │
│  │ - Remove meaningless columns/rows                                     │   │
│  │ - Detect column types                                                 │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                      │                                       │
│                                      ▼                                       │
│  Phase 2: Schema Proposal                                                    │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ Schema Refinement Loop (proposal + critic + check)                    │   │
│  │ - Propose node and relationship types                                 │   │
│  │ - Validate with critic                                                │   │
│  │ - Iterate until valid                                                 │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                      │                                       │
│                                      ▼                                       │
│  Phase 3: Targeted Preprocessing                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ Preprocessing Loop (with rollback to Phase 2 if needed)               │   │
│  │ - Extract entities based on schema                                    │   │
│  │ - Generate entity/relationship files                                  │   │
│  │ - Request schema revision if issues found                             │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                      │                                       │
│                                      ▼                                       │
│  Phase 4: Construction Plan                                                  │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ Generate construction rules from approved schema and extracted data   │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```
"""

from typing import AsyncGenerator

from google.adk.agents import SequentialAgent, LoopAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.events import Event, EventActions

from ..llm import get_adk_llm
from .data_cleaning_agent import create_data_cleaning_loop
from .schema_proposal_agent import create_schema_refinement_loop
from .schema_design_agent import create_schema_design_loop
from .targeted_preprocessing_agent import create_targeted_preprocessing_loop
from ..tools.data_cleaning import DATA_CLEANING_COMPLETE_KEY
from ..tools.kg_construction import APPROVED_CONSTRUCTION_PLAN
from ..models.target_schema import APPROVED_TARGET_SCHEMA_KEY
from ..tools.targeted_preprocessing import (
    TARGETED_PREPROCESSING_COMPLETE,
    NEEDS_SCHEMA_REVISION,
)


def log_agent(callback_context: CallbackContext) -> None:
    """Log agent entry for debugging."""
    print(f"\n### Entering Agent: {callback_context.agent_name}")


def set_coordinator_mode(callback_context: CallbackContext) -> None:
    """Set coordinator mode flag so sub-agents know not to escalate."""
    print(f"\n### Entering Coordinator Agent: {callback_context.agent_name}")
    callback_context.state["running_in_coordinator"] = True


class CheckDataCleaningComplete(BaseAgent):
    """
    Checks if data cleaning phase is complete before proceeding.

    If data cleaning is not complete, yields an event asking the user
    to complete data cleaning first.
    """

    async def _run_async_impl(
        self,
        ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        """Check if data cleaning is complete."""
        cleaning_complete = ctx.session.state.get(DATA_CLEANING_COMPLETE_KEY, False)

        if cleaning_complete:
            print(f"\n### {self.name}: Data cleaning complete, proceeding to schema design")
            yield Event(
                author=self.name,
                actions=EventActions(escalate=False)
            )
        else:
            print(f"\n### {self.name}: Data cleaning not complete, waiting...")
            yield Event(
                author=self.name,
                actions=EventActions(escalate=False)
            )


class CheckSchemaApprovedForPreprocessing(BaseAgent):
    """
    Checks if schema is approved before proceeding to preprocessing.
    """

    async def _run_async_impl(
        self,
        ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        """Check if schema is approved."""
        # Note: Use get() is not None because we clear keys by setting to None
        schema_approved = ctx.session.state.get(APPROVED_TARGET_SCHEMA_KEY) is not None

        if schema_approved:
            print(f"\n### {self.name}: Schema approved, proceeding to preprocessing")
            ctx.session.state["skip_preprocessing"] = False
            yield Event(
                author=self.name,
                actions=EventActions(escalate=False)
            )
        else:
            print(f"\n### {self.name}: Schema not approved yet, skipping preprocessing")
            ctx.session.state["skip_preprocessing"] = True
            yield Event(
                author=self.name,
                actions=EventActions(escalate=False)
            )


class CheckPreprocessingAndRollback(BaseAgent):
    """
    Checks preprocessing status and handles rollback to schema design if needed.

    Decision logic:
    - If NEEDS_SCHEMA_REVISION is True → Clear schema and return to schema design
    - If TARGETED_PREPROCESSING_COMPLETE is True → Escalate to exit loop
    - Otherwise → Continue loop
    """

    async def _run_async_impl(
        self,
        ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        """Check status and handle rollback if needed."""

        needs_rollback = ctx.session.state.get(NEEDS_SCHEMA_REVISION, False)
        preprocessing_complete = ctx.session.state.get(TARGETED_PREPROCESSING_COMPLETE, False)

        if needs_rollback:
            print(f"\n### {self.name}: Rollback requested, clearing schema for revision")

            # ADK State doesn't support __delitem__ or reassignment with items()
            # Clear keys by setting to None
            keys_to_clear = [
                APPROVED_TARGET_SCHEMA_KEY,
                TARGETED_PREPROCESSING_COMPLETE,
                "targeted_extraction_results",
                "targeted_entity_maps",
                "targeted_relationship_data",
                "generated_files",
            ]
            for key in keys_to_clear:
                ctx.session.state[key] = None
            ctx.session.state[NEEDS_SCHEMA_REVISION] = False

            # Continue loop to re-run schema design
            yield Event(
                author=self.name,
                actions=EventActions(escalate=False)
            )
        elif preprocessing_complete:
            print(f"\n### {self.name}: Preprocessing complete, escalating to finish")
            yield Event(
                author=self.name,
                actions=EventActions(escalate=True)
            )
        else:
            print(f"\n### {self.name}: Not complete, not rollback, continuing")
            yield Event(
                author=self.name,
                actions=EventActions(escalate=False)
            )


def create_data_cleaning_phase(llm=None) -> LoopAgent:
    """
    Create the Data Cleaning phase.

    This phase analyzes and cleans raw data files:
    - Removes meaningless columns (empty, constant, unnamed)
    - Removes invalid rows (mostly empty, duplicates)
    - Detects column types for schema design

    Args:
        llm: Optional LLM instance

    Returns:
        LoopAgent for data cleaning phase
    """
    return create_data_cleaning_loop(llm, max_iterations=3, name="data_cleaning_phase")


def create_schema_and_preprocessing_loop(llm=None) -> LoopAgent:
    """
    Create a loop that handles schema design and preprocessing with rollback support.

    The loop contains:
    1. Schema Design Loop (or Schema Refinement Loop for legacy)
    2. Schema approval checker
    3. Targeted Preprocessing Loop
    4. Rollback checker

    If preprocessing finds issues, it requests rollback and the loop
    restarts with schema design.

    Args:
        llm: Optional LLM instance

    Returns:
        LoopAgent for schema + preprocessing with rollback
    """
    if llm is None:
        llm = get_adk_llm()

    # Schema design loop (uses target schema model)
    schema_loop = create_schema_design_loop(llm, max_iterations=3, name="schema_design_sub_loop")

    # Schema approval checker
    schema_checker = CheckSchemaApprovedForPreprocessing(name="schema_approval_checker")

    # Preprocessing loop
    preprocessing_loop = create_targeted_preprocessing_loop(llm, max_iterations=3, name="preprocessing_sub_loop")

    # Rollback checker
    rollback_checker = CheckPreprocessingAndRollback(name="rollback_checker")

    return LoopAgent(
        name="schema_preprocessing_loop",
        description=(
            "Iterates between schema design and preprocessing. "
            "Supports rollback from preprocessing to schema design when issues are found."
        ),
        max_iterations=5,  # Allow multiple schema revision cycles
        sub_agents=[
            schema_loop,
            schema_checker,
            preprocessing_loop,
            rollback_checker,
        ],
        before_agent_callback=log_agent,
    )


def create_structured_data_coordinator(
    llm=None,
    name: str = "structured_data_coordinator"
) -> SequentialAgent:
    """
    Create a Structured Data Coordinator Agent.

    This coordinator manages the complete structured data pipeline:
    1. Data Cleaning - Analyze and clean raw files
    2. Schema Design - Design target schema with node/relationship types
    3. Targeted Preprocessing - Extract entities based on schema
    4. Construction Plan - Generate construction rules

    The coordinator supports:
    - Iterative refinement with critic feedback at each phase
    - Rollback from preprocessing to schema design if issues found
    - Clean handoff between phases with state management

    Args:
        llm: Optional LLM instance. If None, uses default from get_adk_llm()
        name: Agent name for identification

    Returns:
        SequentialAgent: Configured Structured Data Coordinator

    Example:
        ```python
        from src.agents import create_structured_data_coordinator, make_agent_caller

        coordinator = create_structured_data_coordinator()
        caller = await make_agent_caller(coordinator, {
            "approved_user_goal": {"kind_of_graph": "survey analysis"},
            "approved_files": ["survey_data.xlsx"]
        })

        # Start the pipeline
        await caller.call("Process the survey data for knowledge graph")

        # Provide feedback as needed
        await caller.call("Keep the 品牌是? column, it contains brand selections")

        # Approve when ready
        await caller.call("Approve the schema")
        ```
    """
    if llm is None:
        llm = get_adk_llm()

    # Phase 1: Data Cleaning
    cleaning_phase = create_data_cleaning_phase(llm)
    cleaning_checker = CheckDataCleaningComplete(name="data_cleaning_checker")

    # Phase 2+3: Schema Design + Preprocessing (with rollback support)
    schema_preprocessing_loop = create_schema_and_preprocessing_loop(llm)

    return SequentialAgent(
        name=name,
        description=(
            "Coordinates the complete structured data pipeline: "
            "Data Cleaning → Schema Design → Preprocessing → Construction Plan. "
            "Supports iterative refinement and automatic rollback."
        ),
        sub_agents=[
            cleaning_phase,
            cleaning_checker,
            schema_preprocessing_loop,
        ],
        before_agent_callback=set_coordinator_mode,
    )


# Legacy compatibility: Create a simple version without data cleaning
def create_schema_only_coordinator(
    llm=None,
    name: str = "schema_only_coordinator"
) -> LoopAgent:
    """
    Create a coordinator that only handles Schema Design + Preprocessing.

    Use this when data cleaning is not needed or has already been done.
    This is similar to the original schema_preprocess_coordinator.

    Args:
        llm: Optional LLM instance
        name: Agent name

    Returns:
        LoopAgent for schema + preprocessing
    """
    return create_schema_and_preprocessing_loop(llm)
