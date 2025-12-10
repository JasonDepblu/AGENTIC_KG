"""
Schema-Preprocess Coordinator Agent for Agentic KG.

This "super coordinator" manages the bidirectional flow between Schema Design
and Targeted Preprocessing phases, supporting automatic rollback when preprocessing
discovers schema issues.

Architecture:
```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Schema-Preprocess Coordinator (LoopAgent)                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Iteration 1:                                                                │
│  ┌──────────────────┐     ┌──────────────────────┐     ┌────────────────┐   │
│  │ Schema Design    │ ──▶ │ Targeted Preprocessing│ ──▶ │ Status Checker │   │
│  │ Loop             │     │ Loop                  │     │                │   │
│  └──────────────────┘     └──────────────────────┘     └───────┬────────┘   │
│                                                                 │            │
│                           ┌─────────────────────────────────────┤            │
│                           │                                     │            │
│                     needs_rollback?                        complete?         │
│                           │                                     │            │
│                           ▼                                     ▼            │
│  Iteration 2:        Continue Loop                         Escalate         │
│  (Schema Revision)        │                                (Exit Loop)       │
│                           ▼                                                  │
│  ┌──────────────────┐     │                                                  │
│  │ Schema Design    │ ◀───┘                                                  │
│  │ (with rollback   │                                                        │
│  │  reason)         │                                                        │
│  └──────────────────┘                                                        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```
"""

from typing import AsyncGenerator

from google.adk.agents import LoopAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.events import Event, EventActions

from ..llm import get_adk_llm
from .schema_design_agent import create_schema_design_loop
from .targeted_preprocessing_agent import create_targeted_preprocessing_loop
from ..tools.targeted_preprocessing import (
    NEEDS_SCHEMA_REVISION,
    TARGETED_PREPROCESSING_COMPLETE,
)
from ..models.target_schema import APPROVED_TARGET_SCHEMA_KEY

# Flag to track if schema design phase has been completed in this iteration
SCHEMA_DESIGN_PHASE_DONE = "schema_design_phase_done"


def log_agent(callback_context: CallbackContext) -> None:
    """Log agent entry for debugging."""
    print(f"\n### Entering Agent: {callback_context.agent_name}")


def set_coordinator_mode(callback_context: CallbackContext) -> None:
    """Set coordinator mode flag so sub-agents know not to escalate.

    This is critical because escalate signals propagate to the root
    and would exit the entire coordinator if used by sub-agents.
    """
    print(f"\n### Entering Coordinator Agent: {callback_context.agent_name}")
    # Set flag to tell sub-agents we're in coordinator mode
    callback_context.state["running_in_coordinator"] = True


class CheckCoordinatorStatus(BaseAgent):
    """
    Status checker for the Schema-Preprocess Coordinator.

    Determines whether to:
    1. Continue the loop (rollback to schema design)
    2. Escalate/exit (preprocessing complete)

    Decision logic:
    - If needs_schema_revision=True → Continue loop (don't escalate)
    - If preprocessing_complete=True → Escalate (exit loop)
    - If schema_design_phase_done=True but preprocessing not started → Wait for user (escalate)
    - Otherwise → Continue to next sub-agent
    """

    async def _run_async_impl(
        self,
        ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        """Check status and decide whether to continue or exit the loop."""

        # Check if preprocessing is complete
        preprocessing_complete = ctx.session.state.get(TARGETED_PREPROCESSING_COMPLETE, False)

        # Check if schema revision is needed (rollback requested)
        needs_rollback = ctx.session.state.get(NEEDS_SCHEMA_REVISION, False)

        # Check if schema design phase is done (schema approved, waiting for preprocessing)
        schema_phase_done = ctx.session.state.get(SCHEMA_DESIGN_PHASE_DONE, False)

        if preprocessing_complete and not needs_rollback:
            # Success! Preprocessing is done, exit the coordinator loop
            print(f"\n### {self.name}: Preprocessing complete, escalating to exit coordinator")
            yield Event(
                author=self.name,
                actions=EventActions(escalate=True)
            )
        elif needs_rollback:
            # Rollback requested - continue the loop to re-run schema design
            print(f"\n### {self.name}: Rollback requested, continuing loop for schema revision")

            # ADK State doesn't support __delitem__ or reassignment with items()
            # Clear keys by setting to None
            keys_to_clear = [
                APPROVED_TARGET_SCHEMA_KEY,
                SCHEMA_DESIGN_PHASE_DONE,
                TARGETED_PREPROCESSING_COMPLETE,
                "targeted_extraction_results",
                "targeted_entity_maps",
                "targeted_relationship_data",
                "generated_files",
                "preprocessing_feedback",
            ]
            for key in keys_to_clear:
                ctx.session.state[key] = None
            # Clear rollback flag but keep revision_reason if present
            ctx.session.state[NEEDS_SCHEMA_REVISION] = False

            # Don't escalate - continue the loop
            yield Event(
                author=self.name,
                actions=EventActions(escalate=False)
            )
        elif schema_phase_done and not preprocessing_complete:
            # Schema is approved and ready for preprocessing
            # Don't loop back - just wait/escalate to let preprocessing continue
            print(f"\n### {self.name}: Schema design done, preprocessing in progress or pending")
            yield Event(
                author=self.name,
                actions=EventActions(escalate=False)
            )
        else:
            # Neither complete nor rollback - this shouldn't normally happen
            # but if it does, don't escalate to continue the loop
            print(f"\n### {self.name}: Status check - not complete, not rollback, continuing")
            yield Event(
                author=self.name,
                actions=EventActions(escalate=False)
            )


class CheckSchemaApproved(BaseAgent):
    """
    Checks if schema was approved before moving to preprocessing.

    Sets a flag to indicate whether preprocessing should run.
    Does NOT escalate because escalate signals propagate to root.

    IMPORTANT: Only APPROVED_TARGET_SCHEMA_KEY matters, not schema_design_phase_done.
    The phase_done flag means the critic said "valid", but user must still explicitly
    approve before preprocessing can begin.
    """

    async def _run_async_impl(
        self,
        ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        """Check if schema is approved and set skip flag if not."""

        # ONLY check for explicit user approval via approve_target_schema tool
        # Do NOT use schema_design_phase_done - that just means critic said "valid"
        # but user hasn't approved yet
        # Note: Use get() is not None because we clear keys by setting to None
        schema_approved = ctx.session.state.get(APPROVED_TARGET_SCHEMA_KEY) is not None

        if schema_approved:
            print(f"\n### {self.name}: Schema explicitly approved (APPROVED_TARGET_SCHEMA_KEY exists), proceeding to preprocessing")
            # Clear skip flag to allow preprocessing
            ctx.session.state["skip_preprocessing"] = False
        else:
            # Schema not approved yet - set flag to skip preprocessing this iteration
            # This prevents the infinite retry loop
            print(f"\n### {self.name}: Schema NOT approved yet (waiting for user to call approve_target_schema), skipping preprocessing")
            ctx.session.state["skip_preprocessing"] = True

        # Never escalate - the coordinator loop will continue
        yield Event(
            author=self.name,
            actions=EventActions(escalate=False)
        )


def create_schema_preprocess_coordinator(
    llm=None,
    max_iterations: int = 10,  # Increased to accommodate inner loop iterations
    name: str = "schema_preprocess_coordinator"
) -> LoopAgent:
    """
    Create a Schema-Preprocess Coordinator Agent.

    This coordinator manages the bidirectional flow between Schema Design
    and Targeted Preprocessing, supporting automatic rollback when issues
    are discovered during preprocessing.

    Flow:
    1. Schema Design Loop runs until schema is approved
    2. Targeted Preprocessing Loop extracts data based on schema
    3. If preprocessing finds issues → request_schema_revision()
    4. Coordinator detects rollback flag and restarts with Schema Design
    5. Repeat until preprocessing completes successfully

    Args:
        llm: Optional LLM instance. If None, uses default from get_adk_llm()
        max_iterations: Maximum schema→preprocess→rollback cycles (default 3)
        name: Agent name for identification

    Returns:
        LoopAgent: Configured Schema-Preprocess Coordinator

    Example:
        ```python
        from src.agents import create_schema_preprocess_coordinator, make_agent_caller

        coordinator = create_schema_preprocess_coordinator()
        caller = await make_agent_caller(coordinator, {
            "approved_user_goal": {"kind_of_graph": "survey analysis"},
            "approved_files": ["survey.xlsx"]
        })

        # Start the process
        await caller.call("Design schema and extract data from survey")

        # If issues found, coordinator automatically rolls back
        # User may need to provide additional guidance
        await caller.call("The brand column is '品牌是?' not '品牌选择'")
        ```
    """
    if llm is None:
        llm = get_adk_llm()

    # Create sub-agents
    # Sub-loops need enough iterations to complete their work within one coordinator call
    # The stop checkers inside each loop will break out early when done
    # We set moderate iterations (3) to allow design->critic->fix cycles
    schema_loop = create_schema_design_loop(llm, max_iterations=3, name="schema_design_sub_loop")
    schema_checker = CheckSchemaApproved(name="schema_approval_checker")
    preprocess_loop = create_targeted_preprocessing_loop(llm, max_iterations=3, name="preprocessing_sub_loop")
    status_checker = CheckCoordinatorStatus(name="coordinator_status_checker")

    return LoopAgent(
        name=name,
        description=(
            "Coordinates Schema Design and Targeted Preprocessing with bidirectional "
            "feedback. Supports automatic rollback from preprocessing to schema design "
            "when data extraction issues are discovered."
        ),
        max_iterations=max_iterations,
        sub_agents=[
            schema_loop,      # 1. Design/revise schema
            schema_checker,   # 2. Check if schema approved (skip preprocess if not)
            preprocess_loop,  # 3. Extract data based on schema
            status_checker,   # 4. Check if complete or need rollback
        ],
        before_agent_callback=set_coordinator_mode,  # Set coordinator mode flag
    )


# Alternative: Sequential approach for simpler cases
def create_schema_preprocess_sequential(
    llm=None,
    name: str = "schema_preprocess_sequential"
):
    """
    Create a simpler sequential coordinator without rollback support.

    Use this when you don't need automatic rollback - the user can
    manually request schema changes if needed.

    Args:
        llm: Optional LLM instance
        name: Agent name

    Returns:
        SequentialAgent that runs schema design then preprocessing
    """
    from google.adk.agents import SequentialAgent

    if llm is None:
        llm = get_adk_llm()

    schema_loop = create_schema_design_loop(llm)
    preprocess_loop = create_targeted_preprocessing_loop(llm)

    return SequentialAgent(
        name=name,
        description="Runs schema design followed by targeted preprocessing",
        sub_agents=[schema_loop, preprocess_loop],
        before_agent_callback=log_agent,
    )
