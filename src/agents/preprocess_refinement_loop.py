"""
Preprocess Refinement Loop.

Implements iterative preprocessing with validation using LoopAgent pattern.
Each preprocessing phase (Column Classification, NER, Rating) is paired with
a critic agent for validation.
"""

from typing import AsyncGenerator

from google.adk.agents import LoopAgent, BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions

from ..llm import get_adk_llm
from .survey_column_classifier_agent import create_survey_column_classifier_agent
from .survey_ner_agent import create_survey_ner_agent
from .survey_rating_agent import create_survey_rating_agent
from .survey_open_text_agent import create_survey_open_text_agent
from .survey_preprocess_critic import create_survey_preprocess_critic_agent


class CheckPreprocessStatus(BaseAgent):
    """
    Checks the preprocess feedback status and decides whether to continue or retry.

    This agent is used as the final agent in a LoopAgent to control the iteration flow.

    Feedback states:
    - "valid": Validation passed, escalate to exit the loop
    - "retry": Validation failed, continue loop for re-processing
    """

    async def _run_async_impl(
        self,
        ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        """Check feedback and yield appropriate event."""
        # Get feedback from session state
        feedback = ctx.session.state.get("preprocess_feedback", "valid")

        # Determine if we should stop the loop
        should_stop = (feedback == "valid")

        # Yield event with escalate flag
        yield Event(
            author=self.name,
            actions=EventActions(escalate=should_stop)
        )


class CheckPipelineStatus(BaseAgent):
    """
    Checks overall pipeline status and handles Schema-to-Preprocess feedback.

    This agent handles the special "reprocess" feedback from Schema stage.

    Feedback states:
    - "valid": Pipeline complete, escalate to exit
    - "retry": Schema needs adjustment, continue schema loop
    - "reprocess:phase:reason": Need to go back to preprocess stage
    """

    async def _run_async_impl(
        self,
        ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        """Check pipeline status and handle reprocess requests."""
        feedback = ctx.session.state.get("feedback", "valid")

        if feedback == "valid":
            # Pipeline complete
            yield Event(
                author=self.name,
                actions=EventActions(escalate=True)
            )

        elif feedback.startswith("reprocess:"):
            # Need to go back to preprocess
            parts = feedback.split(":")
            phase = parts[1] if len(parts) > 1 else "all"
            reason = parts[2] if len(parts) > 2 else "unknown"

            # Store reprocess instructions
            ctx.session.state["reprocess_needed"] = True
            ctx.session.state["reprocess_instructions"] = {
                "phase": phase,
                "reason": reason,
                "attempt": ctx.session.state.get("reprocess_attempts", 0) + 1
            }

            # Clear the feedback to allow reprocessing
            ctx.session.state["feedback"] = "pending_reprocess"

            # Don't escalate - let the outer loop continue
            yield Event(
                author=self.name,
                actions=EventActions(escalate=False)
            )

        else:
            # Normal retry for schema refinement
            yield Event(
                author=self.name,
                actions=EventActions(escalate=False)
            )


def create_column_classification_loop(
    llm=None,
    max_iterations: int = 2,
    name: str = "column_classification_loop"
) -> LoopAgent:
    """
    Creates a loop for column classification with validation.

    Args:
        llm: LLM instance
        max_iterations: Maximum number of retry attempts
        name: Loop name

    Returns:
        LoopAgent: Column classification refinement loop
    """
    if llm is None:
        llm = get_adk_llm()

    classifier = create_survey_column_classifier_agent(llm=llm)
    critic = create_survey_preprocess_critic_agent(llm=llm, name="column_classifier_critic")
    stop_checker = CheckPreprocessStatus(name="column_classification_stop_checker")

    return LoopAgent(
        name=name,
        description="Classifies survey columns with validation and iterative refinement",
        max_iterations=max_iterations,
        sub_agents=[classifier, critic, stop_checker],
    )


def create_ner_loop(
    llm=None,
    max_iterations: int = 2,
    name: str = "ner_loop"
) -> LoopAgent:
    """
    Creates a loop for entity extraction with validation.

    Args:
        llm: LLM instance
        max_iterations: Maximum number of retry attempts
        name: Loop name

    Returns:
        LoopAgent: NER refinement loop
    """
    if llm is None:
        llm = get_adk_llm()

    ner_agent = create_survey_ner_agent(llm=llm)
    critic = create_survey_preprocess_critic_agent(llm=llm, name="ner_critic")
    stop_checker = CheckPreprocessStatus(name="ner_stop_checker")

    return LoopAgent(
        name=name,
        description="Extracts entities with validation and iterative refinement",
        max_iterations=max_iterations,
        sub_agents=[ner_agent, critic, stop_checker],
    )


def create_rating_loop(
    llm=None,
    max_iterations: int = 2,
    name: str = "rating_loop"
) -> LoopAgent:
    """
    Creates a loop for rating extraction with validation.

    Args:
        llm: LLM instance
        max_iterations: Maximum number of retry attempts
        name: Loop name

    Returns:
        LoopAgent: Rating extraction refinement loop
    """
    if llm is None:
        llm = get_adk_llm()

    rating_agent = create_survey_rating_agent(llm=llm)
    critic = create_survey_preprocess_critic_agent(llm=llm, name="rating_critic")
    stop_checker = CheckPreprocessStatus(name="rating_stop_checker")

    return LoopAgent(
        name=name,
        description="Extracts ratings with validation and iterative refinement",
        max_iterations=max_iterations,
        sub_agents=[rating_agent, critic, stop_checker],
    )


def create_preprocess_refinement_coordinator(
    llm=None,
    max_iterations_per_phase: int = 2,
    include_open_text: bool = True,
    name: str = "preprocess_refinement_coordinator"
) -> LoopAgent:
    """
    Creates a complete preprocessing coordinator with validation loops.

    This coordinator orchestrates:
    1. Column Classification Loop (with critic)
    2. NER Loop (with critic)
    3. Rating Loop (with critic)
    4. Open Text Agent (optional, no loop - LLM-based extraction)

    The outer LoopAgent allows the entire preprocessing to be re-run if
    Schema stage requests it.

    Args:
        llm: LLM instance
        max_iterations_per_phase: Max retries for each phase
        include_open_text: Whether to include open text analysis
        name: Coordinator name

    Returns:
        LoopAgent: Complete preprocessing coordinator with validation
    """
    if llm is None:
        llm = get_adk_llm()

    sub_agents = [
        create_column_classification_loop(llm, max_iterations_per_phase),
        create_ner_loop(llm, max_iterations_per_phase),
        create_rating_loop(llm, max_iterations_per_phase),
    ]

    if include_open_text:
        # Open text agent doesn't need a loop - it's LLM-based extraction
        open_text_agent = create_survey_open_text_agent(llm=llm)
        sub_agents.append(open_text_agent)

    # Final status checker for the preprocessing stage
    preprocess_status_checker = CheckPreprocessStatus(name="preprocess_final_status")
    sub_agents.append(preprocess_status_checker)

    return LoopAgent(
        name=name,
        description="Complete preprocessing pipeline with validation loops for each phase",
        max_iterations=1,  # Single pass through all phases
        sub_agents=sub_agents,
    )


def create_complete_pipeline_coordinator(
    llm=None,
    max_schema_iterations: int = 2,
    max_reprocess_attempts: int = 2,
    include_open_text: bool = True,
    name: str = "complete_pipeline_coordinator"
) -> LoopAgent:
    """
    Creates the complete KG construction pipeline with bidirectional feedback.

    Architecture:
    ```
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                     Complete Pipeline Coordinator                        │
    │                                                                          │
    │  ┌─────────────────────────────────────────────────────────────────┐    │
    │  │  Preprocess Refinement Coordinator                               │    │
    │  │  ├─ Column Classification Loop (Classifier + Critic)             │    │
    │  │  ├─ NER Loop (NER Agent + Critic)                               │    │
    │  │  ├─ Rating Loop (Rating Agent + Critic)                         │    │
    │  │  └─ Open Text Agent (optional)                                  │    │
    │  └─────────────────────────────────────────────────────────────────┘    │
    │                                    ↓                                     │
    │  ┌─────────────────────────────────────────────────────────────────┐    │
    │  │  Schema Refinement Loop                                          │    │
    │  │  ├─ Schema Proposal Agent                                        │    │
    │  │  ├─ Schema Critic Agent (Enhanced with reprocess support)        │    │
    │  │  └─ Check Status (can trigger reprocess)                         │    │
    │  └─────────────────────────────────────────────────────────────────┘    │
    │                                    ↓                                     │
    │  ┌─────────────────────────────────────────────────────────────────┐    │
    │  │  Pipeline Status Checker                                         │    │
    │  │  - "valid" → Complete                                            │    │
    │  │  - "retry" → Continue schema refinement                          │    │
    │  │  - "reprocess:phase:reason" → Go back to preprocessing          │    │
    │  └─────────────────────────────────────────────────────────────────┘    │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘
    ```

    Args:
        llm: LLM instance
        max_schema_iterations: Max iterations for schema refinement
        max_reprocess_attempts: Max times to allow reprocessing from schema feedback
        include_open_text: Whether to include open text analysis
        name: Coordinator name

    Returns:
        LoopAgent: Complete pipeline coordinator
    """
    if llm is None:
        llm = get_adk_llm()

    # Import schema agents here to avoid circular imports
    from .schema_proposal_agent import create_schema_refinement_loop

    # Create sub-components
    preprocess_coordinator = create_preprocess_refinement_coordinator(
        llm=llm,
        include_open_text=include_open_text
    )

    schema_loop = create_schema_refinement_loop(
        llm=llm,
        max_iterations=max_schema_iterations
    )

    pipeline_checker = CheckPipelineStatus(name="pipeline_status_checker")

    return LoopAgent(
        name=name,
        description="Complete KG construction pipeline with preprocess-schema feedback loop",
        max_iterations=max_reprocess_attempts + 1,
        sub_agents=[
            preprocess_coordinator,
            schema_loop,
            pipeline_checker
        ],
    )


# Export for convenience
__all__ = [
    "CheckPreprocessStatus",
    "CheckPipelineStatus",
    "create_column_classification_loop",
    "create_ner_loop",
    "create_rating_loop",
    "create_preprocess_refinement_coordinator",
    "create_complete_pipeline_coordinator",
]
