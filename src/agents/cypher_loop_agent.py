"""
Cypher Generation Loop Agent.

Orchestrates the query generation → validation → refinement loop
using the Critic Agent pattern from the schema proposal module.
"""

from typing import AsyncGenerator, Optional

from google.adk.agents import BaseAgent, LoopAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions

from .cypher_generator_agent import create_cypher_generator_agent
from .cypher_validator_agent import create_cypher_validator_agent


# =============================================================================
# Loop Controller
# =============================================================================

class CheckCypherValidationStatus(BaseAgent):
    """
    Check validation status and decide whether to continue or stop the loop.

    This agent checks the cypher_feedback state key and yields an escalation
    event when validation passes (feedback == "valid").

    Based on the CheckStatusAndEscalate pattern from schema_proposal_agent.py.
    """

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        """
        Check feedback and yield escalation event if valid.

        Args:
            ctx: The invocation context containing session state

        Yields:
            Event with escalate=True if validation passed, False otherwise
        """
        feedback = ctx.session.state.get("cypher_feedback", "")

        # Stop the loop if validation passed
        should_stop = (feedback == "valid")

        yield Event(
            author=self.name,
            actions=EventActions(escalate=should_stop)
        )


# =============================================================================
# Loop Agent Factory
# =============================================================================

def create_cypher_generation_loop(
    generator_llm=None,
    validator_llm=None,
    max_iterations: int = 3,
    name: str = "cypher_generation_loop"
) -> LoopAgent:
    """
    Create a Cypher Generation Loop Agent.

    This loop agent orchestrates the query generation and validation process:
    1. CypherGeneratorAgent generates a Cypher query
    2. CypherValidatorAgent validates syntax, execution, and results
    3. CheckCypherValidationStatus checks if validation passed
    4. If not valid, loop back to step 1 with feedback

    The loop terminates when:
    - Validation passes (feedback == "valid")
    - Maximum iterations reached

    Args:
        generator_llm: LLM for query generation. Defaults to Silicon Cloud Qwen3-Coder.
        validator_llm: LLM for validation. Defaults to DashScope qwen-plus.
        max_iterations: Maximum refinement iterations (default 3)
        name: Agent name for identification

    Returns:
        LoopAgent: Configured loop agent for Cypher generation

    Example:
        ```python
        from src.agents import create_cypher_generation_loop, make_agent_caller

        # Create the loop agent
        loop = create_cypher_generation_loop()

        # Create a caller with initial state
        caller = await make_agent_caller(loop, {})

        # Generate and validate a query
        result = await caller.call("What is the age distribution of respondents?")

        # Get the validated query and results
        session = await caller.get_session()
        validated_query = session.state.get("validated_cypher_query")
        query_results = session.state.get("query_result")
        ```

    State Keys Used:
        - proposed_cypher_query: The query proposed by the generator
        - cypher_feedback: Validation feedback ("valid" or error description)
        - validated_cypher_query: The approved query after validation
        - query_result: Results from executing the validated query
    """
    # Create sub-agents
    generator = create_cypher_generator_agent(generator_llm)
    validator = create_cypher_validator_agent(validator_llm)
    stop_checker = CheckCypherValidationStatus(name="cypher_stop_checker")

    return LoopAgent(
        name=name,
        description=(
            "Generates and validates Cypher queries through iterative refinement. "
            "Uses Silicon Cloud Qwen3-Coder for generation and validates against Neo4j."
        ),
        max_iterations=max_iterations,
        sub_agents=[generator, validator, stop_checker],
    )


# =============================================================================
# Convenience Function
# =============================================================================

async def generate_validated_cypher(
    question: str,
    state: Optional[dict] = None,
    generator_llm=None,
    validator_llm=None,
    max_iterations: int = 3,
    verbose: bool = False
) -> dict:
    """
    Generate and validate a Cypher query for a natural language question.

    This is a convenience function that creates a loop agent, runs it,
    and returns the validated query and results.

    Args:
        question: Natural language question about the knowledge graph
        state: Optional initial state dictionary
        generator_llm: LLM for query generation
        validator_llm: LLM for validation
        max_iterations: Maximum refinement iterations
        verbose: Print debug output

    Returns:
        Dict with:
        - query: The validated Cypher query
        - results: Query execution results
        - feedback: Final feedback status
        - iterations: Number of iterations used

    Example:
        ```python
        from src.agents.cypher_loop_agent import generate_validated_cypher

        result = await generate_validated_cypher(
            "What is the age distribution of respondents?"
        )

        print(f"Query: {result['query']}")
        print(f"Results: {result['results']}")
        ```
    """
    from .base import make_agent_caller

    # Initialize state
    if state is None:
        state = {}

    # Clear any previous feedback
    state["cypher_feedback"] = ""

    # Create and run the loop
    loop = create_cypher_generation_loop(
        generator_llm=generator_llm,
        validator_llm=validator_llm,
        max_iterations=max_iterations,
    )

    caller = await make_agent_caller(loop, state)
    await caller.call(question, verbose=verbose)

    # Get results from session
    session = await caller.get_session()
    final_state = dict(session.state)

    return {
        "query": final_state.get("validated_cypher_query", ""),
        "results": final_state.get("query_result", []),
        "feedback": final_state.get("cypher_feedback", ""),
        "proposed_query": final_state.get("proposed_cypher_query", {}),
    }
