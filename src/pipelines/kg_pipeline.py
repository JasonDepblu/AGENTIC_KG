"""
Knowledge Graph Pipeline for Agentic KG.

Orchestrates the complete workflow from user intent to knowledge graph construction.
"""

from typing import Any, Dict, List, Optional

from ..llm import get_adk_llm
from ..neo4j_client import get_graphdb
from ..agents.base import make_agent_caller
from ..agents.user_intent_agent import create_user_intent_agent
from ..agents.file_suggestion_agent import create_file_suggestion_agent
from ..agents.schema_proposal_agent import create_schema_refinement_loop
from ..agents.kg_builder_agent import create_kg_builder_agent, build_domain_graph
from ..tools.kg_construction import APPROVED_CONSTRUCTION_PLAN


class KGPipeline:
    """
    Complete Knowledge Graph construction pipeline.

    Orchestrates all agents in sequence:
    1. User Intent Agent - captures user goals
    2. File Suggestion Agent - identifies relevant files
    3. Schema Refinement Loop - proposes and refines schema
    4. KG Builder Agent - constructs the graph

    Example:
        ```python
        pipeline = KGPipeline()

        # Run interactive pipeline
        result = await pipeline.run_interactive()

        # Or run with predefined inputs
        result = await pipeline.run_with_inputs(
            user_goal="I want a supply chain graph for bill of materials analysis",
            file_approval="approve",
            schema_approval="approve"
        )
        ```
    """

    def __init__(self, llm=None, verbose: bool = False):
        """
        Initialize the KG Pipeline.

        Args:
            llm: Optional LLM instance. Uses default if not provided.
            verbose: Enable verbose output during execution.
        """
        self.llm = llm or get_adk_llm()
        self.verbose = verbose
        self.state: Dict[str, Any] = {}

    async def run_user_intent_phase(
        self,
        user_input: str,
        approval: str = "approve"
    ) -> Dict[str, Any]:
        """
        Run the user intent capture phase.

        Args:
            user_input: User's description of their knowledge graph goal
            approval: Approval message after goal is perceived

        Returns:
            Dictionary with approved user goal
        """
        print("\n" + "=" * 60)
        print("Phase 1: User Intent")
        print("=" * 60)

        agent = create_user_intent_agent(self.llm)
        caller = await make_agent_caller(agent, self.state)

        await caller.call(user_input, self.verbose)
        await caller.call(approval, self.verbose)

        session = await caller.get_session()
        self.state = dict(session.state)

        return {
            "approved_user_goal": self.state.get("approved_user_goal"),
            "state": self.state
        }

    async def run_file_suggestion_phase(
        self,
        approval: str = "approve"
    ) -> Dict[str, Any]:
        """
        Run the file suggestion phase.

        Requires: approved_user_goal in state

        Args:
            approval: Approval message for suggested files

        Returns:
            Dictionary with approved files
        """
        print("\n" + "=" * 60)
        print("Phase 2: File Suggestion")
        print("=" * 60)

        if "approved_user_goal" not in self.state:
            raise ValueError("approved_user_goal not in state. Run user_intent phase first.")

        agent = create_file_suggestion_agent(self.llm)
        caller = await make_agent_caller(agent, self.state)

        await caller.call("What files can we use for import?", self.verbose)
        await caller.call(approval, self.verbose)

        session = await caller.get_session()
        self.state = dict(session.state)

        return {
            "approved_files": self.state.get("approved_files"),
            "state": self.state
        }

    async def run_schema_proposal_phase(
        self,
        approval: str = "approve"
    ) -> Dict[str, Any]:
        """
        Run the schema proposal and refinement phase.

        Requires: approved_user_goal, approved_files in state

        Args:
            approval: Approval message for proposed schema

        Returns:
            Dictionary with approved construction plan
        """
        print("\n" + "=" * 60)
        print("Phase 3: Schema Proposal")
        print("=" * 60)

        if "approved_files" not in self.state:
            raise ValueError("approved_files not in state. Run file_suggestion phase first.")

        # Initialize feedback for refinement loop
        self.state["feedback"] = ""

        loop = create_schema_refinement_loop(self.llm)
        caller = await make_agent_caller(loop, self.state)

        await caller.call("How can these files be imported?", self.verbose)

        session = await caller.get_session()
        self.state = dict(session.state)

        # If there's a proposed plan, approve it
        if "proposed_construction_plan" in self.state:
            self.state["approved_construction_plan"] = self.state["proposed_construction_plan"]

        return {
            "approved_construction_plan": self.state.get("approved_construction_plan"),
            "state": self.state
        }

    async def run_construction_phase(self) -> Dict[str, Any]:
        """
        Run the knowledge graph construction phase.

        Requires: approved_construction_plan in state

        Returns:
            Dictionary with construction results
        """
        print("\n" + "=" * 60)
        print("Phase 4: KG Construction")
        print("=" * 60)

        if APPROVED_CONSTRUCTION_PLAN not in self.state:
            raise ValueError(
                "approved_construction_plan not in state. "
                "Run schema_proposal phase first."
            )

        construction_plan = self.state[APPROVED_CONSTRUCTION_PLAN]
        result = build_domain_graph(construction_plan)

        return {
            "construction_result": result,
            "state": self.state
        }

    async def run_full_pipeline(
        self,
        user_goal: str,
        file_approval: str = "Yes, use those files",
        schema_approval: str = "Yes, approve that schema"
    ) -> Dict[str, Any]:
        """
        Run the complete pipeline with all phases.

        Args:
            user_goal: User's description of their knowledge graph goal
            file_approval: Approval message for file suggestions
            schema_approval: Approval message for schema proposal

        Returns:
            Dictionary with final state and all results
        """
        print("\n" + "=" * 60)
        print("STARTING FULL KG PIPELINE")
        print("=" * 60)

        # Phase 1: User Intent
        intent_result = await self.run_user_intent_phase(
            user_goal,
            "Approve that goal"
        )

        # Phase 2: File Suggestion
        files_result = await self.run_file_suggestion_phase(file_approval)

        # Phase 3: Schema Proposal
        schema_result = await self.run_schema_proposal_phase(schema_approval)

        # Phase 4: Construction
        construction_result = await self.run_construction_phase()

        print("\n" + "=" * 60)
        print("PIPELINE COMPLETE")
        print("=" * 60)

        return {
            "user_goal": intent_result.get("approved_user_goal"),
            "approved_files": files_result.get("approved_files"),
            "construction_plan": schema_result.get("approved_construction_plan"),
            "construction_result": construction_result.get("construction_result"),
            "final_state": self.state
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """
        Set the pipeline state manually.

        Useful for resuming from a previous state or testing.

        Args:
            state: State dictionary to set
        """
        self.state = dict(state)

    def get_state(self) -> Dict[str, Any]:
        """
        Get the current pipeline state.

        Returns:
            Current state dictionary
        """
        return dict(self.state)


async def run_full_pipeline(
    user_goal: str,
    llm=None,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Convenience function to run the complete pipeline.

    Args:
        user_goal: User's description of their knowledge graph goal
        llm: Optional LLM instance
        verbose: Enable verbose output

    Returns:
        Dictionary with pipeline results
    """
    pipeline = KGPipeline(llm=llm, verbose=verbose)
    return await pipeline.run_full_pipeline(user_goal)
