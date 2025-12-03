"""
Schema Proposal Agent for Agentic KG.

Agents responsible for proposing and critiquing knowledge graph schemas.
"""

from typing import AsyncGenerator

from google.adk.agents import LlmAgent, LoopAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.events import Event, EventActions

from ..llm import get_adk_llm
from ..tools.user_intent import get_approved_user_goal
from ..tools.file_suggestion import get_approved_files, sample_file, search_file
from ..tools.kg_construction import (
    propose_node_construction,
    propose_relationship_construction,
    remove_node_construction,
    remove_relationship_construction,
    get_proposed_construction_plan,
    approve_proposed_construction_plan,
)


# Schema Proposal Agent Instructions
PROPOSAL_ROLE_AND_GOAL = """
You are an expert at knowledge graph modeling with property graphs. Propose an appropriate
schema by specifying construction rules which transform approved files into nodes or relationships.
The resulting schema should describe a knowledge graph based on the user goal.

Consider critic feedback if available (from internal validation):
<critic_feedback>
{feedback}
</critic_feedback>

IMPORTANT: Also pay close attention to the user's latest message in the conversation.
If the user provides domain knowledge corrections (e.g., "上汽集团 纯电动 represents a vehicle series"),
treat that as the highest priority feedback and modify your schema accordingly.
"""

PROPOSAL_HINTS = """
Every file in the approved files list will become either a node or a relationship.
Determining whether a file likely represents a node or a relationship is based
on a hint from the filename (is it a single thing or two things) and the
identifiers found within the file.

Because unique identifiers are so important for determining the structure of the graph,
always verify the uniqueness of suspected unique identifiers using the 'search_file' tool.

General guidance for identifying a node or a relationship:
- If the file name is singular and has only 1 unique identifier it is likely a node
- If the file name is a combination of two things, it is likely a full relationship
- If the file name sounds like a node, but there are multiple unique identifiers, that is likely a node with reference relationships

Design rules for nodes:
- Nodes will have unique identifiers.
- Nodes _may_ have identifiers that are used as reference relationships.

Design rules for relationships:
- Relationships appear in two ways: full relationships and reference relationships.

Full relationships:
- Full relationships appear in dedicated relationship files, often having a filename that references two entities
- Full relationships typically have references to a source and destination node.
- Full relationships _do not have_ unique identifiers, but instead have references to the primary keys of the source and destination nodes.
- The absence of a single, unique identifier is a strong indicator that a file is a full relationship.

Reference relationships:
- Reference relationships appear as foreign key references in node files
- Reference relationship foreign key column names often hint at the destination node and relationship type
- References may be hierarchical container relationships, with terminology revealing parent-child, "has", "contains", membership, or similar relationship
- References may be peer relationships, that is often a self-reference to a similar class of nodes. For example, "knows" or "see also"

The resulting schema should be a connected graph, with no isolated components.
"""

PROPOSAL_CHAIN_OF_THOUGHT = """
WORKFLOW - Follow these steps in order:

Step 1: Gather context
- Get the user goal using the 'get_approved_user_goal' tool
- Get the list of approved files using the 'get_approved_files' tool
- Get the current construction plan using the 'get_proposed_construction_plan' tool
- Check if there is any feedback to incorporate (from critic or user)

Step 2: Handle feedback if exists
There are two types of feedback:
a) Critic feedback (internal): Technical issues like "unique identifiers are not unique"
b) User feedback (external): Domain corrections like "上汽集团 纯电动 represents a vehicle series"

If feedback exists:
- Read the feedback carefully and understand what changes are needed
- Use 'remove_node_construction' or 'remove_relationship_construction' to remove incorrect constructions
- Use 'propose_node_construction' or 'propose_relationship_construction' to add corrected constructions
- Go to Step 4 after incorporating feedback

Step 3: If no existing plan, design the initial schema
- For each approved file, examine its structure using 'sample_file'
- Determine if each file represents nodes or relationships
- For node files: propose using 'propose_node_construction'
- For relationship files: propose using 'propose_relationship_construction'
- Ensure all nodes are connected in the resulting graph

Step 4: Present the schema to the user
- Use 'get_proposed_construction_plan' to retrieve the complete plan
- Explain the schema clearly to the user in a structured format
- Ask explicitly: "Do you want to approve this schema, or do you have modifications?"

Step 5: Handle user response
- If user approves (says "ok", "approve", "确认", "同意", "可以", "批准", etc.):
  IMMEDIATELY call 'approve_proposed_construction_plan' tool to finalize the schema
- If user provides corrections or feedback:
  DO NOT call approve. Instead, incorporate the feedback and modify the schema accordingly.
  The user is the domain expert - trust their corrections about data meaning.

CRITICAL RULES:
- NEVER call 'approve_proposed_construction_plan' until the user EXPLICITLY confirms
- User feedback takes priority over critic feedback
- Chinese column headers like "上汽集团 纯电动" may represent specific domain concepts (e.g., brand-powertrain combinations, vehicle series)
- When user corrects your understanding of the data, update the schema immediately
"""

PROPOSAL_INSTRUCTION = f"""
{PROPOSAL_ROLE_AND_GOAL}
{PROPOSAL_HINTS}
{PROPOSAL_CHAIN_OF_THOUGHT}
"""

# Schema Critic Agent Instructions
CRITIC_ROLE_AND_GOAL = """
You are an expert at knowledge graph modeling with property graphs.
Criticize the proposed schema for relevance to the user goal and approved files.
"""

CRITIC_HINTS = """
Criticize the proposed schema for relevance and correctness:
- Are unique identifiers actually unique? Use the 'search_file' tool to validate. Composite identifier are not acceptable.
- Could any nodes be relationships instead? Double-check that unique identifiers are unique and not references to other nodes. Use the 'search_file' tool to validate
- Can you manually trace through the source data to find the necessary information for answering a hypothetical question?
- Is every node in the schema connected? What relationships could be missing? Every node should connect to at least one other node.
- Are hierarchical container relationships missing?
- Are any relationships redundant? A relationship between two nodes is redundant if it is semantically equivalent to or the inverse of another relationship between those two nodes.
"""

CRITIC_CHAIN_OF_THOUGHT = """
Prepare for the task:
- get the user goal using the 'get_approved_user_goal' tool
- get the list of approved files using the 'get_approved_files' tool
- get the construction plan using the 'get_proposed_construction_plan' tool
- use the 'sample_file' and 'search_file' tools to validate the schema design

Think carefully, using tools to perform actions and reconsidering your actions when a tool returns an error:
1. Analyze each construction rule in the proposed construction plan.
2. Use tools to validate the construction rules for relevance and correctness.
3. If the schema looks good, respond with a one word reply: 'valid'.
4. If the schema has problems, respond with 'retry' and provide feedback as a concise bullet list of problems.
"""

CRITIC_INSTRUCTION = f"""
{CRITIC_ROLE_AND_GOAL}
{CRITIC_HINTS}
{CRITIC_CHAIN_OF_THOUGHT}
"""

# Tool lists
SCHEMA_PROPOSAL_TOOLS = [
    get_approved_user_goal,
    get_approved_files,
    get_proposed_construction_plan,
    sample_file,
    search_file,
    propose_node_construction,
    propose_relationship_construction,
    remove_node_construction,
    remove_relationship_construction,
    approve_proposed_construction_plan,
]

SCHEMA_CRITIC_TOOLS = [
    get_approved_user_goal,
    get_approved_files,
    get_proposed_construction_plan,
    sample_file,
    search_file,
]


def log_agent(callback_context: CallbackContext) -> None:
    """Log agent entry for debugging."""
    print(f"\n### Entering Agent: {callback_context.agent_name}")


class CheckStatusAndEscalate(BaseAgent):
    """
    Agent that checks feedback status and decides whether to continue or stop the loop.

    If feedback is 'valid', escalates to stop the loop.
    Otherwise, continues iteration for refinement.
    """

    async def _run_async_impl(
        self,
        ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        """Check feedback and yield escalation event if valid."""
        feedback = ctx.session.state.get("feedback", "valid")
        should_stop = (feedback == "valid")
        yield Event(
            author=self.name,
            actions=EventActions(escalate=should_stop)
        )


def create_schema_proposal_agent(
    llm=None,
    name: str = "schema_proposal_agent_v1"
) -> LlmAgent:
    """
    Create a Schema Proposal Agent.

    The schema proposal agent analyzes approved files and proposes
    construction rules for building a knowledge graph schema.

    Args:
        llm: Optional LLM instance. If None, uses default from get_adk_llm()
        name: Agent name for identification

    Returns:
        LlmAgent: Configured Schema Proposal Agent
    """
    if llm is None:
        llm = get_adk_llm()

    return LlmAgent(
        name=name,
        model=llm,
        description="Proposes a knowledge graph schema based on the user goal and approved file list",
        instruction=PROPOSAL_INSTRUCTION,
        tools=SCHEMA_PROPOSAL_TOOLS,
        before_agent_callback=log_agent,
    )


def create_schema_critic_agent(
    llm=None,
    name: str = "schema_critic_agent_v1"
) -> LlmAgent:
    """
    Create a Schema Critic Agent.

    The schema critic agent reviews and validates proposed schemas,
    providing feedback for refinement.

    Args:
        llm: Optional LLM instance. If None, uses default from get_adk_llm()
        name: Agent name for identification

    Returns:
        LlmAgent: Configured Schema Critic Agent
    """
    if llm is None:
        llm = get_adk_llm()

    return LlmAgent(
        name=name,
        model=llm,
        description="Criticizes the proposed schema for relevance to the user goal and approved files.",
        instruction=CRITIC_INSTRUCTION,
        tools=SCHEMA_CRITIC_TOOLS,
        output_key="feedback",  # Store critic result in feedback key
        before_agent_callback=log_agent,
    )


def create_schema_refinement_loop(
    llm=None,
    max_iterations: int = 2,
    name: str = "schema_refinement_loop"
) -> LoopAgent:
    """
    Create a Schema Refinement Loop Agent.

    This loop agent coordinates the schema proposal and critic agents
    in an iterative refinement process.

    Args:
        llm: Optional LLM instance. If None, uses default from get_adk_llm()
        max_iterations: Maximum refinement iterations
        name: Agent name for identification

    Returns:
        LoopAgent: Configured Schema Refinement Loop

    Example:
        ```python
        from src.agents import create_schema_refinement_loop, make_agent_caller

        loop = create_schema_refinement_loop()
        caller = await make_agent_caller(loop, {
            "feedback": "",
            "approved_user_goal": {"kind_of_graph": "supply chain", "description": "BOM graph"},
            "approved_files": ["products.csv", "assemblies.csv"]
        })

        # First call - loop proposes and self-critiques schema
        await caller.call("How can these files be imported?")

        # User provides feedback - loop incorporates and re-proposes
        await caller.call("The column X should be a separate node type")

        # User approves - loop calls approve tool
        await caller.call("Approve the schema")
        ```
    """
    proposal_agent = create_schema_proposal_agent(llm)
    critic_agent = create_schema_critic_agent(llm)
    stop_checker = CheckStatusAndEscalate(name="StopChecker")

    return LoopAgent(
        name=name,
        description="Analyzes approved files to propose a schema based on user intent and feedback",
        max_iterations=max_iterations,
        sub_agents=[proposal_agent, critic_agent, stop_checker],
        before_agent_callback=log_agent,
    )
