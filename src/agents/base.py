"""
Base agent utilities for Agentic KG.

Provides AgentCaller class and helper functions for agent execution.
"""

from typing import Any, Dict, Optional

from google.genai import types
from google.adk.agents import Agent
from google.adk.sessions import InMemorySessionService, Session
from google.adk.runners import Runner


class AgentCaller:
    """
    A wrapper class for interacting with an ADK agent.

    Provides convenient methods for calling agents and managing sessions.
    """

    def __init__(
        self,
        agent: Agent,
        runner: Runner,
        user_id: str,
        session_id: str
    ):
        """
        Initialize the AgentCaller.

        Args:
            agent: The ADK Agent instance
            runner: The Runner for executing the agent
            user_id: User identifier for the session
            session_id: Session identifier
        """
        self.agent = agent
        self.runner = runner
        self.user_id = user_id
        self.session_id = session_id
        self.session = None

    async def get_session(self) -> Session:
        """
        Get the current session.

        Returns:
            The current Session instance
        """
        return await self.runner.session_service.get_session(
            app_name=self.runner.app_name,
            user_id=self.user_id,
            session_id=self.session_id
        )

    async def call(self, query: str, verbose: bool = False) -> str:
        """
        Call the agent with a query and return the response.

        Args:
            query: The user's query text
            verbose: If True, print all events during execution

        Returns:
            The agent's final response text
        """
        print(f"\n>>> User Query: {query}")

        # Prepare the user's message in ADK format
        content = types.Content(
            role='user',
            parts=[types.Part(text=query)]
        )

        final_response_text = "Agent did not produce a final response."

        # Execute agent and iterate through events
        async for event in self.runner.run_async(
            user_id=self.user_id,
            session_id=self.session_id,
            new_message=content
        ):
            if verbose:
                print(
                    f"  [Event] Author: {event.author}, "
                    f"Type: {type(event).__name__}, "
                    f"Final: {event.is_final_response()}, "
                    f"Content: {event.content}"
                )

            # Check for final response
            # Note: For LoopAgents, final events come from sub-agents, not the loop itself
            # So we just check is_final_response() without checking author
            # BUT: We need content because before_agent_callbacks emit empty final events
            if event.is_final_response():
                if event.content and event.content.parts:
                    final_response_text = event.content.parts[0].text
                    break  # Only break on final response WITH content
                elif event.actions and event.actions.escalate:
                    final_response_text = (
                        f"Agent escalated: "
                        f"{event.error_message or 'No specific message.'}"
                    )
                    break

        # Update session reference
        self.session = await self.get_session()

        print(f"<<< Agent Response: {final_response_text}")
        return final_response_text

    async def get_state(self) -> Dict[str, Any]:
        """
        Get the current session state.

        Returns:
            Dictionary containing the session state
        """
        session = await self.get_session()
        return session.state if session else {}


async def make_agent_caller(
    agent: Agent,
    initial_state: Optional[Dict[str, Any]] = None
) -> AgentCaller:
    """
    Create and return an AgentCaller instance for the given agent.

    Args:
        agent: The ADK Agent to wrap
        initial_state: Optional initial state for the session

    Returns:
        AgentCaller instance ready for interaction
    """
    if initial_state is None:
        initial_state = {}

    session_service = InMemorySessionService()
    app_name = f"{agent.name}_app"
    user_id = f"{agent.name}_user"
    session_id = f"{agent.name}_session_01"

    # Initialize session with optional initial state
    await session_service.create_session(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
        state=initial_state
    )

    runner = Runner(
        agent=agent,
        app_name=app_name,
        session_service=session_service
    )

    return AgentCaller(agent, runner, user_id, session_id)


async def run_agent_conversation(
    agent: Agent,
    messages: list[str],
    initial_state: Optional[Dict[str, Any]] = None,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Run a complete conversation with an agent.

    Args:
        agent: The ADK Agent to run
        messages: List of user messages to send
        initial_state: Optional initial state for the session
        verbose: If True, print verbose output

    Returns:
        Dictionary with final state and responses
    """
    caller = await make_agent_caller(agent, initial_state)

    responses = []
    for message in messages:
        response = await caller.call(message, verbose)
        responses.append(response)

    final_state = await caller.get_state()

    return {
        "responses": responses,
        "final_state": final_state
    }
