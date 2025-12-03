"""
Pipeline service with streaming support.

Wraps the KGPipeline to provide real-time event streaming via WebSocket.
"""

import asyncio
import uuid
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, Optional

from google.genai import types

from src.llm import get_adk_llm
from src.agents.base import make_agent_caller
from src.agents.user_intent_agent import create_user_intent_agent
from src.agents.file_suggestion_agent import create_file_suggestion_agent
from src.agents.data_preprocessing_agent import create_data_preprocessing_agent
from src.agents.schema_proposal_agent import create_schema_refinement_loop
from src.agents.kg_builder_agent import build_domain_graph
from src.agents.kg_query_agent import create_kg_query_agent
from src.tools.kg_construction import APPROVED_CONSTRUCTION_PLAN

from ..models.schemas import (
    PipelinePhase,
    SessionInfo,
    SessionState,
    WebSocketMessage,
    WebSocketMessageType,
)


class PipelineSession:
    """
    Represents a single pipeline session with state management.
    """

    def __init__(self, session_id: str):
        self.id = session_id
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        self.phase = PipelinePhase.IDLE
        self.state: Dict[str, Any] = {}
        self.messages: list = []
        self.llm = get_adk_llm()
        self._current_caller = None
        self._cancel_requested = False
        # Cache agents and callers to maintain conversation history
        self._agents: Dict[str, Any] = {}
        self._callers: Dict[str, Any] = {}

    def to_info(self) -> SessionInfo:
        """Convert to SessionInfo model."""
        return SessionInfo(
            id=self.id,
            created_at=self.created_at,
            updated_at=self.updated_at,
            phase=self.phase,
            state=SessionState(
                approved_user_goal=self.state.get("approved_user_goal"),
                approved_files=self.state.get("approved_files"),
                proposed_construction_plan=self.state.get("proposed_construction_plan"),
                approved_construction_plan=self.state.get("approved_construction_plan"),
                feedback=self.state.get("feedback"),
            ),
            message_count=len(self.messages),
        )

    def cancel(self):
        """Request cancellation of current operation."""
        self._cancel_requested = True

    async def run_phase_with_streaming(
        self,
        phase: PipelinePhase,
        user_message: str
    ) -> AsyncGenerator[WebSocketMessage, None]:
        """
        Run a pipeline phase with event streaming.

        Args:
            phase: The phase to run
            user_message: User's input message

        Yields:
            WebSocketMessage for each event
        """
        self._cancel_requested = False
        self.phase = phase
        self.updated_at = datetime.now()

        # Notify phase change
        yield WebSocketMessage(
            type=WebSocketMessageType.PHASE_CHANGE,
            phase=phase,
        )

        try:
            if phase == PipelinePhase.USER_INTENT:
                async for msg in self._run_user_intent(user_message):
                    yield msg
            elif phase == PipelinePhase.FILE_SUGGESTION:
                async for msg in self._run_file_suggestion(user_message):
                    yield msg
            elif phase == PipelinePhase.DATA_PREPROCESSING:
                async for msg in self._run_data_preprocessing(user_message):
                    yield msg
            elif phase == PipelinePhase.SCHEMA_PROPOSAL:
                async for msg in self._run_schema_proposal(user_message):
                    yield msg
            elif phase == PipelinePhase.CONSTRUCTION:
                async for msg in self._run_construction():
                    yield msg
            elif phase == PipelinePhase.QUERY:
                async for msg in self._run_query(user_message):
                    yield msg

        except Exception as e:
            self.phase = PipelinePhase.ERROR
            yield WebSocketMessage(
                type=WebSocketMessageType.ERROR,
                error=str(e),
            )

    async def _get_or_create_caller(self, agent, agent_key: str):
        """Get existing caller or create a new one for the agent."""
        if agent_key not in self._callers:
            caller = await make_agent_caller(agent, self.state)
            self._callers[agent_key] = caller
            self._agents[agent_key] = agent
        return self._callers[agent_key]

    async def _run_agent_with_streaming(
        self,
        agent,
        message: str,
        agent_name: str
    ) -> AsyncGenerator[WebSocketMessage, None]:
        """
        Run an agent and stream events.

        Args:
            agent: The ADK agent to run
            message: User message
            agent_name: Name for display

        Yields:
            WebSocketMessage for each event
        """
        # Use agent name as key to reuse caller across messages
        caller = await self._get_or_create_caller(agent, agent.name)
        self._current_caller = caller

        # Prepare user message
        content = types.Content(
            role='user',
            parts=[types.Part(text=message)]
        )

        # Stream events
        async for event in caller.runner.run_async(
            user_id=caller.user_id,
            session_id=caller.session_id,
            new_message=content
        ):
            if self._cancel_requested:
                yield WebSocketMessage(
                    type=WebSocketMessageType.ERROR,
                    error="Operation cancelled",
                )
                return

            # Yield event as message
            event_content = ""
            if event.content and event.content.parts:
                event_content = event.content.parts[0].text

            yield WebSocketMessage(
                type=WebSocketMessageType.AGENT_EVENT,
                content=event_content,
                author=event.author,
                is_final=event.is_final_response(),
            )

            # Check for final response
            if event.is_final_response():
                if event.author == agent.name:
                    break

        # Update state from session
        session = await caller.get_session()
        self.state = dict(session.state)

        # Send state update
        yield WebSocketMessage(
            type=WebSocketMessageType.STATE_UPDATE,
            state=self.state,
        )

    def _get_or_create_agent(self, agent_key: str, creator_func):
        """Get existing agent or create a new one."""
        if agent_key not in self._agents:
            self._agents[agent_key] = creator_func(self.llm)
        return self._agents[agent_key]

    async def _run_user_intent(self, user_message: str) -> AsyncGenerator[WebSocketMessage, None]:
        """Run user intent phase."""
        # Reuse the same agent instance across messages
        agent = self._get_or_create_agent("user_intent", create_user_intent_agent)

        # Run agent with user's message
        async for msg in self._run_agent_with_streaming(agent, user_message, "User Intent Agent"):
            yield msg

        # Check if user goal was approved after this interaction
        if "approved_user_goal" in self.state:
            # Phase complete - goal was approved
            yield WebSocketMessage(
                type=WebSocketMessageType.PHASE_COMPLETE,
                phase=PipelinePhase.USER_INTENT,
                state=self.state,
            )
        # Otherwise, wait for user to continue the conversation (confirmation/refinement)

    async def _run_file_suggestion(self, user_message: str = "What files can we use for import?") -> AsyncGenerator[WebSocketMessage, None]:
        """Run file suggestion phase."""
        if "approved_user_goal" not in self.state:
            yield WebSocketMessage(
                type=WebSocketMessageType.ERROR,
                error="User intent phase must be completed first",
            )
            return

        # Reuse the same agent instance across messages
        agent = self._get_or_create_agent("file_suggestion", create_file_suggestion_agent)

        # Run agent with user's message
        async for msg in self._run_agent_with_streaming(
            agent, user_message, "File Suggestion Agent"
        ):
            yield msg

        # Check if files were approved after this interaction
        if "approved_files" in self.state:
            yield WebSocketMessage(
                type=WebSocketMessageType.PHASE_COMPLETE,
                phase=PipelinePhase.FILE_SUGGESTION,
                state=self.state,
            )
        # Otherwise, wait for user to continue the conversation

    async def _run_data_preprocessing(self, user_message: str = "Analyze and preprocess the approved files") -> AsyncGenerator[WebSocketMessage, None]:
        """Run data preprocessing phase."""
        if "approved_files" not in self.state:
            yield WebSocketMessage(
                type=WebSocketMessageType.ERROR,
                error="File suggestion phase must be completed first",
            )
            return

        # Reuse the same agent instance across messages
        agent = self._get_or_create_agent("data_preprocessing", create_data_preprocessing_agent)

        # Run agent with user's message
        async for msg in self._run_agent_with_streaming(
            agent, user_message, "Data Preprocessing Agent"
        ):
            yield msg

        # Check if preprocessing was completed after this interaction
        if self.state.get("preprocessing_complete"):
            yield WebSocketMessage(
                type=WebSocketMessageType.PHASE_COMPLETE,
                phase=PipelinePhase.DATA_PREPROCESSING,
                state=self.state,
            )
        # Otherwise, wait for user to continue the conversation (e.g., approve transformation)

    async def _run_schema_proposal(self, user_message: str = "How can these files be imported?") -> AsyncGenerator[WebSocketMessage, None]:
        """Run schema proposal phase."""
        if "approved_files" not in self.state:
            yield WebSocketMessage(
                type=WebSocketMessageType.ERROR,
                error="File suggestion phase must be completed first",
            )
            return

        if "feedback" not in self.state:
            self.state["feedback"] = ""

        # Reuse the same loop agent instance
        loop = self._get_or_create_agent("schema_proposal", create_schema_refinement_loop)

        async for msg in self._run_agent_with_streaming(
            loop, user_message, "Schema Proposal Agent"
        ):
            yield msg

        # Check if schema was explicitly approved by the agent calling approve_proposed_construction_plan
        # Do NOT auto-approve - let the agent and user iterate on the schema
        if APPROVED_CONSTRUCTION_PLAN in self.state:
            yield WebSocketMessage(
                type=WebSocketMessageType.PHASE_COMPLETE,
                phase=PipelinePhase.SCHEMA_PROPOSAL,
                state=self.state,
            )
        # Otherwise, continue the conversation - user may provide feedback to refine the schema

    async def _run_construction(self) -> AsyncGenerator[WebSocketMessage, None]:
        """Run KG construction phase."""
        if APPROVED_CONSTRUCTION_PLAN not in self.state:
            yield WebSocketMessage(
                type=WebSocketMessageType.ERROR,
                error="Schema proposal phase must be completed first",
            )
            return

        yield WebSocketMessage(
            type=WebSocketMessageType.AGENT_EVENT,
            content="Starting knowledge graph construction...",
            author="KG Builder",
        )

        try:
            construction_plan = self.state[APPROVED_CONSTRUCTION_PLAN]
            result = build_domain_graph(construction_plan)

            self.state["construction_result"] = result
            self.state["construction_complete"] = True  # Mark construction as done
            self.phase = PipelinePhase.QUERY  # Transition to query phase

            yield WebSocketMessage(
                type=WebSocketMessageType.AGENT_RESPONSE,
                content=f"Knowledge graph construction complete!\n\n{result}\n\nYou can now ask questions about the knowledge graph.",
                author="KG Builder",
                is_final=True,
            )

            yield WebSocketMessage(
                type=WebSocketMessageType.PHASE_COMPLETE,
                phase=PipelinePhase.CONSTRUCTION,
                state=self.state,
            )

        except Exception as e:
            yield WebSocketMessage(
                type=WebSocketMessageType.ERROR,
                error=f"Construction failed: {str(e)}",
            )

    async def _run_query(self, user_message: str) -> AsyncGenerator[WebSocketMessage, None]:
        """Run knowledge graph query phase."""
        if not self.state.get("construction_complete"):
            yield WebSocketMessage(
                type=WebSocketMessageType.ERROR,
                error="Knowledge graph must be constructed first before querying",
            )
            return

        # Reuse the same query agent instance across messages
        agent = self._get_or_create_agent("kg_query", create_kg_query_agent)

        # Run agent with user's query
        async for msg in self._run_agent_with_streaming(
            agent, user_message, "KG Query Agent"
        ):
            yield msg

        # Query phase stays active - user can keep asking questions
        # No PHASE_COMPLETE is sent as this is an ongoing interaction phase


class SessionManager:
    """
    Manages pipeline sessions.
    """

    def __init__(self):
        self._sessions: Dict[str, PipelineSession] = {}

    def create_session(self) -> PipelineSession:
        """Create a new session."""
        session_id = str(uuid.uuid4())
        session = PipelineSession(session_id)
        self._sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> Optional[PipelineSession]:
        """Get a session by ID."""
        return self._sessions.get(session_id)

    def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False

    def list_sessions(self) -> list[SessionInfo]:
        """List all sessions."""
        return [s.to_info() for s in self._sessions.values()]


class PipelineService:
    """
    High-level service for pipeline operations.
    """

    def __init__(self):
        self.session_manager = SessionManager()

    def create_session(self) -> PipelineSession:
        """Create a new pipeline session."""
        return self.session_manager.create_session()

    def get_session(self, session_id: str) -> Optional[PipelineSession]:
        """Get a session by ID."""
        return self.session_manager.get_session(session_id)

    async def process_message(
        self,
        session_id: str,
        message: str,
        phase: Optional[PipelinePhase] = None
    ) -> AsyncGenerator[WebSocketMessage, None]:
        """
        Process a user message and stream responses.

        Args:
            session_id: Session ID
            message: User message
            phase: Optional phase override

        Yields:
            WebSocketMessage events
        """
        session = self.get_session(session_id)
        if not session:
            yield WebSocketMessage(
                type=WebSocketMessageType.ERROR,
                error=f"Session not found: {session_id}",
            )
            return

        # Determine phase
        if phase is None:
            phase = self._determine_next_phase(session)

        async for msg in session.run_phase_with_streaming(phase, message):
            yield msg

    def _determine_next_phase(self, session: PipelineSession) -> PipelinePhase:
        """Determine the next phase based on current state."""
        # Check state to determine which phase to run
        # Order matters! Check from later phases to earlier phases
        if session.state.get("construction_complete"):
            # KG is built, go to query phase
            return PipelinePhase.QUERY
        elif "approved_construction_plan" in session.state:
            return PipelinePhase.CONSTRUCTION
        elif session.state.get("preprocessing_complete"):
            return PipelinePhase.SCHEMA_PROPOSAL
        elif "approved_files" in session.state:
            return PipelinePhase.DATA_PREPROCESSING
        elif "approved_user_goal" in session.state:
            return PipelinePhase.FILE_SUGGESTION
        else:
            return PipelinePhase.USER_INTENT
