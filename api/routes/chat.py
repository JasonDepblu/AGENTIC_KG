"""
WebSocket chat endpoint for real-time agent communication.
"""

import json
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import Optional

from ..services.pipeline import PipelineService
from ..models.schemas import (
    PipelinePhase,
    WebSocketMessage,
    WebSocketMessageType,
)

router = APIRouter(tags=["chat"])

# Singleton pipeline service
_pipeline_service: Optional[PipelineService] = None


def get_pipeline_service() -> PipelineService:
    """Get or create pipeline service instance."""
    global _pipeline_service
    if _pipeline_service is None:
        _pipeline_service = PipelineService()
    return _pipeline_service


class ConnectionManager:
    """Manages WebSocket connections."""

    def __init__(self):
        self.active_connections: dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, session_id: str):
        """Accept and store a WebSocket connection."""
        await websocket.accept()
        self.active_connections[session_id] = websocket

    def disconnect(self, session_id: str):
        """Remove a WebSocket connection."""
        if session_id in self.active_connections:
            del self.active_connections[session_id]

    async def send_message(self, session_id: str, message: WebSocketMessage):
        """Send a message to a specific session."""
        if session_id in self.active_connections:
            websocket = self.active_connections[session_id]
            await websocket.send_json(message.model_dump(mode="json"))


manager = ConnectionManager()


@router.websocket("/chat/{session_id}")
async def websocket_chat(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for chat with the KG pipeline.

    Message format (client -> server):
    - {"type": "message", "content": "user message"}
    - {"type": "approve", "phase": "files"}
    - {"type": "cancel"}

    Message format (server -> client):
    - {"type": "connected", "session_id": "..."}
    - {"type": "agent_event", "content": "...", "author": "...", "is_final": false}
    - {"type": "phase_change", "phase": "user_intent"}
    - {"type": "phase_complete", "phase": "...", "state": {...}}
    - {"type": "error", "error": "..."}
    """
    service = get_pipeline_service()

    # Check if session exists, create if not
    session = service.get_session(session_id)
    if not session:
        session = service.create_session()
        session_id = session.id

    await manager.connect(websocket, session_id)

    # Send connected message
    await manager.send_message(
        session_id,
        WebSocketMessage(
            type=WebSocketMessageType.CONNECTED,
            content=session_id,
            phase=session.phase,
            state=session.state,
        )
    )

    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()

            try:
                message_data = json.loads(data)
                msg_type = message_data.get("type", "message")
                content = message_data.get("content", "")
                phase_str = message_data.get("phase")

                if msg_type == "cancel":
                    session.cancel()
                    await manager.send_message(
                        session_id,
                        WebSocketMessage(
                            type=WebSocketMessageType.ERROR,
                            error="Operation cancelled by user",
                        )
                    )
                    continue

                # Determine phase
                phase = None
                if phase_str:
                    try:
                        phase = PipelinePhase(phase_str)
                    except ValueError:
                        pass

                # Process message and stream responses
                async for response in service.process_message(session_id, content, phase):
                    await manager.send_message(session_id, response)

            except json.JSONDecodeError:
                await manager.send_message(
                    session_id,
                    WebSocketMessage(
                        type=WebSocketMessageType.ERROR,
                        error="Invalid JSON message",
                    )
                )

    except WebSocketDisconnect:
        manager.disconnect(session_id)
