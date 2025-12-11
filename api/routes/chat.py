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
        import asyncio

        # Background task for processing messages
        processing_task: Optional[asyncio.Task] = None

        async def process_and_stream(content: str, phase: Optional[PipelinePhase]):
            """Process message and stream responses in background."""
            try:
                async for response in service.process_message(session_id, content, phase):
                    # Check if cancelled before sending each response
                    if session._cancel_requested:
                        break
                    await manager.send_message(session_id, response)
            except asyncio.CancelledError:
                # Don't send message here - will be sent by cancel handler
                raise
            except Exception as e:
                await manager.send_message(
                    session_id,
                    WebSocketMessage(
                        type=WebSocketMessageType.ERROR,
                        error=f"Processing error: {str(e)}",
                    )
                )

        async def receive_messages(ws_queue: asyncio.Queue):
            """Receive WebSocket messages and put them in queue."""
            while True:
                try:
                    data = await websocket.receive_text()
                    await ws_queue.put(data)
                except Exception:
                    break

        # Start message receiver
        ws_queue: asyncio.Queue = asyncio.Queue()
        receiver_task = asyncio.create_task(receive_messages(ws_queue))

        try:
            while True:
                # Wait for message from queue
                data = await ws_queue.get()

                try:
                    message_data = json.loads(data)
                    msg_type = message_data.get("type", "message")
                    content = message_data.get("content", "")
                    phase_str = message_data.get("phase")

                    if msg_type == "cancel":
                        # Set cancel flag
                        session.cancel()

                        # Cancel the background task if running
                        if processing_task and not processing_task.done():
                            processing_task.cancel()
                            try:
                                await asyncio.wait_for(processing_task, timeout=1.0)
                            except (asyncio.CancelledError, asyncio.TimeoutError):
                                pass

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

                    # Cancel any existing task before starting new one
                    if processing_task and not processing_task.done():
                        session.cancel()
                        processing_task.cancel()
                        try:
                            await asyncio.wait_for(processing_task, timeout=1.0)
                        except (asyncio.CancelledError, asyncio.TimeoutError):
                            pass

                    # Reset cancel flag for new request
                    session._cancel_requested = False

                    # Start processing in background - don't await it
                    processing_task = asyncio.create_task(process_and_stream(content, phase))

                except json.JSONDecodeError:
                    await manager.send_message(
                        session_id,
                        WebSocketMessage(
                            type=WebSocketMessageType.ERROR,
                            error="Invalid JSON message",
                        )
                    )

        finally:
            receiver_task.cancel()
            if processing_task and not processing_task.done():
                processing_task.cancel()

    except WebSocketDisconnect:
        manager.disconnect(session_id)
