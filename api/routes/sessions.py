"""
Session management API routes.
"""

from fastapi import APIRouter, HTTPException
from typing import List

from ..services.pipeline import PipelineService
from ..models.schemas import (
    SessionInfo,
    SessionCreateRequest,
    SessionCreateResponse,
    APIResponse,
)

router = APIRouter(prefix="/sessions", tags=["sessions"])

# Singleton pipeline service
_pipeline_service: PipelineService = None


def get_pipeline_service() -> PipelineService:
    """Get or create pipeline service instance."""
    global _pipeline_service
    if _pipeline_service is None:
        _pipeline_service = PipelineService()
    return _pipeline_service


@router.get("", response_model=List[SessionInfo])
async def list_sessions():
    """
    List all active sessions.
    """
    service = get_pipeline_service()
    return service.session_manager.list_sessions()


@router.post("", response_model=SessionCreateResponse)
async def create_session(request: SessionCreateRequest = None):
    """
    Create a new pipeline session.
    """
    service = get_pipeline_service()
    session = service.create_session()

    return SessionCreateResponse(
        id=session.id,
        created_at=session.created_at,
    )


@router.get("/{session_id}", response_model=SessionInfo)
async def get_session(session_id: str):
    """
    Get session details by ID.
    """
    service = get_pipeline_service()
    session = service.get_session(session_id)

    if not session:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")

    return session.to_info()


@router.delete("/{session_id}", response_model=APIResponse)
async def delete_session(session_id: str):
    """
    Delete a session.
    """
    service = get_pipeline_service()

    if not service.session_manager.delete_session(session_id):
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")

    return APIResponse(
        success=True,
        data={"deleted": session_id},
    )


@router.post("/{session_id}/cancel", response_model=APIResponse)
async def cancel_session(session_id: str):
    """
    Cancel the current operation in a session.
    """
    service = get_pipeline_service()
    session = service.get_session(session_id)

    if not session:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")

    session.cancel()

    return APIResponse(
        success=True,
        data={"cancelled": session_id},
    )
