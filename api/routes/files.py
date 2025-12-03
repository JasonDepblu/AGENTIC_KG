"""
File management API routes.
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Query
from typing import Optional

from ..services.file_manager import FileManager
from ..models.schemas import (
    FileListResponse,
    FileContentResponse,
    FileUploadResponse,
    APIResponse,
)

router = APIRouter(prefix="/files", tags=["files"])

# Singleton file manager
_file_manager: Optional[FileManager] = None


def get_file_manager() -> FileManager:
    """Get or create file manager instance."""
    global _file_manager
    if _file_manager is None:
        _file_manager = FileManager()
    return _file_manager


@router.get("", response_model=FileListResponse)
async def list_files(
    path: str = Query("", description="Directory path relative to import directory"),
    recursive: bool = Query(False, description="List all files recursively"),
):
    """
    List files in the import directory.

    - **path**: Subdirectory to list (empty for root)
    - **recursive**: If true, list all files recursively
    """
    try:
        fm = get_file_manager()
        if recursive:
            return fm.list_all_files()
        return fm.list_files(path)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/tree")
async def get_file_tree():
    """
    Get the file tree structure.

    Returns a nested structure representing the directory hierarchy.
    """
    try:
        fm = get_file_manager()
        return fm.get_tree()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/content/{file_path:path}", response_model=FileContentResponse)
async def get_file_content(
    file_path: str,
    max_lines: int = Query(100, ge=1, le=1000, description="Maximum lines to return"),
):
    """
    Get file content preview.

    - **file_path**: Path to the file relative to import directory
    - **max_lines**: Maximum number of lines to return (1-1000)
    """
    try:
        fm = get_file_manager()
        return await fm.get_file_content(file_path, max_lines=max_lines)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/upload", response_model=FileUploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    directory: str = Query("", description="Target directory"),
):
    """
    Upload a file to the import directory.

    - **file**: The file to upload
    - **directory**: Target subdirectory (empty for root)
    """
    try:
        fm = get_file_manager()
        content = await file.read()
        path, size = await fm.upload_file(file.filename, content, directory)

        return FileUploadResponse(
            path=path,
            size=size,
            message=f"File uploaded successfully: {path}",
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{file_path:path}", response_model=APIResponse)
async def delete_file(file_path: str):
    """
    Delete a file from the import directory.

    - **file_path**: Path to the file relative to import directory
    """
    try:
        fm = get_file_manager()
        fm.delete_file(file_path)
        return APIResponse(
            success=True,
            data={"deleted": file_path},
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
