"""
File management service for the API.

Handles file browsing, preview, upload, and deletion operations.
"""

import os
import aiofiles
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

from ..models.schemas import FileInfo, FileListResponse, FileContentResponse


class FileManager:
    """
    Service for managing files in the Neo4j import directory.
    """

    def __init__(self, import_dir: Optional[str] = None):
        """
        Initialize the file manager.

        Args:
            import_dir: Path to the import directory. Uses NEO4J_IMPORT_DIR env var if not provided.
        """
        self.import_dir = import_dir or os.getenv("NEO4J_IMPORT_DIR", "./data")
        self._base_path = Path(self.import_dir).resolve()

    def _resolve_path(self, relative_path: str) -> Tuple[Path, bool]:
        """
        Resolve a relative path to an absolute path within the import directory.

        Returns:
            Tuple of (resolved_path, is_valid)
        """
        if not relative_path or relative_path == ".":
            return self._base_path, True

        # Normalize and resolve the path
        target = (self._base_path / relative_path).resolve()

        # Security check: ensure path is within import directory
        try:
            target.relative_to(self._base_path)
            return target, True
        except ValueError:
            return target, False

    def _get_file_info(self, path: Path, relative_to: Path) -> FileInfo:
        """Get FileInfo for a path."""
        stat = path.stat()
        rel_path = str(path.relative_to(relative_to))

        return FileInfo(
            name=path.name,
            path=rel_path,
            is_directory=path.is_dir(),
            size=stat.st_size if path.is_file() else None,
            modified=datetime.fromtimestamp(stat.st_mtime),
            extension=path.suffix.lower() if path.is_file() else None,
        )

    def list_files(self, relative_path: str = "") -> FileListResponse:
        """
        List files in the specified directory.

        Args:
            relative_path: Path relative to import directory

        Returns:
            FileListResponse with file list
        """
        target, is_valid = self._resolve_path(relative_path)

        if not is_valid:
            raise ValueError(f"Invalid path: {relative_path}")

        if not target.exists():
            raise FileNotFoundError(f"Directory not found: {relative_path}")

        if not target.is_dir():
            raise ValueError(f"Not a directory: {relative_path}")

        files: List[FileInfo] = []

        for item in sorted(target.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())):
            # Skip hidden files
            if item.name.startswith("."):
                continue

            files.append(self._get_file_info(item, self._base_path))

        return FileListResponse(
            files=files,
            total=len(files),
            base_path=relative_path or ".",
        )

    def list_all_files(self) -> FileListResponse:
        """
        List all files recursively in the import directory.

        Returns:
            FileListResponse with all files
        """
        files: List[FileInfo] = []

        for item in self._base_path.rglob("*"):
            if item.is_file() and not any(p.startswith(".") for p in item.parts):
                files.append(self._get_file_info(item, self._base_path))

        # Sort by path
        files.sort(key=lambda x: x.path.lower())

        return FileListResponse(
            files=files,
            total=len(files),
            base_path=".",
        )

    async def get_file_content(
        self,
        relative_path: str,
        max_lines: int = 100,
        encoding: str = "utf-8"
    ) -> FileContentResponse:
        """
        Get the content of a file.

        Args:
            relative_path: Path to the file
            max_lines: Maximum number of lines to return
            encoding: File encoding

        Returns:
            FileContentResponse with content
        """
        target, is_valid = self._resolve_path(relative_path)

        if not is_valid:
            raise ValueError(f"Invalid path: {relative_path}")

        if not target.exists():
            raise FileNotFoundError(f"File not found: {relative_path}")

        if not target.is_file():
            raise ValueError(f"Not a file: {relative_path}")

        lines = []
        truncated = False

        async with aiofiles.open(target, "r", encoding=encoding) as f:
            line_count = 0
            async for line in f:
                if line_count >= max_lines:
                    truncated = True
                    break
                lines.append(line)
                line_count += 1

        content = "".join(lines)

        return FileContentResponse(
            path=relative_path,
            content=content,
            lines=len(lines),
            truncated=truncated,
        )

    async def upload_file(
        self,
        filename: str,
        content: bytes,
        relative_dir: str = ""
    ) -> Tuple[str, int]:
        """
        Upload a file to the import directory.

        Args:
            filename: Name of the file
            content: File content as bytes
            relative_dir: Directory to upload to (relative to import dir)

        Returns:
            Tuple of (relative_path, size)
        """
        # Validate filename
        if "/" in filename or "\\" in filename or filename.startswith("."):
            raise ValueError(f"Invalid filename: {filename}")

        target_dir, is_valid = self._resolve_path(relative_dir)
        if not is_valid:
            raise ValueError(f"Invalid directory: {relative_dir}")

        # Create directory if needed
        target_dir.mkdir(parents=True, exist_ok=True)

        target_file = target_dir / filename

        async with aiofiles.open(target_file, "wb") as f:
            await f.write(content)

        relative_path = str(target_file.relative_to(self._base_path))
        return relative_path, len(content)

    def delete_file(self, relative_path: str) -> bool:
        """
        Delete a file from the import directory.

        Args:
            relative_path: Path to the file

        Returns:
            True if deleted successfully
        """
        target, is_valid = self._resolve_path(relative_path)

        if not is_valid:
            raise ValueError(f"Invalid path: {relative_path}")

        if not target.exists():
            raise FileNotFoundError(f"File not found: {relative_path}")

        if target.is_dir():
            raise ValueError(f"Cannot delete directory: {relative_path}")

        target.unlink()
        return True

    def get_tree(self) -> dict:
        """
        Get the file tree structure.

        Returns:
            Nested dictionary representing the file tree
        """
        def build_tree(path: Path) -> dict:
            tree = {
                "name": path.name,
                "path": str(path.relative_to(self._base_path)) if path != self._base_path else "",
                "is_directory": path.is_dir(),
            }

            if path.is_dir():
                children = []
                for item in sorted(path.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())):
                    if not item.name.startswith("."):
                        children.append(build_tree(item))
                tree["children"] = children
            else:
                stat = path.stat()
                tree["size"] = stat.st_size
                tree["extension"] = path.suffix.lower()

            return tree

        return build_tree(self._base_path)
