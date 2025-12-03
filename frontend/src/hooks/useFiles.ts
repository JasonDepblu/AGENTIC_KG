import { useCallback } from 'react';
import { filesApi } from '../api/client';
import { useFileStore } from '../stores/fileStore';
import type { FileInfo } from '../types';

export function useFiles() {
  const {
    tree,
    isLoading,
    error,
    selectedFiles,
    previewFile,
    previewContent,
    isPreviewLoading,
    expandedFolders,
    setTree,
    setLoading,
    setError,
    toggleFileSelection,
    setSelectedFiles,
    clearSelection,
    setPreviewFile,
    setPreviewContent,
    setPreviewLoading,
    toggleFolder,
  } = useFileStore();

  const loadTree = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const treeData = await filesApi.getTree();
      setTree(treeData);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load files');
    } finally {
      setLoading(false);
    }
  }, [setTree, setLoading, setError]);

  const previewFileContent = useCallback(async (file: FileInfo) => {
    if (file.is_directory) return;

    setPreviewFile(file);
    setPreviewLoading(true);
    setPreviewContent(null);

    try {
      const result = await filesApi.getContent(file.path);
      setPreviewContent(result.content);
    } catch (err) {
      setPreviewContent(`Error loading file: ${err instanceof Error ? err.message : 'Unknown error'}`);
    } finally {
      setPreviewLoading(false);
    }
  }, [setPreviewFile, setPreviewContent, setPreviewLoading]);

  const closePreview = useCallback(() => {
    setPreviewFile(null);
    setPreviewContent(null);
  }, [setPreviewFile, setPreviewContent]);

  const uploadFile = useCallback(async (file: File, directory: string = '') => {
    try {
      await filesApi.upload(file, directory);
      await loadTree(); // Refresh tree after upload
    } catch (err) {
      throw err;
    }
  }, [loadTree]);

  const deleteFile = useCallback(async (path: string) => {
    try {
      await filesApi.delete(path);
      // Remove from selection if selected
      if (selectedFiles.includes(path)) {
        toggleFileSelection(path);
      }
      await loadTree(); // Refresh tree after delete
    } catch (err) {
      throw err;
    }
  }, [loadTree, selectedFiles, toggleFileSelection]);

  return {
    // State
    tree,
    isLoading,
    error,
    selectedFiles,
    previewFile,
    previewContent,
    isPreviewLoading,
    expandedFolders,

    // Actions
    loadTree,
    toggleFileSelection,
    setSelectedFiles,
    clearSelection,
    previewFileContent,
    closePreview,
    uploadFile,
    deleteFile,
    toggleFolder,
  };
}
