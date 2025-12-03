import { useEffect, useRef } from 'react';
import { FolderOpen, Upload, RefreshCw, Trash2 } from 'lucide-react';
import { useFiles } from '../../hooks/useFiles';
import { FileTree } from './FileTree';

export function Sidebar() {
  const {
    tree,
    isLoading,
    error,
    selectedFiles,
    loadTree,
    clearSelection,
    uploadFile,
    deleteFile,
  } = useFiles();

  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    loadTree();
  }, [loadTree]);

  const handleUploadClick = () => {
    fileInputRef.current?.click();
  };

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files?.length) return;

    for (const file of Array.from(files)) {
      try {
        await uploadFile(file);
      } catch (err) {
        console.error('Upload failed:', err);
      }
    }

    // Reset input
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleDeleteSelected = async () => {
    if (!selectedFiles.length) return;

    if (confirm(`Delete ${selectedFiles.length} file(s)?`)) {
      for (const path of selectedFiles) {
        try {
          await deleteFile(path);
        } catch (err) {
          console.error('Delete failed:', err);
        }
      }
    }
  };

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="p-4 border-b border-border">
        <div className="flex items-center gap-2 text-text-primary">
          <FolderOpen size={20} />
          <span className="font-semibold">Data Files</span>
        </div>
      </div>

      {/* Toolbar */}
      <div className="flex items-center gap-1 p-2 border-b border-border">
        <button
          onClick={handleUploadClick}
          className="p-2 rounded hover:bg-bg-hover text-text-secondary hover:text-text-primary transition-colors"
          title="Upload file"
        >
          <Upload size={16} />
        </button>
        <button
          onClick={() => loadTree()}
          className="p-2 rounded hover:bg-bg-hover text-text-secondary hover:text-text-primary transition-colors"
          title="Refresh"
          disabled={isLoading}
        >
          <RefreshCw size={16} className={isLoading ? 'animate-spin' : ''} />
        </button>
        {selectedFiles.length > 0 && (
          <button
            onClick={handleDeleteSelected}
            className="p-2 rounded hover:bg-bg-hover text-red-400 hover:text-red-300 transition-colors ml-auto"
            title={`Delete ${selectedFiles.length} file(s)`}
          >
            <Trash2 size={16} />
          </button>
        )}
        <input
          ref={fileInputRef}
          type="file"
          multiple
          onChange={handleFileChange}
          className="hidden"
          accept=".csv,.json,.txt"
        />
      </div>

      {/* File tree */}
      <div className="flex-1 overflow-y-auto p-2">
        {error ? (
          <div className="text-red-400 text-sm p-2">{error}</div>
        ) : isLoading && !tree ? (
          <div className="text-text-secondary text-sm p-2">Loading...</div>
        ) : tree ? (
          <FileTree node={tree} level={0} />
        ) : (
          <div className="text-text-secondary text-sm p-2">No files found</div>
        )}
      </div>

      {/* Selected files indicator */}
      {selectedFiles.length > 0 && (
        <div className="p-3 border-t border-border bg-bg-secondary">
          <div className="flex items-center justify-between">
            <span className="text-sm text-text-secondary">
              {selectedFiles.length} file(s) selected
            </span>
            <button
              onClick={clearSelection}
              className="text-xs text-accent hover:text-accent-hover"
            >
              Clear
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
