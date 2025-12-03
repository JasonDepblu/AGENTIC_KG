import { ChevronRight, ChevronDown, File, Folder, FolderOpen } from 'lucide-react';
import { useFiles } from '../../hooks/useFiles';
import type { FileTree as FileTreeType } from '../../types';

interface FileTreeProps {
  node: FileTreeType;
  level: number;
}

export function FileTree({ node, level }: FileTreeProps) {
  const {
    selectedFiles,
    expandedFolders,
    toggleFileSelection,
    toggleFolder,
    previewFileContent,
  } = useFiles();

  const isExpanded = expandedFolders.has(node.path);
  const isSelected = selectedFiles.includes(node.path);
  const paddingLeft = level * 12 + 8;

  const handleClick = () => {
    if (node.is_directory) {
      toggleFolder(node.path);
    } else {
      toggleFileSelection(node.path);
    }
  };

  const handleDoubleClick = () => {
    if (!node.is_directory) {
      previewFileContent({
        name: node.name,
        path: node.path,
        is_directory: false,
        size: node.size,
        extension: node.extension,
      });
    }
  };

  const getFileIcon = () => {
    if (node.is_directory) {
      return isExpanded ? (
        <FolderOpen size={16} className="text-yellow-500" />
      ) : (
        <Folder size={16} className="text-yellow-500" />
      );
    }

    // Color by extension
    const ext = node.extension?.toLowerCase();
    let colorClass = 'text-text-secondary';
    if (ext === '.csv') colorClass = 'text-green-400';
    else if (ext === '.json') colorClass = 'text-yellow-400';
    else if (ext === '.txt') colorClass = 'text-blue-400';

    return <File size={16} className={colorClass} />;
  };

  const formatSize = (bytes?: number) => {
    if (bytes === undefined) return '';
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  return (
    <div>
      <div
        onClick={handleClick}
        onDoubleClick={handleDoubleClick}
        className={`
          flex items-center gap-1 py-1 px-2 rounded cursor-pointer select-none
          ${isSelected ? 'bg-accent/20 text-accent' : 'hover:bg-bg-hover text-text-primary'}
        `}
        style={{ paddingLeft }}
      >
        {/* Expand/collapse indicator */}
        {node.is_directory ? (
          <span className="w-4 flex-shrink-0">
            {isExpanded ? (
              <ChevronDown size={14} className="text-text-secondary" />
            ) : (
              <ChevronRight size={14} className="text-text-secondary" />
            )}
          </span>
        ) : (
          <span className="w-4 flex-shrink-0" />
        )}

        {/* Icon */}
        {getFileIcon()}

        {/* Name */}
        <span className="flex-1 truncate text-sm">{node.name}</span>

        {/* Size for files */}
        {!node.is_directory && node.size !== undefined && (
          <span className="text-xs text-text-secondary flex-shrink-0">
            {formatSize(node.size)}
          </span>
        )}
      </div>

      {/* Children */}
      {node.is_directory && isExpanded && node.children && (
        <div>
          {node.children.map((child) => (
            <FileTree key={child.path} node={child} level={level + 1} />
          ))}
        </div>
      )}
    </div>
  );
}
