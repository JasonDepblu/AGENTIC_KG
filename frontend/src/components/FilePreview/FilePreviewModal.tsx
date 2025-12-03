import { X, Download, Copy, Check } from 'lucide-react';
import { useState } from 'react';
import { useFiles } from '../../hooks/useFiles';

export function FilePreviewModal() {
  const { previewFile, previewContent, isPreviewLoading, closePreview } = useFiles();
  const [copied, setCopied] = useState(false);

  if (!previewFile) return null;

  const handleCopy = async () => {
    if (previewContent) {
      await navigator.clipboard.writeText(previewContent);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  const handleDownload = () => {
    if (previewContent) {
      const blob = new Blob([previewContent], { type: 'text/plain' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = previewFile.name;
      a.click();
      URL.revokeObjectURL(url);
    }
  };

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
      <div className="bg-bg-secondary rounded-lg w-full max-w-4xl max-h-[80vh] flex flex-col shadow-2xl">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-border">
          <div>
            <h3 className="text-text-primary font-medium">{previewFile.name}</h3>
            <p className="text-text-secondary text-sm">{previewFile.path}</p>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={handleCopy}
              className="p-2 rounded hover:bg-bg-hover text-text-secondary hover:text-text-primary transition-colors"
              title="Copy content"
            >
              {copied ? <Check size={18} className="text-green-400" /> : <Copy size={18} />}
            </button>
            <button
              onClick={handleDownload}
              className="p-2 rounded hover:bg-bg-hover text-text-secondary hover:text-text-primary transition-colors"
              title="Download"
            >
              <Download size={18} />
            </button>
            <button
              onClick={closePreview}
              className="p-2 rounded hover:bg-bg-hover text-text-secondary hover:text-text-primary transition-colors"
              title="Close"
            >
              <X size={18} />
            </button>
          </div>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-auto p-4">
          {isPreviewLoading ? (
            <div className="text-text-secondary">Loading...</div>
          ) : (
            <pre className="text-sm text-text-primary font-mono whitespace-pre-wrap break-all">
              {previewContent}
            </pre>
          )}
        </div>
      </div>
    </div>
  );
}
