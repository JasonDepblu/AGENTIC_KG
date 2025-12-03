import { create } from 'zustand';
import type { FileTree, FileInfo } from '../types';

interface FileStore {
  // File tree
  tree: FileTree | null;
  isLoading: boolean;
  error: string | null;

  // Selected files
  selectedFiles: string[];

  // Preview
  previewFile: FileInfo | null;
  previewContent: string | null;
  isPreviewLoading: boolean;

  // Expanded folders
  expandedFolders: Set<string>;

  // Actions
  setTree: (tree: FileTree | null) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;

  toggleFileSelection: (path: string) => void;
  setSelectedFiles: (files: string[]) => void;
  clearSelection: () => void;

  setPreviewFile: (file: FileInfo | null) => void;
  setPreviewContent: (content: string | null) => void;
  setPreviewLoading: (loading: boolean) => void;

  toggleFolder: (path: string) => void;
  expandFolder: (path: string) => void;
  collapseFolder: (path: string) => void;
}

export const useFileStore = create<FileStore>((set) => ({
  // Initial state
  tree: null,
  isLoading: false,
  error: null,
  selectedFiles: [],
  previewFile: null,
  previewContent: null,
  isPreviewLoading: false,
  expandedFolders: new Set<string>(),

  // Actions
  setTree: (tree) => set({ tree }),
  setLoading: (loading) => set({ isLoading: loading }),
  setError: (error) => set({ error }),

  toggleFileSelection: (path) =>
    set((state) => {
      const newSelection = state.selectedFiles.includes(path)
        ? state.selectedFiles.filter((p) => p !== path)
        : [...state.selectedFiles, path];
      return { selectedFiles: newSelection };
    }),

  setSelectedFiles: (files) => set({ selectedFiles: files }),
  clearSelection: () => set({ selectedFiles: [] }),

  setPreviewFile: (file) => set({ previewFile: file }),
  setPreviewContent: (content) => set({ previewContent: content }),
  setPreviewLoading: (loading) => set({ isPreviewLoading: loading }),

  toggleFolder: (path) =>
    set((state) => {
      const newExpanded = new Set(state.expandedFolders);
      if (newExpanded.has(path)) {
        newExpanded.delete(path);
      } else {
        newExpanded.add(path);
      }
      return { expandedFolders: newExpanded };
    }),

  expandFolder: (path) =>
    set((state) => {
      const newExpanded = new Set(state.expandedFolders);
      newExpanded.add(path);
      return { expandedFolders: newExpanded };
    }),

  collapseFolder: (path) =>
    set((state) => {
      const newExpanded = new Set(state.expandedFolders);
      newExpanded.delete(path);
      return { expandedFolders: newExpanded };
    }),
}));
