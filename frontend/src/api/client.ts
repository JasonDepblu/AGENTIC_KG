import type { FileInfo, FileTree, Session } from '../types';

// Use direct backend URL in development, proxy in production
const API_BASE = import.meta.env.DEV ? 'http://localhost:8000/api' : '/api';

// Generic API request helper
async function request<T>(
  endpoint: string,
  options: RequestInit = {}
): Promise<T> {
  const response = await fetch(`${API_BASE}${endpoint}`, {
    headers: {
      'Content-Type': 'application/json',
      ...options.headers,
    },
    ...options,
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Request failed' }));
    throw new Error(error.detail || 'Request failed');
  }

  return response.json();
}

// File API
export const filesApi = {
  async list(path: string = '', recursive: boolean = false): Promise<{ files: FileInfo[]; total: number }> {
    const params = new URLSearchParams();
    if (path) params.set('path', path);
    if (recursive) params.set('recursive', 'true');
    return request(`/files?${params}`);
  },

  async getTree(): Promise<FileTree> {
    return request('/files/tree');
  },

  async getContent(path: string, maxLines: number = 100): Promise<{ path: string; content: string; lines: number; truncated: boolean }> {
    return request(`/files/content/${encodeURIComponent(path)}?max_lines=${maxLines}`);
  },

  async upload(file: File, directory: string = ''): Promise<{ path: string; size: number; message: string }> {
    const formData = new FormData();
    formData.append('file', file);

    const params = directory ? `?directory=${encodeURIComponent(directory)}` : '';

    const response = await fetch(`${API_BASE}/files/upload${params}`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: 'Upload failed' }));
      throw new Error(error.detail || 'Upload failed');
    }

    return response.json();
  },

  async delete(path: string): Promise<void> {
    await request(`/files/${encodeURIComponent(path)}`, { method: 'DELETE' });
  },
};

// Session API
export const sessionsApi = {
  async list(): Promise<Session[]> {
    return request('/sessions');
  },

  async create(): Promise<{ id: string; created_at: string }> {
    return request('/sessions', { method: 'POST' });
  },

  async get(id: string): Promise<Session> {
    return request(`/sessions/${id}`);
  },

  async delete(id: string): Promise<void> {
    await request(`/sessions/${id}`, { method: 'DELETE' });
  },

  async cancel(id: string): Promise<void> {
    await request(`/sessions/${id}/cancel`, { method: 'POST' });
  },
};

// WebSocket helper
export function createChatWebSocket(sessionId: string): WebSocket {
  // Use direct backend URL in development
  if (import.meta.env.DEV) {
    return new WebSocket(`ws://localhost:8000/api/chat/${sessionId}`);
  }
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const host = window.location.host;
  return new WebSocket(`${protocol}//${host}/api/chat/${sessionId}`);
}
