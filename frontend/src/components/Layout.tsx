import { ReactNode, useState } from 'react';
import { ChevronLeft, ChevronRight } from 'lucide-react';

interface LayoutProps {
  sidebar: ReactNode;
  children: ReactNode;
}

export function Layout({ sidebar, children }: LayoutProps) {
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);

  return (
    <div className="flex h-screen bg-bg-primary">
      {/* Sidebar */}
      <aside
        className={`
          flex-shrink-0 bg-bg-sidebar border-r border-border overflow-hidden flex flex-col
          transition-all duration-300 ease-in-out
          ${sidebarCollapsed ? 'w-0' : 'w-64'}
        `}
      >
        <div className={`w-64 h-full flex flex-col ${sidebarCollapsed ? 'invisible' : 'visible'}`}>
          {sidebar}
        </div>
      </aside>

      {/* Collapse toggle button */}
      <button
        onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
        className={`
          absolute z-10 top-1/2 -translate-y-1/2
          w-5 h-10 flex items-center justify-center
          bg-bg-secondary border border-border rounded-r-md
          hover:bg-bg-hover transition-colors
          ${sidebarCollapsed ? 'left-0' : 'left-64'}
          transition-all duration-300 ease-in-out
        `}
        title={sidebarCollapsed ? '展开侧边栏' : '折叠侧边栏'}
      >
        {sidebarCollapsed ? (
          <ChevronRight size={14} className="text-text-secondary" />
        ) : (
          <ChevronLeft size={14} className="text-text-secondary" />
        )}
      </button>

      {/* Main content */}
      <main className="flex-1 flex flex-col overflow-hidden">
        {children}
      </main>
    </div>
  );
}
