import { useEffect, useRef, useState, useCallback } from 'react';
import ForceGraph2D, { ForceGraphMethods, NodeObject } from 'react-force-graph-2d';
import { X, RefreshCw, ZoomIn, ZoomOut, Maximize2, ChevronDown, Trash2, Download } from 'lucide-react';

interface GraphNode {
  id: string;
  label: string;
  name: string;
  properties: Record<string, unknown>;
  isCenter?: boolean;
}

interface GraphLink {
  source: string;
  target: string;
  type: string;
  properties: Record<string, unknown>;
}

interface GraphData {
  nodes: GraphNode[];
  links: GraphLink[];
  node_count: number;
  link_count: number;
}

interface GraphStats {
  total_nodes: number;
  total_relationships: number;
}

interface GraphSchema {
  labels: string[];
  relationship_types: string[];
  node_counts: Record<string, number>;
  relationship_counts: Record<string, number>;
}

interface FilterOption {
  id: string;
  name: string;
  properties?: Record<string, unknown>;
}

// Color palette for different node labels
const LABEL_COLORS: Record<string, string> = {
  // Survey data node types
  Respondent: '#10a37f',
  Aspect: '#f59e0b',
  Brand: '#3b82f6',
  Model: '#8b5cf6',
  Store: '#ec4899',
  Entity: '#14b8a6',
  // Legacy node types
  BrandPowertrain: '#10a37f',
  VehicleAttribute: '#3b82f6',
  Attribute: '#f97316',
  AttentionAttribute: '#ef4444',
  Product: '#f59e0b',
  Category: '#ef4444',
  Person: '#8b5cf6',
  Company: '#ec4899',
  Location: '#14b8a6',
  Event: '#f97316',
  Document: '#6366f1',
  default: '#6b7280',
};

function getNodeColor(label: string): string {
  return LABEL_COLORS[label] || LABEL_COLORS.default;
}

interface Props {
  isOpen: boolean;
  onClose: () => void;
}

export function GraphVisualization({ isOpen, onClose }: Props) {
  const graphRef = useRef<ForceGraphMethods>();
  const containerRef = useRef<HTMLDivElement>(null);
  const [graphData, setGraphData] = useState<GraphData | null>(null);
  const [stats, setStats] = useState<GraphStats | null>(null);
  const [schema, setSchema] = useState<GraphSchema | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null);
  const [dimensions, setDimensions] = useState({ width: 800, height: 600 });

  // Filter state
  const [selectedLabel, setSelectedLabel] = useState<string>('');
  const [filterOptions, setFilterOptions] = useState<FilterOption[]>([]);
  const [selectedFilterValue, setSelectedFilterValue] = useState<string>('__all__');
  const [labelDropdownOpen, setLabelDropdownOpen] = useState(false);
  const [valueDropdownOpen, setValueDropdownOpen] = useState(false);
  const [loadingOptions, setLoadingOptions] = useState(false);

  // Fetch schema on open
  const fetchSchema = useCallback(async () => {
    try {
      const [statsRes, schemaRes] = await Promise.all([
        fetch('/api/graph/stats'),
        fetch('/api/graph/schema'),
      ]);

      if (statsRes.ok && schemaRes.ok) {
        const [statsData, schemaData] = await Promise.all([
          statsRes.json(),
          schemaRes.json(),
        ]);
        setStats(statsData);
        setSchema(schemaData);

        // Auto-select first label with nodes
        const labelsWithNodes = schemaData.labels.filter(
          (label: string) => (schemaData.node_counts[label] || 0) > 0
        );
        if (labelsWithNodes.length > 0 && !selectedLabel) {
          setSelectedLabel(labelsWithNodes[0]);
        }
      }
    } catch (err) {
      console.error('Failed to fetch schema:', err);
    }
  }, [selectedLabel]);

  // Fetch filter options when label changes
  const fetchFilterOptions = useCallback(async (label: string) => {
    if (!label) return;

    setLoadingOptions(true);
    try {
      const res = await fetch(`/api/graph/filter-options/${encodeURIComponent(label)}`);
      if (res.ok) {
        const data = await res.json();
        setFilterOptions(data.options || []);
        // Reset to "All" when label changes
        setSelectedFilterValue('__all__');
      }
    } catch (err) {
      console.error('Failed to fetch filter options:', err);
    } finally {
      setLoadingOptions(false);
    }
  }, []);

  // Fetch graph data based on filter
  const fetchGraphData = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      let graphRes;

      if (selectedFilterValue === '__all__') {
        // Fetch all data (limited sample)
        graphRes = await fetch('/api/graph/sample?limit=200');
      } else {
        // Fetch data centered on selected node
        graphRes = await fetch(`/api/graph/by-center-node/${encodeURIComponent(selectedFilterValue)}`);
      }

      if (!graphRes.ok) {
        throw new Error('Failed to fetch graph data');
      }

      const graphDataResult = await graphRes.json();
      setGraphData(graphDataResult);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  }, [selectedFilterValue]);

  useEffect(() => {
    if (isOpen) {
      fetchSchema();
    }
  }, [isOpen, fetchSchema]);

  useEffect(() => {
    if (selectedLabel) {
      fetchFilterOptions(selectedLabel);
    }
  }, [selectedLabel, fetchFilterOptions]);

  useEffect(() => {
    if (isOpen && (selectedFilterValue || selectedLabel)) {
      fetchGraphData();
    }
  }, [isOpen, selectedFilterValue, fetchGraphData]);

  useEffect(() => {
    const updateDimensions = () => {
      if (containerRef.current) {
        const rect = containerRef.current.getBoundingClientRect();
        setDimensions({
          width: rect.width,
          height: rect.height - 180, // Account for header, filter bars, and legend
        });
      }
    };

    updateDimensions();
    window.addEventListener('resize', updateDimensions);
    return () => window.removeEventListener('resize', updateDimensions);
  }, [isOpen]);

  const handleZoomIn = () => {
    if (graphRef.current) {
      graphRef.current.zoom(graphRef.current.zoom() * 1.5, 300);
    }
  };

  const handleZoomOut = () => {
    if (graphRef.current) {
      graphRef.current.zoom(graphRef.current.zoom() / 1.5, 300);
    }
  };

  const handleFitView = () => {
    if (graphRef.current) {
      graphRef.current.zoomToFit(400, 50);
    }
  };

  const handleNodeClick = useCallback((node: NodeObject) => {
    setSelectedNode(node as unknown as GraphNode);
  }, []);

  const handleLabelSelect = (label: string) => {
    setSelectedLabel(label);
    setLabelDropdownOpen(false);
  };

  const handleValueSelect = (value: string) => {
    setSelectedFilterValue(value);
    setValueDropdownOpen(false);
  };

  const handleClearDatabase = async () => {
    if (!window.confirm('确定要清空数据库吗？此操作不可撤销。')) {
      return;
    }

    setLoading(true);
    try {
      const res = await fetch('/api/graph/clear', { method: 'DELETE' });
      if (res.ok) {
        // Refresh data after clearing
        setGraphData(null);
        setStats(null);
        setSchema(null);
        setSelectedLabel('');
        setFilterOptions([]);
        await fetchSchema();
      } else {
        const error = await res.json();
        alert(`清空失败: ${error.detail || '未知错误'}`);
      }
    } catch (err) {
      alert(`清空失败: ${err instanceof Error ? err.message : '未知错误'}`);
    } finally {
      setLoading(false);
    }
  };

  const handleDownloadData = async () => {
    try {
      const res = await fetch('/api/graph/export');
      if (!res.ok) {
        const error = await res.json();
        alert(`下载失败: ${error.detail || '未知错误'}`);
        return;
      }

      // Get filename from Content-Disposition header or use default
      const contentDisposition = res.headers.get('Content-Disposition');
      let filename = 'graph_export.zip';
      if (contentDisposition) {
        const match = contentDisposition.match(/filename="?([^"]+)"?/);
        if (match) {
          filename = match[1];
        }
      }

      // Download the file
      const blob = await res.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (err) {
      alert(`下载失败: ${err instanceof Error ? err.message : '未知错误'}`);
    }
  };

  // Get labels that have nodes
  const labelsWithNodes = schema?.labels.filter(
    label => (schema.node_counts[label] || 0) > 0
  ) || [];

  // Get selected option name
  const selectedOptionName = selectedFilterValue === '__all__'
    ? '全部'
    : filterOptions.find(opt => opt.id === selectedFilterValue)?.name || '选择节点';

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70">
      <div
        ref={containerRef}
        className="bg-bg-secondary rounded-lg shadow-xl w-[90vw] h-[85vh] flex flex-col"
      >
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-border">
          <div className="flex items-center gap-4">
            <h2 className="text-xl font-semibold text-text-primary">Knowledge Graph</h2>
            {stats && (
              <div className="flex gap-4 text-sm text-text-secondary">
                <span>总计: {stats.total_nodes} 节点</span>
                <span>{stats.total_relationships} 关系</span>
              </div>
            )}
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={fetchGraphData}
              className="p-2 rounded hover:bg-bg-primary transition-colors"
              title="刷新"
            >
              <RefreshCw className={`w-5 h-5 text-text-secondary ${loading ? 'animate-spin' : ''}`} />
            </button>
            <button
              onClick={handleZoomIn}
              className="p-2 rounded hover:bg-bg-primary transition-colors"
              title="放大"
            >
              <ZoomIn className="w-5 h-5 text-text-secondary" />
            </button>
            <button
              onClick={handleZoomOut}
              className="p-2 rounded hover:bg-bg-primary transition-colors"
              title="缩小"
            >
              <ZoomOut className="w-5 h-5 text-text-secondary" />
            </button>
            <button
              onClick={handleFitView}
              className="p-2 rounded hover:bg-bg-primary transition-colors"
              title="适应窗口"
            >
              <Maximize2 className="w-5 h-5 text-text-secondary" />
            </button>
            <button
              onClick={handleDownloadData}
              className="p-2 rounded hover:bg-green-900/30 transition-colors"
              title="下载图谱数据"
            >
              <Download className="w-5 h-5 text-green-400" />
            </button>
            <button
              onClick={handleClearDatabase}
              className="p-2 rounded hover:bg-red-900/30 transition-colors"
              title="清空数据库"
            >
              <Trash2 className="w-5 h-5 text-red-400" />
            </button>
            <button
              onClick={onClose}
              className="p-2 rounded hover:bg-bg-primary transition-colors ml-2"
            >
              <X className="w-5 h-5 text-text-secondary" />
            </button>
          </div>
        </div>

        {/* Filter Bar */}
        <div className="flex items-center gap-4 px-6 py-3 border-b border-border bg-bg-primary/50">
          {/* Label Selector */}
          <div className="flex items-center gap-2">
            <span className="text-sm text-text-secondary whitespace-nowrap">节点类型:</span>
            <div className="relative">
              <button
                onClick={() => setLabelDropdownOpen(!labelDropdownOpen)}
                className="flex items-center gap-2 px-4 py-2 bg-bg-primary border border-border rounded-lg hover:border-accent transition-colors min-w-[180px] justify-between"
              >
                <div className="flex items-center gap-2">
                  {selectedLabel && (
                    <div
                      className="w-3 h-3 rounded-full"
                      style={{ backgroundColor: getNodeColor(selectedLabel) }}
                    />
                  )}
                  <span className="text-text-primary truncate">
                    {selectedLabel || '选择类型'}
                  </span>
                </div>
                <ChevronDown className={`w-4 h-4 text-text-secondary transition-transform ${labelDropdownOpen ? 'rotate-180' : ''}`} />
              </button>
              {labelDropdownOpen && (
                <div className="absolute top-full left-0 mt-1 w-full bg-bg-secondary border border-border rounded-lg shadow-xl z-20 max-h-[300px] overflow-y-auto">
                  {labelsWithNodes.map((label) => (
                    <button
                      key={label}
                      onClick={() => handleLabelSelect(label)}
                      className={`w-full px-4 py-2 text-left hover:bg-bg-primary transition-colors flex items-center gap-2 ${
                        selectedLabel === label ? 'bg-accent/20 text-accent' : 'text-text-primary'
                      }`}
                    >
                      <div
                        className="w-3 h-3 rounded-full"
                        style={{ backgroundColor: getNodeColor(label) }}
                      />
                      <span>{label}</span>
                      <span className="text-text-secondary text-xs ml-auto">
                        ({schema?.node_counts[label] || 0})
                      </span>
                    </button>
                  ))}
                </div>
              )}
            </div>
          </div>

          {/* Value Selector */}
          <div className="flex items-center gap-2">
            <span className="text-sm text-text-secondary whitespace-nowrap">筛选节点:</span>
            <div className="relative">
              <button
                onClick={() => setValueDropdownOpen(!valueDropdownOpen)}
                disabled={loadingOptions}
                className="flex items-center gap-2 px-4 py-2 bg-bg-primary border border-border rounded-lg hover:border-accent transition-colors min-w-[240px] justify-between disabled:opacity-50"
              >
                <span className="text-text-primary truncate">
                  {loadingOptions ? '加载中...' : selectedOptionName}
                </span>
                <ChevronDown className={`w-4 h-4 text-text-secondary transition-transform ${valueDropdownOpen ? 'rotate-180' : ''}`} />
              </button>
              {valueDropdownOpen && !loadingOptions && (
                <div className="absolute top-full left-0 mt-1 w-full bg-bg-secondary border border-border rounded-lg shadow-xl z-20 max-h-[300px] overflow-y-auto">
                  {/* All option */}
                  <button
                    onClick={() => handleValueSelect('__all__')}
                    className={`w-full px-4 py-2 text-left hover:bg-bg-primary transition-colors font-medium ${
                      selectedFilterValue === '__all__' ? 'bg-accent/20 text-accent' : 'text-text-primary'
                    }`}
                  >
                    全部 (显示采样数据)
                  </button>
                  <div className="border-t border-border" />
                  {/* Individual options */}
                  {filterOptions.map((opt) => (
                    <button
                      key={opt.id}
                      onClick={() => handleValueSelect(opt.id)}
                      className={`w-full px-4 py-2 text-left hover:bg-bg-primary transition-colors ${
                        selectedFilterValue === opt.id ? 'bg-accent/20 text-accent' : 'text-text-primary'
                      }`}
                    >
                      {opt.name}
                    </button>
                  ))}
                </div>
              )}
            </div>
          </div>

          {/* Current display info */}
          {graphData && (
            <span className="text-sm text-text-secondary ml-auto">
              显示: {graphData.node_count} 节点, {graphData.link_count} 关系
            </span>
          )}
        </div>

        {/* Legend */}
        {schema && (
          <div className="flex items-center gap-4 px-6 py-2 border-b border-border overflow-x-auto">
            {labelsWithNodes.map((label) => (
              <div key={label} className="flex items-center gap-2 whitespace-nowrap">
                <div
                  className="w-3 h-3 rounded-full"
                  style={{ backgroundColor: getNodeColor(label) }}
                />
                <span className="text-sm text-text-secondary">
                  {label} ({schema.node_counts[label] || 0})
                </span>
              </div>
            ))}
          </div>
        )}

        {/* Graph Canvas */}
        <div className="flex-1 relative">
          {loading && (
            <div className="absolute inset-0 flex items-center justify-center bg-black/30 z-10">
              <div className="text-text-secondary">加载中...</div>
            </div>
          )}

          {error && (
            <div className="absolute inset-0 flex items-center justify-center">
              <div className="text-red-400">错误: {error}</div>
            </div>
          )}

          {graphData && graphData.nodes.length > 0 && (
            <ForceGraph2D
              ref={graphRef}
              width={dimensions.width}
              height={dimensions.height}
              graphData={graphData}
              nodeLabel={(node) => {
                const n = node as unknown as GraphNode;
                const score = n.properties?.score;
                return score ? `${n.label}: ${n.name} (score: ${score})` : `${n.label}: ${n.name}`;
              }}
              nodeColor={(node) => {
                const n = node as unknown as GraphNode;
                return getNodeColor(n.label);
              }}
              nodeRelSize={6}
              linkLabel={(link) => {
                const l = link as unknown as GraphLink;
                const score = l.properties?.score;
                return score ? `${l.type} (${score})` : l.type;
              }}
              linkColor={() => '#4b5563'}
              linkWidth={1.5}
              linkDirectionalArrowLength={4}
              linkDirectionalArrowRelPos={1}
              onNodeClick={handleNodeClick}
              backgroundColor="#212121"
              nodeCanvasObject={(node, ctx, globalScale) => {
                const n = node as unknown as GraphNode & { x: number; y: number };
                const label = n.name || n.id.slice(0, 8);
                const fontSize = 12 / globalScale;
                ctx.font = `${fontSize}px Sans-Serif`;

                // Determine node size based on selection state
                // - Center node (from specific filter): largest
                // - Nodes matching selected label type: medium (highlighted)
                // - Other nodes: small
                const isExactCenter = n.isCenter || (selectedFilterValue !== '__all__' && n.label === selectedLabel);
                const isSelectedType = n.label === selectedLabel;
                const nodeSize = isExactCenter ? 14 : (isSelectedType ? 10 : 5);

                // Draw node circle
                ctx.beginPath();
                ctx.arc(n.x, n.y, nodeSize, 0, 2 * Math.PI);
                ctx.fillStyle = getNodeColor(n.label);
                ctx.fill();

                // Draw border for highlighted nodes (center or selected type)
                if (isExactCenter || isSelectedType) {
                  ctx.strokeStyle = isExactCenter ? '#ffffff' : 'rgba(255,255,255,0.5)';
                  ctx.lineWidth = isExactCenter ? 2 : 1;
                  ctx.stroke();
                }

                // Draw label for highlighted nodes or when zoomed in
                if (isSelectedType || globalScale > 0.5) {
                  ctx.textAlign = 'center';
                  ctx.textBaseline = 'top';
                  ctx.fillStyle = isSelectedType ? '#ffffff' : '#ececec';
                  ctx.fillText(label, n.x, n.y + nodeSize + 2);
                }
              }}
            />
          )}

          {graphData && graphData.nodes.length === 0 && !loading && (
            <div className="absolute inset-0 flex items-center justify-center">
              <div className="text-text-secondary">无数据</div>
            </div>
          )}
        </div>

        {/* Selected Node Details */}
        {selectedNode && (
          <div className="border-t border-border p-4 bg-bg-primary">
            <div className="flex items-start justify-between">
              <div>
                <div className="flex items-center gap-2">
                  <div
                    className="w-3 h-3 rounded-full"
                    style={{ backgroundColor: getNodeColor(selectedNode.label) }}
                  />
                  <span className="font-semibold text-text-primary">{selectedNode.label}</span>
                  {selectedNode.isCenter && (
                    <span className="text-xs bg-accent/20 text-accent px-2 py-0.5 rounded">中心节点</span>
                  )}
                </div>
                <div className="text-sm text-text-secondary mt-1">{selectedNode.name}</div>
              </div>
              <button
                onClick={() => setSelectedNode(null)}
                className="p-1 hover:bg-bg-secondary rounded"
              >
                <X className="w-4 h-4 text-text-secondary" />
              </button>
            </div>
            {Object.keys(selectedNode.properties).length > 0 && (
              <div className="mt-2 grid grid-cols-2 gap-2 text-sm">
                {Object.entries(selectedNode.properties).map(([key, value]) => (
                  <div key={key} className="flex gap-2">
                    <span className="text-text-secondary">{key}:</span>
                    <span className="text-text-primary">{String(value)}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
