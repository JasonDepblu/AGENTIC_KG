# Agentic KG

A multi-agent system for building knowledge graphs from structured and unstructured data using Google ADK (Agent Development Kit). Features a ChatGPT-like web interface for interactive knowledge graph construction, natural language querying, and 3D visualization.

## Features

- **Schema-First Design**: First design what to extract (schema), then execute extraction (preprocessing), with bidirectional feedback support
- **Multi-Agent Pipeline**: Orchestrates 25+ specialized AI agents for each phase of KG construction
- **Interactive Web Interface**: ChatGPT-like chat UI with real-time streaming updates
- **3D Graph Visualization**: Interactive force-directed graph visualization
- **Session Persistence**: Refresh page without losing progress, with ability to start new chats
- **Natural Language Queries**: Query your knowledge graph using plain language (GraphRAG)
- **Automatic Rollback**: Preprocessing can rollback to schema design when issues are detected

---

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Web Interface (React/TypeScript)                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐ │
│  │ Chat UI      │  │ Phase        │  │ File Browser │  │ Graph            │ │
│  │ (Messages)   │  │ Indicator    │  │ (Sidebar)    │  │ Visualization    │ │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────────┘ │
│                              │ WebSocket (Real-time streaming)              │
└──────────────────────────────┼──────────────────────────────────────────────┘
                               │
┌──────────────────────────────┼──────────────────────────────────────────────┐
│                      FastAPI Backend (Port 8000)                            │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                    Pipeline Service (Orchestrator)                     │ │
│  │  Session Management │ Phase Control │ Event Streaming │ State Storage  │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────┼──────────────────────────────────────────────┘
                               │
┌──────────────────────────────┼──────────────────────────────────────────────┐
│                        Agent Layer (Google ADK)                             │
│                                                                             │
│  Phase 1         Phase 2           Phase 3: Schema-Preprocess Coordinator  │
│  ┌─────────┐    ┌─────────────┐   ┌─────────────────────────────────────┐   │
│  │ User    │───►│ File        │──►│  ┌─────────────────┐                │   │
│  │ Intent  │    │ Suggestion  │   │  │ Schema Design   │                │   │
│  │ Agent   │    │ Agent       │   │  │ Loop + Critic   │                │   │
│  └─────────┘    └─────────────┘   │  └────────┬────────┘                │   │
│                                   │           ↕ rollback                │   │
│                                   │  ┌────────┴────────┐                │   │
│                                   │  │ Preprocessing   │                │   │
│                                   │  │ Loop + Critic   │                │   │
│                                   │  └─────────────────┘                │   │
│                                   └──────────────┬──────────────────────┘   │
│                                                  │                          │
│  Phase 4                    Phase 5              │                          │
│  ┌─────────────┐           ┌─────────────────────┐                          │
│  │ KG Builder  │◄──────────┤ KG Query Agent      │◄─────────────────────────┘
│  │ Agent       │           │ ┌─────────────────┐ │                          │
│  └──────┬──────┘           │ │ Cypher Generator│ │                          │
│         │                  │ │ Cypher Validator│ │                          │
│         │                  │ │ Cypher Loop     │ │                          │
│         │                  │ └─────────────────┘ │                          │
│         │                  └─────────────────────┘                          │
└─────────┼───────────────────────────────────────────────────────────────────┘
          │
┌─────────┼────────────────────────────────────────────────────────────────────┐
│         ▼                  External Services                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                    │
│  │   Neo4j      │    │  DashScope   │    │ Silicon Cloud│                    │
│  │   Database   │    │  LLM (Qwen)  │    │ (Cypher Gen) │                    │
│  └──────────────┘    └──────────────┘    └──────────────┘                    │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Pipeline Phases & Data Flow

### Pipeline Modes

The system supports multiple pipeline modes:

| Mode | Flow | Status |
|------|------|--------|
| **Schema-First (Default)** | Intent → Files → Schema → Extract → Build → Query | **Active** |
| **Schema-First + Cleaning** | Intent → Files → Clean → Schema → Extract → Build → Query | Reserved |
| **Legacy** | Intent → Files → Preprocess → Schema → Build → Query | Deprecated |

> **Note**: The DATA_CLEANING phase is fully implemented (`data_cleaning_agent.py`, `data_cleaning.py`)
> but currently **skipped** in the default Schema-First mode. The Schema Design Agent performs basic
> data analysis internally using `sample_raw_file_structure()`. The cleaning phase can be enabled
> for scenarios requiring extensive data quality improvement before schema design.

### Default Pipeline Flow (Schema-First)

```
USER_INTENT ──► FILE_SUGGESTION ──► SCHEMA_PREPROCESS_COORDINATOR ──► CONSTRUCTION ──► QUERY
                                              │
                                    ┌─────────┴─────────┐
                                    │                   │
                              SCHEMA_DESIGN ◄──────► TARGETED_PREPROCESSING
                                    │                   │
                                    └───── rollback ◄───┘
```

### Phase Details

| Phase | Agent | Description | Key Tools |
|-------|-------|-------------|-----------|
| **USER_INTENT** | `user_intent_agent` | Captures and clarifies user goals for the knowledge graph | `set_perceived_user_goal`, `approve_perceived_user_goal` |
| **FILE_SUGGESTION** | `file_suggestion_agent` | Identifies and recommends relevant data files | `list_available_files`, `sample_file`, `approve_suggested_files` |
| **SCHEMA_DESIGN** | `schema_design_agent` + critic | Designs target schema with nodes, relationships, extraction hints | `propose_node_type`, `propose_relationship_type`, `approve_target_schema` |
| **TARGETED_PREPROCESSING** | `targeted_preprocessing_agent` + critic | Extracts entities and relationships based on schema | `extract_entities_for_node`, `extract_relationship_data`, `save_extracted_data` |
| **CONSTRUCTION** | `kg_builder_agent` | Executes the construction plan in Neo4j | `construct_domain_graph`, `load_nodes_from_csv` |
| **QUERY** | `kg_query_agent` | Natural language querying of the knowledge graph (GraphRAG) | `propose_cypher_query`, `execute_and_validate_query` |

#### Reserved Phase (Not Active by Default)

| Phase | Agent | Description | Key Tools |
|-------|-------|-------------|-----------|
| **DATA_CLEANING** | `data_cleaning_agent` | Removes meaningless columns/rows, validates data quality | `analyze_file_quality`, `detect_column_types`, `clean_file` |

### Data Flow Diagram

```
                          ┌─────────────────┐
                          │   User Input    │
                          │ (Goal + Files)  │
                          └────────┬────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │      SCHEMA DESIGN          │
                    │  • Analyze file structure   │
                    │  • Detect entities          │
                    │  • Define nodes + hints     │
                    │  • Define relationships     │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │  TARGETED PREPROCESSING     │
                    │  • Extract entities by hint │
                    │  • Extract relationships    │
                    │  • LLM text extraction      │
                    │  • Entity deduplication     │
                    │  • Save to CSV files        │
                    └──────────────┬──────────────┘
                                   │
         ┌─────────────────────────▼─────────────────────────┐
         │                 CONSTRUCTION                       │
         │  ┌─────────────────────────────────────────────┐  │
         │  │  data/extracted_data/                       │  │
         │  │  ├── Respondent_entities.csv               │  │
         │  │  ├── Brand_entities.csv                    │  │
         │  │  ├── RATES_relationships.csv               │  │
         │  │  └── ...                                    │  │
         │  └─────────────────────────────────────────────┘  │
         │                        │                          │
         │                        ▼                          │
         │  ┌─────────────────────────────────────────────┐  │
         │  │              Neo4j Database                 │  │
         │  │  • Create uniqueness constraints            │  │
         │  │  • Batch load nodes (1000/transaction)      │  │
         │  │  • Create relationships                     │  │
         │  └─────────────────────────────────────────────┘  │
         └───────────────────────────────────────────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │          QUERY              │
                    │  • Natural language input   │
                    │  • Cypher generation        │
                    │  • Query execution          │
                    │  • Result formatting        │
                    └─────────────────────────────┘
```

---

## Tech Stack

### Backend
| Technology | Purpose |
|------------|---------|
| Python 3.10+ | Core language |
| FastAPI | REST API + WebSocket server |
| Google ADK | Agent Development Kit for multi-agent orchestration |
| Neo4j | Graph database |
| DashScope | LLM API (Alibaba Qwen models) |
| Silicon Cloud | Secondary LLM (Qwen3-Coder for Cypher) |
| Pydantic | Data validation |
| LiteLLM | LLM abstraction layer with rate limiting |
| Pandas | Data processing |

### Frontend
| Technology | Purpose |
|------------|---------|
| React 18 | UI framework |
| TypeScript | Type safety |
| Vite | Build tool |
| Zustand | State management |
| TailwindCSS | Styling |
| react-force-graph-3d | 3D graph visualization |
| Lucide React | Icons |

---

## Project Structure

```
Agentic_KG/
├── api/                              # FastAPI backend
│   ├── main.py                       # FastAPI app with CORS configuration
│   ├── routes/
│   │   ├── chat.py                   # WebSocket chat endpoint
│   │   ├── files.py                  # File listing and preview
│   │   ├── sessions.py               # Session management
│   │   └── graph.py                  # Graph visualization API
│   ├── services/
│   │   └── pipeline.py               # Pipeline orchestrator with streaming
│   └── models/
│       └── schemas.py                # Pydantic request/response models
│
├── src/                              # Core agent and tool logic
│   ├── config.py                     # Configuration management
│   ├── llm.py                        # LLM interface with rate limiting
│   ├── neo4j_client.py               # Neo4j database client
│   │
│   ├── models/
│   │   └── target_schema.py          # TargetSchema, NodeDefinition, RelationshipDefinition
│   │
│   ├── agents/                       # 25+ Agent definitions (Google ADK)
│   │   ├── base.py                   # AgentCaller wrapper class
│   │   ├── user_intent_agent.py      # Phase 1: User goal capture
│   │   ├── file_suggestion_agent.py  # Phase 2: File selection
│   │   ├── schema_design_agent.py    # Phase 3a: Schema design + critic + loop
│   │   ├── targeted_preprocessing_agent.py  # Phase 3b: Data extraction + critic + loop
│   │   ├── schema_preprocess_coordinator.py # Phase 3: Super coordinator with rollback
│   │   ├── kg_builder_agent.py       # Phase 4: Neo4j construction
│   │   ├── kg_query_agent.py         # Phase 5: Natural language queries
│   │   ├── data_cleaning_agent.py    # Reserved: Data quality improvement
│   │   ├── cypher_generator_agent.py # Cypher query generation
│   │   ├── cypher_validator_agent.py # Cypher validation
│   │   ├── cypher_loop_agent.py      # Query refinement loop
│   │   ├── survey_*.py               # Survey-specific agents
│   │   └── ...                       # More specialized agents
│   │
│   └── tools/                        # Tool functions for agents (~13,000 lines)
│       ├── common.py                 # Utility functions
│       ├── user_intent.py            # User goal tools
│       ├── file_suggestion.py        # File management tools
│       ├── data_cleaning.py          # Data cleaning tools
│       ├── schema_design.py          # Schema design tools
│       ├── targeted_preprocessing.py # Entity/relationship extraction
│       ├── kg_construction.py        # Graph construction tools
│       ├── kg_query.py               # Graph query tools
│       ├── kg_extraction.py          # Text extraction tools
│       ├── cypher_validation.py      # Cypher validation tools
│       └── survey_*.py               # Survey-specific tools
│
├── frontend/                         # React frontend
│   ├── src/
│   │   ├── components/
│   │   │   ├── Chat/                 # Chat interface
│   │   │   │   ├── ChatContainer.tsx # Main chat container
│   │   │   │   ├── MessageList.tsx   # Message list with streaming
│   │   │   │   ├── Message.tsx       # Individual message component
│   │   │   │   ├── InputBar.tsx      # User input
│   │   │   │   └── PhaseIndicator.tsx# Pipeline phase display
│   │   │   ├── Graph/
│   │   │   │   └── GraphVisualization.tsx  # 3D force graph
│   │   │   └── Sidebar/
│   │   │       └── FileTree.tsx      # File browser
│   │   ├── hooks/
│   │   │   ├── useWebSocket.ts       # WebSocket with session persistence
│   │   │   └── useFiles.ts           # File operations
│   │   ├── stores/
│   │   │   ├── chatStore.ts          # Chat message state
│   │   │   └── fileStore.ts          # File browser state
│   │   ├── types/
│   │   │   └── index.ts              # TypeScript definitions
│   │   └── api/
│   │       └── client.ts             # REST API client
│   ├── package.json
│   └── tsconfig.json
│
├── data/                             # Data files for KG construction
│   └── extracted_data/               # Generated entity/relationship CSVs
│
├── main.py                           # CLI entry point
├── test_pipeline.py                  # Pipeline smoke test
├── docker-compose.yml                # Neo4j Docker configuration
├── requirements.txt                  # Python dependencies
└── .env                              # Environment variables
```

---

## Agent Inventory

### Core Pipeline Agents (Default Flow)

| Agent | Phase | Purpose |
|-------|-------|---------|
| `user_intent_agent` | 1 | Capture and clarify user goals |
| `file_suggestion_agent` | 2 | File discovery and selection |
| `schema_design_agent` | 3a | Design target schema |
| `targeted_preprocessing_agent` | 3b | Extract entities/relationships |
| `schema_preprocess_coordinator` | 3 | Orchestrate schema + preprocessing with rollback |
| `kg_builder_agent` | 4 | Execute Neo4j import |
| `kg_query_agent` | 5 | Natural language queries |

### Reserved Agents (Not Active by Default)

| Agent | Purpose | Status |
|-------|---------|--------|
| `data_cleaning_agent` | Data quality improvement before schema design | Reserved for future use |

### Query Refinement Agents

| Agent | Purpose |
|-------|---------|
| `cypher_generator_agent` | Generate Cypher from natural language |
| `cypher_validator_agent` | Validate and execute Cypher |
| `cypher_loop_agent` | Query refinement loop |

### Specialized Agents

| Agent | Purpose |
|-------|---------|
| `survey_preprocessing_coordinator` | Survey-specific data processing |
| `survey_column_classifier_agent` | Classify survey column types |
| `survey_ner_agent` | Extract entities from surveys |
| `survey_rating_agent` | Extract ratings from surveys |
| `ner_agent` | Named entity recognition |
| `unstructured_data_agent` | Unstructured text processing |

---

## Extraction Hints

### Entity Extraction Types

| source_type | Description | Example |
|-------------|-------------|---------|
| `entity_selection` | Extract unique values from column | Brand names, Store names |
| `column_header` | Extract entity names from headers | Aspect names from rating columns |
| `text_extraction` | LLM-based text entity extraction | Entities from open-ended responses |

### Relationship Extraction Types

| source_type | Description | Example |
|-------------|-------------|---------|
| `rating_column` | Extract ratings from columns | RATES (Respondent → Aspect) |
| `entity_reference` | Extract entity selections per row | EVALUATED (Respondent → Brand) |
| `foreign_key` | Extract via foreign key columns | BELONGS_TO (Model → Brand) |

### Example Schema Definition

```python
{
  "nodes": [
    {
      "label": "Respondent",
      "unique_property": "respondent_id",
      "properties": ["respondent_id", "name"],
      "extraction_hints": {
        "source_type": "entity_selection",
        "column_pattern": "序号"
      }
    },
    {
      "label": "Aspect",
      "unique_property": "aspect_id",
      "properties": ["aspect_id", "name"],
      "extraction_hints": {
        "source_type": "column_header",
        "column_pattern": "打多少分|评分",
        "name_regex": "[\""]([^\"\"]+)[\""]"
      }
    }
  ],
  "relationships": [
    {
      "type": "RATES",
      "from_node": "Respondent",
      "to_node": "Aspect",
      "properties": ["score"],
      "extraction_hints": {
        "source_type": "rating_column",
        "column_pattern": "打多少分"
      }
    }
  ]
}
```

---

## Frontend Architecture

### State Management (Zustand)

```typescript
// Chat Store State
interface ChatStore {
  sessionId: string | null;
  phase: PipelinePhase;
  sessionState: Record<string, unknown>;
  messages: ChatMessage[];
  isConnected: boolean;
  isLoading: boolean;
}
```

### WebSocket Communication

The frontend maintains a persistent WebSocket connection with session management:

```typescript
// Message types from server
type MessageType =
  | 'connected'       // Session initialized
  | 'phase_change'    // Pipeline phase transition
  | 'agent_event'     // Agent execution (streaming)
  | 'phase_complete'  // Phase finished
  | 'error';          // Error occurred

// Message types to server
{ type: 'message', content: string }  // User message
{ type: 'approve', phase: string }    // Approve phase result
{ type: 'cancel' }                    // Cancel current operation
```

### Session Persistence

Sessions are persisted using localStorage:

- **Page Refresh**: Automatically reconnects to previous session
- **New Chat**: Button to clear session and start fresh
- **Auto-reconnect**: Reconnects after disconnection (3s delay)

---

## API Endpoints

### Chat Routes
| Method | Endpoint | Description |
|--------|----------|-------------|
| WebSocket | `/chat/{session_id}` | Real-time chat with agents |

### File Routes
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/files` | List files in import directory |
| GET | `/api/files/{path}` | Get file content preview |
| POST | `/api/files/upload` | Upload new file |

### Session Routes
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/sessions` | Create new session |
| GET | `/api/sessions/{id}` | Get session info |
| GET | `/api/sessions` | List all sessions |

### Graph Routes
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/graph/schema` | Get node labels and relationship types |
| GET | `/api/graph/stats` | Graph statistics |
| GET | `/api/graph/sample` | Get full graph sample |
| GET | `/api/graph/nodes` | Fetch nodes for visualization |
| GET | `/api/graph/export` | Export graph as ZIP |
| DELETE | `/api/graph/clear` | Clear all graph data |

---

## Schema-First Pipeline

### Schema Design Loop

```
┌────────────────────────────────────────────────────────────────┐
│              LoopAgent: schema_design_loop                     │
│              max_iterations: 4                                 │
├────────────────────────────────────────────────────────────────┤
│  ┌──────────────────────┐                                      │
│  │ schema_design_agent  │  ◄── Designs nodes and relationships │
│  │     (LlmAgent)       │      Calls propose_node_type, etc.   │
│  └──────────┬───────────┘                                      │
│             │                                                  │
│             ▼                                                  │
│  ┌──────────────────────┐                                      │
│  │ schema_design_critic │  ◄── Validates schema quality        │
│  │     (LlmAgent)       │                                      │
│  └──────────┬───────────┘                                      │
│             │                                                  │
│             ▼ "valid" or "retry"                               │
│  ┌──────────────────────┐                                      │
│  │ CheckSchemaStatus    │  ◄── Checks if schema approved       │
│  └──────────────────────┘                                      │
└────────────────────────────────────────────────────────────────┘
```

### Targeted Preprocessing Loop

```
┌────────────────────────────────────────────────────────────────┐
│           LoopAgent: preprocessing_sub_loop                    │
│           max_iterations: 3                                    │
├────────────────────────────────────────────────────────────────┤
│  ┌──────────────────────────┐                                  │
│  │ targeted_preprocessing   │  ◄── Extracts entities/relations │
│  │     _agent (LlmAgent)    │      Based on schema hints       │
│  └──────────┬───────────────┘                                  │
│             │                                                  │
│             ▼                                                  │
│  ┌──────────────────────────┐                                  │
│  │ preprocessing_critic     │  ◄── Validates extraction        │
│  │     (LlmAgent)           │      quality                     │
│  └──────────┬───────────────┘                                  │
│             │                                                  │
│             ▼                                                  │
│  ┌──────────────────────────┐                                  │
│  │ CheckPreprocessingStatus │  ◄── May trigger rollback        │
│  └──────────────────────────┘                                  │
└────────────────────────────────────────────────────────────────┘
```

### Rollback Mechanism

When preprocessing encounters issues, it can request schema revision:

```
Preprocessing Agent detects issue
         │
         ▼ request_schema_revision(reason, suggested_changes)
         │
         ▼ Sets NEEDS_SCHEMA_REVISION = true
         │
CheckCoordinatorStatus detects rollback
         │
         ▼ Clears approved schema, resets preprocessing state
         │
         ▼ Next iteration → Schema Design Agent
         │
         ▼ get_schema_revision_reason() → sees rollback context
         │
         ▼ Addresses suggested_changes in redesign
```

---

## LLM Configuration

### Primary LLM: DashScope (Alibaba Qwen)
- **API Base**: `https://dashscope.aliyuncs.com/compatible-mode/v1`
- **Default Model**: `qwen-plus-latest`
- **Flash Model**: `qwen-flash` (fast, cheap)
- **Embedding**: `text-embedding-v3`

### Secondary LLM: Silicon Cloud
- **API Base**: `https://api.siliconflow.cn/v1`
- **Model**: `Qwen/Qwen3-Coder-480B-A35B-Instruct` (Cypher generation)

### Rate Limit Protection
- Max retries: 5
- Base delay: 5 seconds
- Max delay: 120 seconds
- Fallback models: qwen-plus-latest → qwen-turbo-latest

---

## Quick Start

### 1. Prerequisites

- Python 3.10+
- Node.js 18+ (for web interface)
- Docker (for Neo4j)
- DashScope API key

### 2. Installation

```bash
# Clone the repository
git clone https://github.com/your-username/Agentic_KG.git
cd Agentic_KG

# Create virtual environment
python -m venv venv_adk
source venv_adk/bin/activate  # Windows: venv_adk\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt

# Install frontend dependencies
cd frontend
npm install
cd ..
```

### 3. Configuration

Create `.env` file:

```bash
# Required
DASHSCOPE_API_KEY=your_api_key_here
NEO4J_PASSWORD=your_password_here
NEO4J_IMPORT_DIR=/path/to/your/data

# Optional
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
LLM_MODEL=qwen-plus-latest
SILICON_API_KEY=your_silicon_key_here
```

### 4. Start Services

```bash
# Start Neo4j
docker compose up -d

# Terminal 1: Start backend API
source venv_adk/bin/activate
uvicorn api.main:app --reload --port 8000

# Terminal 2: Start frontend
cd frontend
npm run dev
```

### 5. Access

Open http://localhost:5173 in your browser.

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DASHSCOPE_API_KEY` | DashScope API key | Required |
| `DASHSCOPE_BASE_URL` | API base URL | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| `LLM_MODEL` | Default LLM model | `qwen-plus-latest` |
| `NEO4J_URI` | Neo4j connection URI | `bolt://localhost:7687` |
| `NEO4J_USERNAME` | Neo4j username | `neo4j` |
| `NEO4J_PASSWORD` | Neo4j password | Required |
| `NEO4J_IMPORT_DIR` | Path to import directory | Required |
| `SILICON_API_KEY` | Silicon Cloud API key | Optional |
| `TEXT_EXTRACTION_ENABLE_DEDUP` | Enable entity deduplication | `true` |

---

## State Management

### Key Session State Variables

| State Key | Set By | Used By | Description |
|-----------|--------|---------|-------------|
| `approved_user_goal` | User Intent Agent | All agents | User's approved goal |
| `approved_files` | File Suggestion Agent | All agents | Selected data files |
| `data_cleaning_complete` | Data Cleaning Agent | Coordinator | Cleaning phase done |
| `target_schema` | Schema Design Agent | Critic, Preprocessing | Current schema design |
| `approved_target_schema` | Schema Design Agent | Preprocessing | User-approved schema |
| `schema_design_feedback` | Schema Critic | Design Agent | Validation result |
| `targeted_extraction_results` | Preprocessing Agent | Critic, Builder | Extracted entities |
| `targeted_preprocessing_complete` | Preprocessing Agent | Coordinator | Preprocessing done |
| `needs_schema_revision` | Preprocessing Agent | Coordinator | Rollback trigger |
| `construction_rules` | Preprocessing Agent | Builder | Import instructions |

---

## CLI Usage

```bash
# Test connections (Neo4j, LLM)
python main.py --test-connection

# Run demonstration
python main.py --demo

# Interactive mode
python main.py --interactive

# Verbose logging
python main.py --verbose
```

---

## Troubleshooting

### Rate Limiting Errors

The system includes built-in retry with exponential backoff:
- Max retries: 5
- Base delay: 5 seconds
- Max delay: 120 seconds

### Schema Design Loop Issues

If the schema design loop keeps retrying:
- Check extraction hints use correct `source_type`
- Ensure column patterns match actual column names
- Review critic feedback for specific issues

### Extraction Returns 0 Results

If entity/relationship extraction returns empty:
- Verify column names match `column_pattern`
- Check `source_type` is appropriate for data type
- Use `request_schema_revision()` to rollback and fix schema

### WebSocket Connection Issues

If the frontend shows "disconnected":
- Check backend is running on port 8000
- Verify no CORS issues in browser console
- Session will auto-reconnect after 3 seconds

---

## Code Statistics

| Component | Lines of Code |
|-----------|---------------|
| Agent Code (`src/agents/`) | ~6,000 |
| Tool Code (`src/tools/`) | ~13,000 |
| API Code (`api/`) | ~2,000 |
| Frontend Code (`frontend/src/`) | ~3,000 |
| **Total** | **~24,000** |

---

## License

MIT License - See LICENSE file for details.

## Acknowledgments

- [Google ADK](https://github.com/google/adk-python) - Agent Development Kit
- [Neo4j](https://neo4j.com/) - Graph Database
- [LiteLLM](https://github.com/BerriAI/litellm) - LLM Abstraction Layer
- [DashScope](https://dashscope.aliyun.com/) - LLM API Provider (Alibaba Qwen)
- [React Force Graph](https://github.com/vasturiano/react-force-graph) - 3D Graph Visualization
