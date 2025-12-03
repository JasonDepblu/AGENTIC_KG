# Agentic KG

A multi-agent system for building knowledge graphs from structured and unstructured data using Google ADK (Agent Development Kit). Features a ChatGPT-like web interface for interactive knowledge graph construction, natural language querying, and visualization.

## Overview

Agentic KG orchestrates multiple AI agents to guide users through the complete knowledge graph lifecycle:

### Pipeline Phases

1. **User Intent** - Captures and clarifies user goals for the knowledge graph
2. **File Suggestion** - Identifies and recommends relevant data files
3. **Data Preprocessing** - Validates, transforms, and preprocesses data (ETL)
4. **Schema Proposal** - Proposes graph schema with iterative refinement
5. **Construction** - Executes the construction plan in Neo4j
6. **Query** - Natural language querying of the constructed knowledge graph

### Agent Architecture

| Agent | Purpose | Key Tools |
|-------|---------|-----------|
| User Intent Agent | Capture and clarify user goals | `set_perceived_user_goal`, `approve_perceived_user_goal` |
| File Suggestion Agent | Identify relevant data files | `list_available_files`, `sample_file`, `set_suggested_files` |
| Data Preprocessing Agent | ETL operations on data files | `analyze_survey_format`, `extract_entities`, `extract_ratings` |
| Schema Proposal Agent | Design graph schema | `propose_node_construction`, `propose_relationship_construction` |
| Schema Critic Agent | Review and validate schemas | Evaluates schema quality and completeness |
| KG Builder Agent | Execute Neo4j construction | `import_nodes`, `import_relationships` |
| KG Query Agent | Natural language graph queries | `find_best_stores`, `search_entities`, `query_graph_cypher` |
| NER Agent | Named entity recognition for unstructured text | `set_proposed_entities`, `approve_proposed_entities` |
| Fact Type Agent | Relationship discovery for unstructured text | `add_proposed_fact`, `approve_proposed_facts` |

## System Architecture

```
                                    ┌─────────────────────────────────┐
                                    │         Web Interface           │
                                    │    (React + TypeScript + Vite)  │
                                    └───────────────┬─────────────────┘
                                                    │ WebSocket
                                    ┌───────────────┴─────────────────┐
                                    │         FastAPI Backend          │
                                    │      (Pipeline Orchestrator)     │
                                    └───────────────┬─────────────────┘
                                                    │
                    ┌───────────────────────────────┼───────────────────────────────┐
                    │                               │                               │
            ┌───────┴───────┐               ┌───────┴───────┐               ┌───────┴───────┐
            │  Structured   │               │ Unstructured  │               │   Query &     │
            │  Data Path    │               │  Data Path    │               │  Analytics    │
            └───────┬───────┘               └───────┬───────┘               └───────┬───────┘
                    │                               │                               │
    ┌───────────────┼───────────────┐               │                               │
    ▼               ▼               ▼               ▼                               ▼
┌────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐                   ┌──────────┐
│  User  │    │   File   │    │  Data    │    │   NER    │                   │   KG     │
│ Intent │───▶│Suggestion│───▶│Preprocess│    │  Agent   │                   │  Query   │
│ Agent  │    │  Agent   │    │  Agent   │    └────┬─────┘                   │  Agent   │
└────────┘    └──────────┘    └────┬─────┘         │                         └──────────┘
                                   │               ▼
                                   │         ┌──────────┐
                                   │         │Fact Type │
                                   │         │  Agent   │
                                   │         └────┬─────┘
                                   │              │
                                   ▼              ▼
                            ┌─────────────────────────────┐
                            │   Schema Refinement Loop    │
                            │  ┌──────────┐ ┌──────────┐  │
                            │  │ Proposal │◀│  Critic  │  │
                            │  │  Agent   │▶│  Agent   │  │
                            │  └──────────┘ └──────────┘  │
                            └──────────────┬──────────────┘
                                           │
                                           ▼
                                    ┌──────────────┐
                                    │  KG Builder  │
                                    │    Agent     │
                                    └──────┬───────┘
                                           │
                                           ▼
                                    ┌──────────────┐
                                    │    Neo4j     │
                                    │   Database   │
                                    └──────────────┘
```

## Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              Pipeline State Flow                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐                    │
│  │   User       │     │   Files      │     │  Preprocessing│                    │
│  │   Intent     │────▶│   Selected   │────▶│   Results     │                    │
│  │   Captured   │     │              │     │               │                    │
│  └──────────────┘     └──────────────┘     └───────┬───────┘                    │
│                                                    │                            │
│  State Keys:                                       │                            │
│  • approved_user_goal                              │                            │
│  • approved_files                                  ▼                            │
│  • preprocessing_complete              ┌──────────────────┐                    │
│  • proposed_construction_plan          │  Schema Design   │                    │
│  • approved_construction_plan          │  & Approval      │                    │
│  • construction_complete               └────────┬─────────┘                    │
│                                                  │                              │
│                                                  ▼                              │
│                                        ┌──────────────────┐                    │
│                                        │  KG Construction │                    │
│                                        │  in Neo4j        │                    │
│                                        └────────┬─────────┘                    │
│                                                  │                              │
│                                                  ▼                              │
│                                        ┌──────────────────┐                    │
│                                        │  Query Phase     │                    │
│                                        │  (GraphRAG)      │◀─── User Questions │
│                                        └──────────────────┘                    │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
Agentic_KG/
├── src/                              # Core source code
│   ├── config.py                     # Unified configuration management
│   ├── llm.py                        # LLM interface (DashScope/OpenAI compatible)
│   ├── neo4j_client.py               # Neo4j database client
│   ├── embed_model.py                # Embedding model adapters
│   │
│   ├── tools/                        # Tool functions for agents
│   │   ├── common.py                 # Common utilities (tool_success, tool_error)
│   │   ├── user_intent.py            # User goal capture tools
│   │   ├── file_suggestion.py        # File management tools
│   │   ├── data_preprocessing.py     # ETL tools (extract, transform, normalize)
│   │   ├── kg_construction.py        # Graph construction tools
│   │   ├── kg_query.py               # Graph query tools (NEW)
│   │   ├── kg_extraction.py          # GraphRAG extraction tools
│   │   └── unstructured_extraction.py # NER and fact type tools
│   │
│   └── agents/                       # Agent definitions
│       ├── base.py                   # AgentCaller base class
│       ├── user_intent_agent.py      # User goal agent
│       ├── file_suggestion_agent.py  # File selection agent
│       ├── data_preprocessing_agent.py # ETL agent
│       ├── schema_proposal_agent.py  # Schema design agent + critic
│       ├── kg_builder_agent.py       # Neo4j construction agent
│       ├── kg_query_agent.py         # Knowledge graph query agent (NEW)
│       ├── ner_agent.py              # Named entity recognition agent
│       ├── fact_type_agent.py        # Relationship discovery agent
│       └── unstructured_data_agent.py # Unstructured data orchestrator
│
├── api/                              # FastAPI backend
│   ├── main.py                       # FastAPI app entry point
│   ├── routes/
│   │   ├── files.py                  # File management endpoints
│   │   ├── chat.py                   # WebSocket chat endpoint
│   │   ├── sessions.py               # Session management
│   │   └── graph.py                  # Graph visualization API
│   ├── services/
│   │   ├── pipeline.py               # Pipeline orchestrator with streaming
│   │   └── file_manager.py           # File operations
│   └── models/
│       └── schemas.py                # Pydantic models (includes PipelinePhase)
│
├── frontend/                         # React frontend
│   ├── src/
│   │   ├── components/
│   │   │   ├── Chat/                 # Chat interface components
│   │   │   ├── Sidebar/              # File browser
│   │   │   └── Graph/                # Graph visualization
│   │   ├── hooks/                    # Custom React hooks
│   │   ├── stores/                   # Zustand state stores
│   │   └── api/                      # API client
│   └── package.json
│
├── reference/                        # Reference notebooks from course
│   ├── schema_proposal_structured.ipynb
│   ├── schema_proposal_unstructured.ipynb
│   ├── kg_construction_1.ipynb
│   └── kg_construction_2.ipynb
│
├── data/                             # Sample data files
├── main.py                           # CLI entry point
├── docker-compose.yml                # Neo4j Docker configuration
└── requirements.txt                  # Python dependencies
```

## Quick Start

### 1. Prerequisites

- Python 3.10+
- Node.js 18+ (for web interface)
- Docker (for Neo4j)
- DashScope API key (or other OpenAI-compatible LLM)

### 2. Installation

```bash
# Clone the repository
git clone https://github.com/your-username/Agentic_KG.git
cd Agentic_KG

# Create virtual environment
python -m venv venv_adk
source venv_adk/bin/activate  # On Windows: venv_adk\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your settings
# Required:
#   - DASHSCOPE_API_KEY
#   - NEO4J_PASSWORD
#   - NEO4J_IMPORT_DIR (path to your data files)
```

### 4. Start Neo4j

```bash
docker compose up -d
```

### 5. Run the Application

**Option A: Web Interface (Recommended)**

```bash
# Terminal 1: Start the backend API
source venv_adk/bin/activate
uvicorn api.main:app --reload --port 8000

# Terminal 2: Start the frontend
cd frontend
npm install  # First time only
npm run dev
```

Open http://localhost:5173 in your browser.

**Option B: Command Line**

```bash
# Test connections
python main.py --test-connection

# Run demonstration
python main.py --demo

# Interactive mode
python main.py --interactive
```

## Features

### Web Interface

The web interface provides a ChatGPT-like experience for building knowledge graphs.

- **Chat Interface**: Interactive conversation with AI agents
- **File Browser**: Browse and select data files from the import directory
- **Phase Indicator**: Visual feedback showing the current pipeline phase
- **Graph Visualization**: Interactive force-directed graph view with filtering

### Data Preprocessing (ETL)

The Data Preprocessing Agent handles various data transformations:

| Tool | Purpose |
|------|---------|
| `analyze_survey_format` | Detect survey/questionnaire format patterns |
| `classify_columns` | Classify columns as ID, entity, rating, opinion, etc. |
| `normalize_values` | Handle special values (N/A, -, etc.) |
| `split_multi_value_column` | Split comma-separated values into rows |
| `extract_entities` | Extract entity values from columns |
| `extract_ratings` | Extract numeric ratings with metadata |
| `extract_opinion_pairs` | Extract subject-opinion pairs |

### Knowledge Graph Query (GraphRAG)

After construction, the KG Query Agent enables natural language queries:

```
User: "Which store has the best ratings?"
Agent: Uses find_best_stores() tool to query Neo4j and summarize results

User: "What do customers think about service quality?"
Agent: Uses analyze_aspect_sentiment() to aggregate sentiment data
```

Available query tools:
- `get_graph_schema` - Get node labels and relationship types
- `get_graph_statistics` - Get counts and metrics
- `find_best_stores` - Rank entities by ratings/opinions
- `find_store_opinions` - Search opinions with filters
- `search_entities` - Full-text entity search
- `analyze_aspect_sentiment` - Sentiment analysis by aspect
- `query_graph_cypher` - Custom Cypher queries

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check |
| `/api/files` | GET | List files in import directory |
| `/api/files/upload` | POST | Upload file |
| `/api/sessions` | GET/POST | Manage chat sessions |
| `/api/chat/{session_id}` | WebSocket | Real-time chat |
| `/api/graph/schema` | GET | Get graph schema |
| `/api/graph/stats` | GET | Get node/relationship counts |
| `/api/graph/sample` | GET | Get sample graph data |
| `/api/graph/filter-options/{label}` | GET | Get filter options for a label |
| `/api/graph/by-center-node/{node_id}` | GET | Get graph centered on a node |

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DASHSCOPE_API_KEY` | DashScope API key | Required |
| `DASHSCOPE_BASE_URL` | API base URL | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| `LLM_MODEL` | Default LLM model | `qwen-plus-latest` |
| `EMBEDDING_MODEL` | Embedding model | `text-embedding-v3` |
| `NEO4J_URI` | Neo4j connection URI | `bolt://localhost:7687` |
| `NEO4J_USERNAME` | Neo4j username | `neo4j` |
| `NEO4J_PASSWORD` | Neo4j password | Required |
| `NEO4J_DATABASE` | Neo4j database name | `neo4j` |
| `NEO4J_IMPORT_DIR` | Path to import directory | Required for file access |

### Supported LLM Models

The system uses DashScope's OpenAI-compatible API:

- `qwen-plus-latest` (default, recommended for stability)
- `qwen3-235b-a22b-instruct-2507` (larger model)
- `Moonshot-Kimi-K2-Instruct`

## Usage Examples

### Interactive Web Flow

1. Start the application
2. Describe your goal: "I want to analyze customer feedback about car dealerships"
3. Review and approve suggested files
4. Let the preprocessing agent transform the data
5. Review and approve the proposed schema
6. Build the knowledge graph
7. Ask questions: "Which dealership has the best service ratings?"

### Programmatic Usage

```python
import asyncio
from src.agents import create_kg_query_agent, make_agent_caller

async def query_knowledge_graph():
    # Create query agent
    agent = create_kg_query_agent()
    caller = await make_agent_caller(agent, {
        "construction_complete": True  # Assumes KG exists
    })

    # Ask questions
    response = await caller.call("Which store has the best ratings?")
    print(response)

asyncio.run(query_knowledge_graph())
```

### Using Individual Agents

```python
from src.agents import create_user_intent_agent, make_agent_caller

async def capture_user_intent():
    agent = create_user_intent_agent()
    caller = await make_agent_caller(agent)

    await caller.call("I want to build a customer feedback analysis graph")
    await caller.call("Approve that goal")

    session = await caller.get_session()
    print(f"Approved goal: {session.state['approved_user_goal']}")
```

## Key Concepts

### Pipeline Phases

```python
class PipelinePhase(str, Enum):
    IDLE = "idle"
    USER_INTENT = "user_intent"
    FILE_SUGGESTION = "file_suggestion"
    DATA_PREPROCESSING = "data_preprocessing"
    SCHEMA_PROPOSAL = "schema_proposal"
    CONSTRUCTION = "construction"
    QUERY = "query"        # NEW: GraphRAG query phase
    COMPLETE = "complete"
    ERROR = "error"
```

### Tool Context State

Agents share information through a session state dictionary:

| State Key | Set By | Used By |
|-----------|--------|---------|
| `approved_user_goal` | User Intent Agent | All subsequent agents |
| `approved_files` | File Suggestion Agent | Preprocessing, Schema agents |
| `preprocessing_complete` | Preprocessing Agent | Schema agent |
| `proposed_construction_plan` | Schema Agent | Critic, Builder |
| `approved_construction_plan` | Schema Agent | Builder |
| `construction_complete` | Builder Agent | Query Agent |

### Propose-Approve Pattern

The system uses a collaborative pattern:
1. Agent proposes a solution
2. User reviews the proposal
3. User approves or provides feedback
4. Approved items are stored for subsequent agents

## Development

### Adding New Agents

1. Create agent file in `src/agents/`
2. Define instruction prompt
3. Specify required tools
4. Export from `src/agents/__init__.py`

### Adding New Tools

1. Create tool function in appropriate `src/tools/` module
2. Use `tool_success()` and `tool_error()` for responses
3. Access state via `tool_context.state`
4. Export from `src/tools/__init__.py`

### Testing

```bash
# Test connections
python main.py --test-connection

# Run with verbose output
python main.py --demo --verbose
```

## Troubleshooting

### Rate Limiting

If you encounter rate limit errors:
- The system uses `qwen-plus-latest` by default which has looser limits
- Rate limits typically recover within 1 minute
- Consider using batch API for non-real-time tasks

### Connection Issues

```bash
# Test all connections
python main.py --test-connection
```

### Neo4j Issues

```bash
# Check Docker logs
docker logs neo4j

# Restart container
docker compose restart neo4j
```

## License

MIT License - See LICENSE file for details.

## Acknowledgments

- [Google ADK](https://github.com/google/adk-python) - Agent Development Kit
- [Neo4j](https://neo4j.com/) - Graph Database
- [Neo4j GraphRAG](https://github.com/neo4j/neo4j-graphrag-python) - GraphRAG Library
- [LiteLLM](https://github.com/BerriAI/litellm) - LLM Abstraction Layer
- [DashScope](https://dashscope.aliyun.com/) - LLM API Provider
