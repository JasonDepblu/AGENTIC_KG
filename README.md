# Agentic KG

A multi-agent system for building knowledge graphs from structured and unstructured data using Google ADK (Agent Development Kit).

## Overview

Agentic KG orchestrates multiple AI agents to guide users through the complete knowledge graph construction process:

1. **User Intent Agent** - Captures and clarifies user goals for the knowledge graph
2. **File Suggestion Agent** - Identifies and recommends relevant data files
3. **Schema Proposal Agent** - Proposes graph schema with node and relationship definitions
4. **Schema Critic Agent** - Reviews and validates proposed schemas
5. **KG Builder Agent** - Executes the construction plan in Neo4j

## Architecture

```
Agentic_KG/
├── src/                         # Core source code
│   ├── config.py               # Unified configuration management
│   ├── llm.py                  # LLM interface (DashScope/OpenAI compatible)
│   ├── neo4j_client.py         # Neo4j database client
│   ├── tools/                  # Tool functions for agents
│   │   ├── common.py           # Common utilities
│   │   ├── user_intent.py      # User goal tools
│   │   ├── file_suggestion.py  # File management tools
│   │   └── kg_construction.py  # Graph construction tools
│   ├── agents/                 # Agent definitions
│   │   ├── base.py             # AgentCaller base class
│   │   ├── user_intent_agent.py
│   │   ├── file_suggestion_agent.py
│   │   ├── schema_proposal_agent.py
│   │   └── kg_builder_agent.py
│   └── pipelines/              # Workflow orchestration
│       └── kg_pipeline.py      # Complete KG construction pipeline
├── data/                       # Sample data files
├── main.py                     # CLI entry point
├── docker-compose.yml          # Neo4j Docker configuration
└── requirements.txt            # Python dependencies
```

## Quick Start

### 1. Prerequisites

- Python 3.10+
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
```

### 4. Start Neo4j

```bash
docker compose up -d
```

### 5. Run the Application

```bash
# Test connections
python main.py --test-connection

# Run demonstration
python main.py --demo

# Interactive mode
python main.py --interactive
```

## Usage

### Interactive Mode

```bash
python main.py --interactive
```

Follow the prompts to:
1. Describe your knowledge graph goal
2. Review and approve suggested files
3. Review and approve the proposed schema
4. Build the knowledge graph

### Programmatic Usage

```python
import asyncio
from src.pipelines import KGPipeline

async def main():
    pipeline = KGPipeline(verbose=True)

    result = await pipeline.run_full_pipeline(
        user_goal="I want a supply chain graph for bill of materials analysis"
    )

    print(f"Construction complete: {result}")

asyncio.run(main())
```

### Using Individual Agents

```python
from src.agents import create_user_intent_agent, make_agent_caller

async def capture_user_intent():
    agent = create_user_intent_agent()
    caller = await make_agent_caller(agent)

    await caller.call("I want to build a social network graph")
    await caller.call("Approve that goal")

    session = await caller.get_session()
    print(f"Approved goal: {session.state['approved_user_goal']}")
```

## Configuration Options

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DASHSCOPE_API_KEY` | DashScope API key | Required |
| `DASHSCOPE_BASE_URL` | API base URL | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| `NEO4J_URI` | Neo4j connection URI | `bolt://localhost:7687` |
| `NEO4J_USERNAME` | Neo4j username | `neo4j` |
| `NEO4J_PASSWORD` | Neo4j password | Required |
| `NEO4J_DATABASE` | Neo4j database name | `neo4j` |
| `NEO4J_IMPORT_DIR` | Path to import directory | Optional |

### Supported LLM Models

The system uses DashScope's OpenAI-compatible API with these models:

- `Moonshot-Kimi-K2-Instruct` (default)
- `qwen3-235b-a22b-instruct-2507`
- `qwen-plus-latest`

## Agent Workflow

```
┌─────────────────┐
│  User Intent    │ → Captures user's KG goals
│     Agent       │
└────────┬────────┘
         ↓
┌─────────────────┐
│ File Suggestion │ → Identifies relevant data files
│     Agent       │
└────────┬────────┘
         ↓
┌─────────────────────────────────┐
│      Schema Refinement Loop     │
│  ┌────────────┐  ┌────────────┐ │
│  │  Proposal  │→→│   Critic   │ │ → Iterative schema design
│  │   Agent    │←←│   Agent    │ │
│  └────────────┘  └────────────┘ │
└────────────────┬────────────────┘
                 ↓
┌─────────────────┐
│   KG Builder    │ → Executes construction in Neo4j
│     Agent       │
└─────────────────┘
```

## Key Concepts

### Tool Context State

Agents share information through a session state dictionary:

- `approved_user_goal` - User's validated graph objective
- `approved_files` - Files approved for import
- `proposed_construction_plan` - Schema rules (nodes & relationships)
- `approved_construction_plan` - Validated schema for execution

### Propose-Approve Pattern

The system uses a collaborative pattern where:
1. Agent proposes a solution (e.g., perceived goal, suggested files)
2. User reviews the proposal
3. User approves or provides feedback
4. Approved items are stored for subsequent agents

### Construction Plan Format

```python
{
    "Product": {
        "construction_type": "node",
        "source_file": "products.csv",
        "label": "Product",
        "unique_column_name": "product_id",
        "properties": ["product_name", "price", "description"]
    },
    "SUPPLIED_BY": {
        "construction_type": "relationship",
        "source_file": "part_supplier_mapping.csv",
        "relationship_type": "SUPPLIED_BY",
        "from_node_label": "Part",
        "from_node_column": "part_id",
        "to_node_label": "Supplier",
        "to_node_column": "supplier_id",
        "properties": ["lead_time_days", "unit_cost"]
    }
}
```

## Development

### Project Structure

```
src/
├── __init__.py          # Package init with version
├── config.py            # Configuration dataclasses
├── llm.py               # LLM factory functions
├── neo4j_client.py      # Neo4j wrapper with ADK-friendly responses
├── tools/               # Tool implementations
├── agents/              # Agent definitions
└── pipelines/           # Workflow orchestration
```

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

## Troubleshooting

### Connection Issues

```bash
# Test all connections
python main.py --test-connection
```

### Neo4j Not Starting

```bash
# Check Docker logs
docker logs neo4j

# Restart container
docker compose restart neo4j
```

### LLM Errors

- Verify `DASHSCOPE_API_KEY` is set correctly
- Check API quota and rate limits
- Try a different model if one fails

## License

MIT License - See LICENSE file for details.

## Acknowledgments

- [Google ADK](https://github.com/google/adk-python) - Agent Development Kit
- [Neo4j](https://neo4j.com/) - Graph Database
- [LiteLLM](https://github.com/BerriAI/litellm) - LLM Abstraction Layer
