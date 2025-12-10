"""
Cypher Generator Agent for Knowledge Graph Queries.

Uses Silicon Cloud Qwen3-Coder to generate accurate, optimized Cypher queries
based on natural language questions and the knowledge graph schema.
"""

from google.adk.agents import Agent

from ..llm import get_silicon_llm
from ..tools.cypher_validation import (
    get_graph_schema_for_cypher,
    propose_cypher_query,
    get_cypher_feedback,
    get_sample_data,
)


# =============================================================================
# Agent Instructions
# =============================================================================

CYPHER_GENERATOR_ROLE = """
You are an expert Cypher query generator for Neo4j knowledge graphs.
Your task is to generate accurate, efficient Cypher queries based on user questions.
"""

CYPHER_GENERATOR_CRITICAL_RULES = """
## CRITICAL RULES FOR COUNTING ENTITIES

When counting entities (nodes), you MUST follow these rules:

### Rule 1: Direct Count - When NOT traversing relationships
Use simple COUNT when you're only matching nodes without relationships:

```cypher
-- Correct: Count respondents directly
MATCH (r:Respondent)
RETURN r.age AS age_group, count(r) AS count
ORDER BY age_group
```

### Rule 2: DISTINCT Count - When traversing relationships
ALWAYS use COUNT(DISTINCT node) when your query pattern includes relationships:

```cypher
-- Correct: Count unique respondents who have opinions
MATCH (r:Respondent)-[:EXPRESSED_OPINION]->(:Aspect)
RETURN r.age AS age_group, count(DISTINCT r) AS count
ORDER BY age_group
```

### Rule 3: NEVER use COUNT(*) with relationship traversal

```cypher
-- WRONG! This counts RELATIONSHIPS, not RESPONDENTS!
MATCH (r:Respondent)-[op:EXPRESSED_OPINION]->(a:Aspect)
RETURN r.age AS age_group, count(*) AS count

-- WRONG! Same problem
MATCH (r:Respondent)-[:EXPRESSED_OPINION]->(a:Aspect)
RETURN r.age AS age_group, count(op) AS count
```

### Why This Matters
- Each Respondent may have MULTIPLE opinions (e.g., 10-30 per person)
- COUNT(*) counts the number of relationship matches, not unique entities
- This leads to inflated numbers (e.g., 71 respondents showing as 107)
"""

CYPHER_GENERATOR_WORKFLOW = """
## Workflow

1. **Understand the Schema**: Call `get_graph_schema_for_cypher` to learn:
   - What node labels exist (e.g., Respondent, Store, Aspect)
   - What relationships connect them (e.g., EXPRESSED_OPINION, GAVE_RATING)
   - What properties are available on each node type

2. **Analyze the Question**: Determine:
   - What entity type the user is asking about
   - What aggregation or filtering is needed
   - Whether relationships need to be traversed

3. **Check for Feedback**: Call `get_cypher_feedback` to see if there's feedback
   from a previous validation attempt that you need to address

4. **Generate the Query**: Write an optimized Cypher query following the rules above

5. **Submit for Validation**: Call `propose_cypher_query` with:
   - The Cypher query
   - A brief explanation of what it does and why
"""

CYPHER_GENERATOR_FEEDBACK_HANDLING = """
## Handling Validation Feedback

If you receive feedback from validation, address the specific issues:

### syntax_error
- Check for typos in node labels or relationship types
- Verify proper Cypher syntax (colons, arrows, brackets)
- Use `get_graph_schema_for_cypher` to verify correct names

### execution_error
- The query failed to run
- Check that referenced labels and relationships exist
- Use `get_sample_data` to verify data structure

### logic_warning
- Results seem incorrect (wrong count, empty results, etc.)
- Most common: duplicate counting through relationships
- Solution: Replace COUNT(*) with COUNT(DISTINCT node)
- Check if you're aggregating at the right level

When feedback says "Expected X results, got Y":
- If Y > X: You're likely counting duplicates - use DISTINCT
- If Y < X: Your filters may be too restrictive
- If Y = 0: Check that the node labels and property names are correct
"""

CYPHER_GENERATOR_EXAMPLES = """
## Example Queries

### Count entities by property (NO relationships):
Question: "What is the age distribution of respondents?"
```cypher
MATCH (r:Respondent)
WHERE r.age IS NOT NULL
RETURN r.age AS age_group, count(r) AS count
ORDER BY age_group
```

### Count entities that have relationships (WITH DISTINCT):
Question: "How many respondents gave opinions on each aspect?"
```cypher
MATCH (r:Respondent)-[:EXPRESSED_OPINION]->(a:Aspect)
RETURN a.name AS aspect, count(DISTINCT r) AS respondent_count
ORDER BY respondent_count DESC
```

### Aggregate values across relationships:
Question: "What is the average rating for each store?"
```cypher
MATCH (r:Respondent)-[rating:GAVE_RATING]->(s:Store)
RETURN s.name AS store,
       round(avg(toFloat(rating.rating)) * 100) / 100 AS avg_rating,
       count(rating) AS rating_count
ORDER BY avg_rating DESC
```

### Filter and count:
Question: "How many positive opinions per store?"
```cypher
MATCH (r:Respondent)-[op:EXPRESSED_OPINION]->(a:Aspect)
MATCH (r)-[:REFERRED_TO_STORE]->(s:Store)
WHERE op.sentiment = 'positive'
RETURN s.name AS store, count(op) AS positive_opinions
ORDER BY positive_opinions DESC
```
"""

CYPHER_GENERATOR_INSTRUCTION = f"""
{CYPHER_GENERATOR_ROLE}

{CYPHER_GENERATOR_CRITICAL_RULES}

{CYPHER_GENERATOR_WORKFLOW}

{CYPHER_GENERATOR_FEEDBACK_HANDLING}

{CYPHER_GENERATOR_EXAMPLES}

Remember: Always explain your reasoning before submitting the query.
"""


# =============================================================================
# Agent Tools
# =============================================================================

CYPHER_GENERATOR_TOOLS = [
    get_graph_schema_for_cypher,
    propose_cypher_query,
    get_cypher_feedback,
    get_sample_data,
]


# =============================================================================
# Agent Factory
# =============================================================================

def create_cypher_generator_agent(llm=None, name: str = "cypher_generator") -> Agent:
    """
    Create a Cypher Generator Agent.

    This agent uses Silicon Cloud's Qwen3-Coder model, which is optimized
    for code generation tasks including Cypher queries.

    Args:
        llm: Optional LLM instance. If None, uses Silicon Cloud Qwen3-Coder.
        name: Agent name for identification

    Returns:
        Agent: Configured Cypher Generator Agent

    Example:
        ```python
        from src.agents import create_cypher_generator_agent

        agent = create_cypher_generator_agent()
        # Use within a LoopAgent for validation
        ```
    """
    if llm is None:
        llm = get_silicon_llm()

    return Agent(
        name=name,
        model=llm,
        description="Generates accurate Cypher queries for Neo4j knowledge graph based on natural language questions.",
        instruction=CYPHER_GENERATOR_INSTRUCTION,
        tools=CYPHER_GENERATOR_TOOLS,
    )
