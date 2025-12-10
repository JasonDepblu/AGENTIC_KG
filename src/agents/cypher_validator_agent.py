"""
Cypher Validator Agent for Query Verification.

Validates Cypher queries for syntax, execution, and result correctness.
Works as part of the Cypher Generation Loop to ensure query accuracy.
"""

from google.adk.agents import Agent

from ..llm import get_adk_llm
from ..tools.cypher_validation import (
    validate_cypher_syntax,
    execute_and_validate_query,
    approve_cypher_query,
    get_cypher_feedback,
)


# =============================================================================
# Agent Instructions
# =============================================================================

CYPHER_VALIDATOR_ROLE = """
You are a Cypher query validator. Your job is to verify that proposed queries
are syntactically correct, execute successfully, and return accurate results.
"""

CYPHER_VALIDATOR_WORKFLOW = """
## Validation Workflow

Follow these steps to validate the proposed query:

### Step 1: Syntax Validation
Call `validate_cypher_syntax` to check for syntax errors.
- If syntax is invalid, provide feedback about the specific error
- If syntax is valid, proceed to execution

### Step 2: Execution Validation
Call `execute_and_validate_query` to run the query and check results.
- This will catch runtime errors (missing labels, wrong relationship types)
- It also performs sanity checks on the results

### Step 3: Result Analysis
Analyze the execution results:
- Are there results? (empty results may indicate wrong filters)
- Is the count reasonable? (very high counts may indicate duplicate counting)
- Do the results make sense for the question asked?

### Step 4: Decision
Based on your analysis:
- If everything looks correct: Call `approve_cypher_query` and respond with "valid"
- If there are issues: Provide specific feedback for the generator to fix
"""

CYPHER_VALIDATOR_FEEDBACK_FORMAT = """
## Providing Feedback

When validation fails, provide specific, actionable feedback:

### For Syntax Errors
```
syntax_error: [specific error message]
Suggestion: Check the Cypher syntax at [location]. Common issues include...
```

### For Execution Errors
```
execution_error: [error message]
Suggestion: Verify that node label 'X' exists in the graph. Use get_graph_schema_for_cypher to check.
```

### For Logic Warnings
```
logic_warning: [description of the issue]
Suggestion: The query returned [X] results but expected around [Y]. This may indicate...
```

### Common Issues to Watch For

1. **Duplicate Counting**
   - Symptom: Count is higher than expected
   - Cause: Using COUNT(*) while traversing relationships
   - Fix: Use COUNT(DISTINCT node) instead

2. **Empty Results**
   - Symptom: Query returns 0 rows
   - Cause: Filter conditions too restrictive or wrong property names
   - Fix: Check property names and filter values

3. **Wrong Aggregation Level**
   - Symptom: Results don't match the question
   - Cause: Aggregating at wrong level (e.g., per opinion vs per respondent)
   - Fix: Adjust GROUP BY / RETURN structure
"""

CYPHER_VALIDATOR_APPROVAL = """
## Approving Queries

Only approve a query when ALL of these conditions are met:

1. Syntax validation passed
2. Execution completed without errors
3. Results are non-empty (unless empty is expected)
4. Result counts seem reasonable
5. The query logic matches what the user asked for

When approving, call `approve_cypher_query` and respond with exactly "valid".
This signals the loop to stop and return the results.
"""

CYPHER_VALIDATOR_INSTRUCTION = f"""
{CYPHER_VALIDATOR_ROLE}

{CYPHER_VALIDATOR_WORKFLOW}

{CYPHER_VALIDATOR_FEEDBACK_FORMAT}

{CYPHER_VALIDATOR_APPROVAL}

Remember: Be thorough but efficient. The goal is to catch errors early
and provide clear feedback so the generator can fix issues quickly.
"""


# =============================================================================
# Agent Tools
# =============================================================================

CYPHER_VALIDATOR_TOOLS = [
    validate_cypher_syntax,
    execute_and_validate_query,
    approve_cypher_query,
    get_cypher_feedback,
]


# =============================================================================
# Agent Factory
# =============================================================================

def create_cypher_validator_agent(llm=None, name: str = "cypher_validator") -> Agent:
    """
    Create a Cypher Validator Agent.

    This agent validates proposed Cypher queries for syntax, execution,
    and result correctness. It provides feedback to the generator when
    issues are found.

    Args:
        llm: Optional LLM instance. If None, uses default DashScope LLM.
        name: Agent name for identification

    Returns:
        Agent: Configured Cypher Validator Agent

    Example:
        ```python
        from src.agents import create_cypher_validator_agent

        agent = create_cypher_validator_agent()
        # Use within a LoopAgent for validation
        ```
    """
    if llm is None:
        llm = get_adk_llm()

    return Agent(
        name=name,
        model=llm,
        description="Validates Cypher queries for syntax, execution, and result correctness.",
        instruction=CYPHER_VALIDATOR_INSTRUCTION,
        tools=CYPHER_VALIDATOR_TOOLS,
        output_key="cypher_feedback",  # Stores validation result in session state
    )
