"""
Unstructured Data Extraction Tools for Agentic KG.

Tools for Named Entity Recognition (NER) and Fact Type extraction
from unstructured text files (markdown, text, etc.)

Based on the original architecture from reference/schema_proposal_unstructured.ipynb
"""

from typing import Dict, List
from google.adk.tools import ToolContext

from .common import tool_success, tool_error

# =============================================================================
# State Keys
# =============================================================================

# NER Agent state keys
PROPOSED_ENTITIES = "proposed_entity_types"
APPROVED_ENTITIES = "approved_entity_types"

# Fact Type Agent state keys
PROPOSED_FACTS = "proposed_fact_types"
APPROVED_FACTS = "approved_fact_types"


# =============================================================================
# NER Agent Tools - Entity Type Proposal
# =============================================================================

def get_well_known_types(tool_context: ToolContext) -> Dict:
    """
    Gets the approved node labels that represent well-known entity types
    from the existing graph schema (approved_construction_plan).

    Well-known entities are those that already exist in the graph schema
    from structured data processing, such as Product, Part, Supplier, etc.

    Returns:
        dict: A dictionary with status and the set of approved labels
    """
    construction_plan = tool_context.state.get("approved_construction_plan", {})

    if not construction_plan:
        return tool_error(
            "No approved_construction_plan found. "
            "Run the structured data pipeline first to create a graph schema, "
            "or proceed with only discovered entity types."
        )

    # Extract labels from node construction plans
    approved_labels = {
        entry["label"]
        for entry in construction_plan.values()
        if entry.get("construction_type") == "node"
    }

    return tool_success("well_known_types", list(approved_labels))


def set_proposed_entities(
    proposed_entity_types: List[str],
    tool_context: ToolContext
) -> Dict:
    """
    Sets the list of proposed entity types to extract from unstructured text.

    Entity types include:
    - Well-known entities: existing labels like Product, Part, Supplier
    - Discovered entities: new types found in text like Issue, Feature, Reviewer

    Args:
        proposed_entity_types: List of entity type names (e.g., ["Product", "Issue", "Feature"])

    Returns:
        dict: Success status with the proposed entity types
    """
    if not proposed_entity_types:
        return tool_error("proposed_entity_types cannot be empty")

    if not isinstance(proposed_entity_types, list):
        return tool_error("proposed_entity_types must be a list of strings")

    # Validate all items are strings
    for entity_type in proposed_entity_types:
        if not isinstance(entity_type, str) or not entity_type.strip():
            return tool_error(f"Invalid entity type: {entity_type}. Must be a non-empty string.")

    # Normalize: strip whitespace and capitalize
    normalized_types = [t.strip() for t in proposed_entity_types]

    tool_context.state[PROPOSED_ENTITIES] = normalized_types
    return tool_success(PROPOSED_ENTITIES, normalized_types)


def get_proposed_entities(tool_context: ToolContext) -> Dict:
    """
    Gets the list of proposed entity types to extract from unstructured text.

    Returns:
        dict: The list of proposed entity types, or empty list if none proposed
    """
    proposed = tool_context.state.get(PROPOSED_ENTITIES, [])
    return tool_success(PROPOSED_ENTITIES, proposed)


def approve_proposed_entities(tool_context: ToolContext) -> Dict:
    """
    Upon approval from user, records the proposed entity types as approved.

    Only call this tool if the user has explicitly approved the proposed entities.

    Returns:
        dict: Success status with the approved entity types
    """
    if PROPOSED_ENTITIES not in tool_context.state:
        return tool_error(
            "No proposed entity types to approve. "
            "Please use set_proposed_entities first, ask for user approval, "
            "then call this tool."
        )

    proposed = tool_context.state.get(PROPOSED_ENTITIES)
    if not proposed:
        return tool_error("Proposed entity types list is empty.")

    tool_context.state[APPROVED_ENTITIES] = proposed
    return tool_success(APPROVED_ENTITIES, tool_context.state[APPROVED_ENTITIES])


def get_approved_entities(tool_context: ToolContext) -> Dict:
    """
    Gets the approved list of entity types to extract from unstructured text.

    Returns:
        dict: The list of approved entity types, or error if not yet approved
    """
    if APPROVED_ENTITIES not in tool_context.state:
        return tool_error(
            "No approved entity types. "
            "Use the NER agent to propose and approve entity types first."
        )

    return tool_success(APPROVED_ENTITIES, tool_context.state[APPROVED_ENTITIES])


# =============================================================================
# Fact Type Agent Tools - Relationship Type Proposal
# =============================================================================

def add_proposed_fact(
    approved_subject_label: str,
    proposed_predicate_label: str,
    approved_object_label: str,
    tool_context: ToolContext
) -> Dict:
    """
    Add a proposed type of fact that could be extracted from the files.

    A proposed fact type is a tuple of (subject, predicate, object) where
    the subject and object are approved entity types and the predicate
    is a proposed relationship label.

    Example: (Product, has_issue, Issue) or (Product, includes_feature, Feature)

    Args:
        approved_subject_label: Approved label of the subject entity type
        proposed_predicate_label: Label of the predicate/relationship
        approved_object_label: Approved label of the object entity type

    Returns:
        dict: Success status with all proposed facts
    """
    # Validate inputs
    if not approved_subject_label or not approved_subject_label.strip():
        return tool_error("approved_subject_label cannot be empty")
    if not proposed_predicate_label or not proposed_predicate_label.strip():
        return tool_error("proposed_predicate_label cannot be empty")
    if not approved_object_label or not approved_object_label.strip():
        return tool_error("approved_object_label cannot be empty")

    # Get approved entities to validate subject and object
    approved_entities = tool_context.state.get(APPROVED_ENTITIES, [])

    if not approved_entities:
        return tool_error(
            "No approved entity types found. "
            "Please run the NER agent first to approve entity types."
        )

    # Normalize inputs
    subject = approved_subject_label.strip()
    predicate = proposed_predicate_label.strip()
    obj = approved_object_label.strip()

    # Validate that subject and object are in approved entities
    if subject not in approved_entities:
        return tool_error(
            f"Subject label '{subject}' is not an approved entity type. "
            f"Approved types: {approved_entities}"
        )
    if obj not in approved_entities:
        return tool_error(
            f"Object label '{obj}' is not an approved entity type. "
            f"Approved types: {approved_entities}"
        )

    # Add to proposed facts
    current_facts = tool_context.state.get(PROPOSED_FACTS, {})
    current_facts[predicate] = {
        "subject_label": subject,
        "predicate_label": predicate,
        "object_label": obj
    }
    tool_context.state[PROPOSED_FACTS] = current_facts

    return tool_success(PROPOSED_FACTS, current_facts)


def get_proposed_facts(tool_context: ToolContext) -> Dict:
    """
    Gets all proposed fact types that could be extracted from the files.

    Returns:
        dict: Dictionary of proposed fact types keyed by predicate label
    """
    proposed = tool_context.state.get(PROPOSED_FACTS, {})
    return tool_success(PROPOSED_FACTS, proposed)


def remove_proposed_fact(
    predicate_label: str,
    tool_context: ToolContext
) -> Dict:
    """
    Remove a proposed fact type by its predicate label.

    Args:
        predicate_label: The predicate label of the fact to remove

    Returns:
        dict: Success status with remaining proposed facts
    """
    if not predicate_label or not predicate_label.strip():
        return tool_error("predicate_label cannot be empty")

    current_facts = tool_context.state.get(PROPOSED_FACTS, {})
    predicate = predicate_label.strip()

    if predicate not in current_facts:
        return tool_error(f"Fact type '{predicate}' not found in proposed facts")

    del current_facts[predicate]
    tool_context.state[PROPOSED_FACTS] = current_facts

    return tool_success(PROPOSED_FACTS, current_facts)


def approve_proposed_facts(tool_context: ToolContext) -> Dict:
    """
    Upon user approval, records the proposed fact types as approved.

    Only call this tool if the user has explicitly approved the proposed fact types.

    Returns:
        dict: Success status with the approved fact types
    """
    if PROPOSED_FACTS not in tool_context.state:
        return tool_error(
            "No proposed fact types to approve. "
            "Please use add_proposed_fact first, ask for user approval, "
            "then call this tool."
        )

    proposed = tool_context.state.get(PROPOSED_FACTS)
    if not proposed:
        return tool_error("Proposed fact types is empty.")

    tool_context.state[APPROVED_FACTS] = proposed
    return tool_success(APPROVED_FACTS, tool_context.state[APPROVED_FACTS])


def get_approved_facts(tool_context: ToolContext) -> Dict:
    """
    Gets the approved fact types to extract from unstructured text.

    Returns:
        dict: Dictionary of approved fact types, or error if not yet approved
    """
    if APPROVED_FACTS not in tool_context.state:
        return tool_error(
            "No approved fact types. "
            "Use the Fact Type agent to propose and approve fact types first."
        )

    return tool_success(APPROVED_FACTS, tool_context.state[APPROVED_FACTS])
