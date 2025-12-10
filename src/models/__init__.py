"""
Models module for Agentic KG.

Contains data structure definitions for the Schema-First pipeline.
"""

from .target_schema import (
    EntityType,
    NodeDefinition,
    RelationshipDefinition,
    TargetSchema,
)

__all__ = [
    "EntityType",
    "NodeDefinition",
    "RelationshipDefinition",
    "TargetSchema",
]
