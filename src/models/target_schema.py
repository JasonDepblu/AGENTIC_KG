"""
Target Schema Data Structures for Schema-First Pipeline.

Defines the data structures for representing a target knowledge graph schema
that guides the preprocessing stage.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum
import json


class EntityType(str, Enum):
    """Predefined entity types for knowledge graph nodes."""
    BRAND = "Brand"
    MODEL = "Model"
    STORE = "Store"
    RESPONDENT = "Respondent"
    ASPECT = "Aspect"
    FEATURE = "Feature"
    ISSUE = "Issue"
    MEDIA = "Media"
    CUSTOM = "Custom"


@dataclass
class NodeDefinition:
    """
    Definition of a node type in the target schema.

    Attributes:
        label: Node label in the graph (e.g., "Brand")
        entity_type: Type category for the node
        unique_property: Property that uniquely identifies nodes (e.g., "brand_id")
        properties: List of property names for this node type
        extraction_hints: Hints for data extraction (column patterns, etc.)
    """
    label: str
    unique_property: str
    properties: List[str] = field(default_factory=list)
    entity_type: EntityType = EntityType.CUSTOM
    extraction_hints: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "label": self.label,
            "unique_property": self.unique_property,
            "properties": self.properties,
            "entity_type": self.entity_type.value if isinstance(self.entity_type, EntityType) else self.entity_type,
            "extraction_hints": self.extraction_hints,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NodeDefinition":
        """Create from dictionary."""
        entity_type = data.get("entity_type", "Custom")
        if isinstance(entity_type, str):
            try:
                entity_type = EntityType(entity_type)
            except ValueError:
                entity_type = EntityType.CUSTOM

        return cls(
            label=data["label"],
            unique_property=data["unique_property"],
            properties=data.get("properties", []),
            entity_type=entity_type,
            extraction_hints=data.get("extraction_hints", {}),
        )


@dataclass
class RelationshipDefinition:
    """
    Definition of a relationship type in the target schema.

    Attributes:
        relationship_type: Type name for the relationship (e.g., "RATES")
        from_node: Label of the source node
        to_node: Label of the target node
        properties: List of property names for this relationship
        extraction_hints: Hints for data extraction
    """
    relationship_type: str
    from_node: str
    to_node: str
    properties: List[str] = field(default_factory=list)
    extraction_hints: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "relationship_type": self.relationship_type,
            "from_node": self.from_node,
            "to_node": self.to_node,
            "properties": self.properties,
            "extraction_hints": self.extraction_hints,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RelationshipDefinition":
        """Create from dictionary."""
        return cls(
            relationship_type=data["relationship_type"],
            from_node=data["from_node"],
            to_node=data["to_node"],
            properties=data.get("properties", []),
            extraction_hints=data.get("extraction_hints", {}),
        )


@dataclass
class TargetSchema:
    """
    Complete target schema definition for knowledge graph construction.

    This schema is designed during the SCHEMA_DESIGN phase and guides
    the TARGETED_PREPROCESSING phase.

    Attributes:
        name: Name of the schema
        description: Description of what this schema represents
        nodes: Dictionary of node definitions keyed by label
        relationships: Dictionary of relationship definitions keyed by type
    """
    name: str
    description: str = ""
    nodes: Dict[str, NodeDefinition] = field(default_factory=dict)
    relationships: Dict[str, RelationshipDefinition] = field(default_factory=dict)

    def add_node(self, node: NodeDefinition) -> None:
        """Add a node definition to the schema."""
        self.nodes[node.label] = node

    def remove_node(self, label: str) -> Optional[NodeDefinition]:
        """Remove a node definition and its associated relationships."""
        if label not in self.nodes:
            return None

        node = self.nodes.pop(label)

        # Remove relationships involving this node
        to_remove = [
            rel_type for rel_type, rel in self.relationships.items()
            if rel.from_node == label or rel.to_node == label
        ]
        for rel_type in to_remove:
            del self.relationships[rel_type]

        return node

    def add_relationship(self, relationship: RelationshipDefinition) -> None:
        """Add a relationship definition to the schema."""
        self.relationships[relationship.relationship_type] = relationship

    def remove_relationship(self, rel_type: str) -> Optional[RelationshipDefinition]:
        """Remove a relationship definition."""
        return self.relationships.pop(rel_type, None)

    def get_node(self, label: str) -> Optional[NodeDefinition]:
        """Get a node definition by label."""
        return self.nodes.get(label)

    def get_relationship(self, rel_type: str) -> Optional[RelationshipDefinition]:
        """Get a relationship definition by type."""
        return self.relationships.get(rel_type)

    def validate_connectivity(self) -> tuple[bool, List[str]]:
        """
        Validate that all nodes are connected through relationships.

        Returns:
            Tuple of (is_valid, list_of_isolated_nodes)
        """
        if not self.nodes:
            return True, []

        connected_nodes = set()
        for rel in self.relationships.values():
            connected_nodes.add(rel.from_node)
            connected_nodes.add(rel.to_node)

        isolated = [label for label in self.nodes.keys() if label not in connected_nodes]
        return len(isolated) == 0, isolated

    def validate_relationships(self) -> tuple[bool, List[str]]:
        """
        Validate that all relationship endpoints reference existing nodes.

        Returns:
            Tuple of (is_valid, list_of_error_messages)
        """
        errors = []
        for rel_type, rel in self.relationships.items():
            if rel.from_node not in self.nodes:
                errors.append(f"Relationship '{rel_type}' references undefined node '{rel.from_node}'")
            if rel.to_node not in self.nodes:
                errors.append(f"Relationship '{rel_type}' references undefined node '{rel.to_node}'")
        return len(errors) == 0, errors

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "description": self.description,
            "nodes": {k: v.to_dict() for k, v in self.nodes.items()},
            "relationships": {k: v.to_dict() for k, v in self.relationships.items()},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TargetSchema":
        """Create from dictionary."""
        schema = cls(
            name=data.get("name", "Unnamed Schema"),
            description=data.get("description", ""),
        )

        for label, node_data in data.get("nodes", {}).items():
            schema.nodes[label] = NodeDefinition.from_dict(node_data)

        for rel_type, rel_data in data.get("relationships", {}).items():
            schema.relationships[rel_type] = RelationshipDefinition.from_dict(rel_data)

        return schema

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_json(cls, json_str: str) -> "TargetSchema":
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))

    def get_summary(self) -> str:
        """Get a human-readable summary of the schema."""
        lines = [
            f"Schema: {self.name}",
            f"Description: {self.description}" if self.description else "",
            "",
            f"Nodes ({len(self.nodes)}):",
        ]

        for label, node in self.nodes.items():
            props = ", ".join(node.properties) if node.properties else "none"
            lines.append(f"  - {label} (unique: {node.unique_property}, properties: {props})")

        lines.append("")
        lines.append(f"Relationships ({len(self.relationships)}):")

        for rel_type, rel in self.relationships.items():
            props = ", ".join(rel.properties) if rel.properties else "none"
            lines.append(f"  - ({rel.from_node})-[{rel_type}]->({rel.to_node}) (properties: {props})")

        return "\n".join(lines)

    def __str__(self) -> str:
        return self.get_summary()


# State keys for tool context
TARGET_SCHEMA_KEY = "target_schema"
APPROVED_TARGET_SCHEMA_KEY = "approved_target_schema"
DETECTED_ENTITIES_KEY = "detected_entities"
