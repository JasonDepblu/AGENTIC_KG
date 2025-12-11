"""
Target Schema Data Structures for Schema-First Pipeline.

Defines the data structures for representing a target knowledge graph schema
that guides the preprocessing stage.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from enum import Enum
import json


class EntityType(str, Enum):
    """
    Predefined entity types for knowledge graph nodes.

    DEPRECATED: This enum is kept for backward compatibility.
    New code should use DynamicEntityType or raw strings instead.
    When USE_DYNAMIC_ENTITY_TYPES=True, entity types are dynamically
    detected by LLM and registered in the EntityTypeRegistry.
    """
    BRAND = "Brand"
    MODEL = "Model"
    STORE = "Store"
    RESPONDENT = "Respondent"
    ASPECT = "Aspect"
    FEATURE = "Feature"
    ISSUE = "Issue"
    MEDIA = "Media"
    CUSTOM = "Custom"


# Configuration flag for dynamic entity types
USE_DYNAMIC_ENTITY_TYPES = True


class EntityTypeRegistry:
    """
    Registry for dynamically discovered entity types.

    This enables domain-agnostic entity type detection by allowing
    LLM-detected types to be registered at runtime.
    """
    _instance = None
    _types: Dict[str, Dict[str, Any]] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._types = {}
            # Pre-register built-in types
            for et in EntityType:
                cls._types[et.value] = {
                    "name": et.value,
                    "source": "builtin",
                    "description": f"Built-in {et.value} type",
                }
        return cls._instance

    @classmethod
    def register(
        cls,
        name: str,
        description: str = "",
        source: str = "llm_detected",
        domain: str = "",
        **metadata
    ) -> str:
        """
        Register a new entity type.

        Args:
            name: Entity type name (e.g., "Patient", "Department")
            description: Description of this entity type
            source: Where this type was detected from
            domain: Domain this type belongs to (e.g., "medical", "automotive")
            **metadata: Additional metadata

        Returns:
            Registered entity type name
        """
        if cls._instance is None:
            cls()

        cls._types[name] = {
            "name": name,
            "description": description,
            "source": source,
            "domain": domain,
            **metadata,
        }
        return name

    @classmethod
    def get(cls, name: str) -> Optional[Dict[str, Any]]:
        """Get entity type info by name."""
        if cls._instance is None:
            cls()
        return cls._types.get(name)

    @classmethod
    def exists(cls, name: str) -> bool:
        """Check if entity type exists."""
        if cls._instance is None:
            cls()
        return name in cls._types

    @classmethod
    def all_types(cls) -> List[str]:
        """Get list of all registered entity type names."""
        if cls._instance is None:
            cls()
        return list(cls._types.keys())

    @classmethod
    def get_by_domain(cls, domain: str) -> List[str]:
        """Get entity types for a specific domain."""
        if cls._instance is None:
            cls()
        return [
            name for name, info in cls._types.items()
            if info.get("domain") == domain
        ]

    @classmethod
    def clear_dynamic(cls) -> int:
        """Clear all dynamically registered types, keeping built-in ones."""
        if cls._instance is None:
            cls()
        to_remove = [
            name for name, info in cls._types.items()
            if info.get("source") != "builtin"
        ]
        for name in to_remove:
            del cls._types[name]
        return len(to_remove)


def register_entity_type(
    name: str,
    description: str = "",
    domain: str = "",
) -> str:
    """
    Convenience function to register a new entity type.

    Use this when LLM detects a new entity type that should be added
    to the schema.

    Args:
        name: Entity type name (PascalCase, e.g., "Patient", "Department")
        description: Description of what this entity represents
        domain: Domain this type belongs to

    Returns:
        Registered entity type name
    """
    return EntityTypeRegistry.register(
        name=name,
        description=description,
        domain=domain,
        source="llm_detected",
    )


def get_or_create_entity_type(name: str) -> Union[EntityType, str]:
    """
    Get an entity type by name, creating it dynamically if needed.

    First tries to match built-in EntityType enum values.
    If not found and USE_DYNAMIC_ENTITY_TYPES is True, registers as dynamic.
    Otherwise returns EntityType.CUSTOM.

    Args:
        name: Entity type name

    Returns:
        EntityType enum or string for dynamic type
    """
    # Try built-in enum first
    try:
        return EntityType(name)
    except ValueError:
        pass

    # Dynamic entity type handling
    if USE_DYNAMIC_ENTITY_TYPES:
        if not EntityTypeRegistry.exists(name):
            EntityTypeRegistry.register(name, source="auto_created")
        return name

    # Fallback to CUSTOM
    return EntityType.CUSTOM


@dataclass
class NodeDefinition:
    """
    Definition of a node type in the target schema.

    Attributes:
        label: Node label in the graph (e.g., "Brand")
        entity_type: Type category for the node (EntityType enum or string for dynamic types)
        unique_property: Property that uniquely identifies nodes (e.g., "brand_id")
        properties: List of property names for this node type
        extraction_hints: Hints for data extraction (column patterns, etc.)
    """
    label: str
    unique_property: str
    properties: List[str] = field(default_factory=list)
    entity_type: Union[EntityType, str] = EntityType.CUSTOM
    extraction_hints: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        # Handle both EntityType enum and dynamic string types
        if isinstance(self.entity_type, EntityType):
            entity_type_value = self.entity_type.value
        else:
            entity_type_value = str(self.entity_type)

        return {
            "label": self.label,
            "unique_property": self.unique_property,
            "properties": self.properties,
            "entity_type": entity_type_value,
            "extraction_hints": self.extraction_hints,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NodeDefinition":
        """Create from dictionary."""
        entity_type_str = data.get("entity_type", "Custom")

        # Use the smart entity type resolver
        entity_type = get_or_create_entity_type(entity_type_str)

        return cls(
            label=data["label"],
            unique_property=data["unique_property"],
            properties=data.get("properties", []),
            entity_type=entity_type,
            extraction_hints=data.get("extraction_hints", {}),
        )

    def get_entity_type_name(self) -> str:
        """Get the entity type name as a string."""
        if isinstance(self.entity_type, EntityType):
            return self.entity_type.value
        return str(self.entity_type)


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

    def to_markdown(self) -> str:
        """
        Export schema to Markdown format for human-readable documentation.

        Returns:
            Markdown string representation of the schema
        """
        lines = [
            f"# {self.name}",
            "",
        ]

        if self.description:
            lines.append(f"> {self.description}")
            lines.append("")

        lines.append("## Node Types")
        lines.append("")

        for label, node in self.nodes.items():
            lines.append(f"### {label}")
            lines.append(f"- **Unique Property**: `{node.unique_property}`")
            if node.properties:
                lines.append(f"- **Properties**: {', '.join(f'`{p}`' for p in node.properties)}")
            entity_type_name = node.get_entity_type_name()
            if entity_type_name and entity_type_name != "Custom":
                lines.append(f"- **Entity Type**: {entity_type_name}")
            if node.extraction_hints:
                hints_summary = []
                if "source_type" in node.extraction_hints:
                    hints_summary.append(f"source_type={node.extraction_hints['source_type']}")
                if "column_pattern" in node.extraction_hints:
                    hints_summary.append(f"column_pattern=`{node.extraction_hints['column_pattern']}`")
                if hints_summary:
                    lines.append(f"- **Extraction Hints**: {', '.join(hints_summary)}")
            lines.append("")

        lines.append("## Relationship Types")
        lines.append("")

        for rel_type, rel in self.relationships.items():
            lines.append(f"### {rel_type}")
            lines.append(f"- **Pattern**: `({rel.from_node})-[{rel_type}]->({rel.to_node})`")
            if rel.properties:
                lines.append(f"- **Properties**: {', '.join(f'`{p}`' for p in rel.properties)}")
            if rel.extraction_hints:
                hints_summary = []
                if "source_type" in rel.extraction_hints:
                    hints_summary.append(f"source_type={rel.extraction_hints['source_type']}")
                if "column_pattern" in rel.extraction_hints:
                    hints_summary.append(f"column_pattern=`{rel.extraction_hints['column_pattern']}`")
                if hints_summary:
                    lines.append(f"- **Extraction Hints**: {', '.join(hints_summary)}")
            lines.append("")

        return "\n".join(lines)


# State keys for tool context
TARGET_SCHEMA_KEY = "target_schema"
APPROVED_TARGET_SCHEMA_KEY = "approved_target_schema"
DETECTED_ENTITIES_KEY = "detected_entities"
