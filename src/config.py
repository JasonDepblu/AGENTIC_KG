"""
Configuration management for Agentic KG.

This module provides unified configuration management for all components,
including Neo4j connection, LLM settings, and environment variables.
"""

import os
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


@dataclass
class Neo4jConfig:
    """Neo4j database configuration."""
    uri: str = field(default_factory=lambda: os.getenv("NEO4J_URI", "bolt://localhost:7687"))
    username: str = field(default_factory=lambda: os.getenv("NEO4J_USERNAME", "neo4j"))
    password: str = field(default_factory=lambda: os.getenv("NEO4J_PASSWORD", ""))
    database: str = field(default_factory=lambda: os.getenv("NEO4J_DATABASE", "neo4j"))
    import_dir: Optional[str] = field(default_factory=lambda: os.getenv("NEO4J_IMPORT_DIR"))

    def validate(self) -> bool:
        """Validate Neo4j configuration."""
        if not self.uri:
            raise ValueError("NEO4J_URI is required")
        if not self.password:
            raise ValueError("NEO4J_PASSWORD is required")
        return True


@dataclass
class LLMConfig:
    """LLM configuration supporting multiple providers."""
    # DashScope (primary provider)
    api_key: str = field(default_factory=lambda: os.getenv("DASHSCOPE_API_KEY", ""))
    api_base: str = field(default_factory=lambda: os.getenv(
        "DASHSCOPE_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1"
    ))
    default_model: str = field(default_factory=lambda: os.getenv(
        "LLM_MODEL",
        "qwen-plus-2025-07-28"  # Default to qwen-plus for stability
    ))
    embedding_model: str = field(default_factory=lambda: os.getenv(
        "EMBEDDING_MODEL",
        "text-embedding-v3"  # DashScope embedding model
    ))

    # Silicon Cloud (for specialized tasks like Cypher generation)
    silicon_api_key: str = field(default_factory=lambda: os.getenv("SILICON_API_KEY", ""))
    silicon_api_base: str = field(default_factory=lambda: os.getenv(
        "SILICON_API_BASE",
        "https://api.siliconflow.cn/v1"
    ))
    silicon_model: str = field(default_factory=lambda: os.getenv(
        "SILICON_MODEL",
        "Qwen/Qwen3-Coder-480B-A35B-Instruct"
    ))

    # Available models
    MODELS = {
        "kimi": "Moonshot-Kimi-K2-Instruct",
        "qwen-large": "qwen3-235b-a22b-instruct-2507",
        "qwen-plus": "qwen-plus-2025-07-28",
        "qwen-medium": "qwen3-30b-a3b-instruct-2507",
        "qwen-flash": "qwen-flash",  # Fast and cheap, supports deep thinking
    }

    # Silicon Cloud models
    SILICON_MODELS = {
        "qwen3-coder": "Qwen/Qwen3-Coder-480B-A35B-Instruct",
        "qwen3-large": "Qwen/Qwen3-235B-A22B-FP8",
    }

    def validate(self) -> bool:
        """Validate LLM configuration."""
        if not self.api_key:
            raise ValueError("DASHSCOPE_API_KEY is required")
        return True

    def get_model(self, model_alias: Optional[str] = None) -> str:
        """Get model name by alias or return default."""
        if model_alias is None:
            return self.default_model
        return self.MODELS.get(model_alias, model_alias)


@dataclass
class TextExtractionConfig:
    """Configuration for text feedback column extraction."""
    # Whether to require user confirmation for extracted entity types
    require_user_confirmation: bool = field(
        default_factory=lambda: os.getenv("TEXT_EXTRACTION_REQUIRE_CONFIRMATION", "true").lower() == "true"
    )
    # Enable LLM-based entity deduplication
    enable_entity_dedup: bool = field(
        default_factory=lambda: os.getenv("TEXT_EXTRACTION_ENABLE_DEDUP", "true").lower() == "true"
    )
    # Batch size for LLM text extraction calls
    batch_size: int = field(
        default_factory=lambda: int(os.getenv("TEXT_EXTRACTION_BATCH_SIZE", "10"))
    )
    # LLM model for text analysis (defaults to qwen-plus-2025-07-28)
    model: str = field(
        default_factory=lambda: os.getenv("TEXT_EXTRACTION_MODEL", "qwen-plus-2025-07-28")
    )


@dataclass
class AppConfig:
    """Application-wide configuration."""
    neo4j: Neo4jConfig = field(default_factory=Neo4jConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    text_extraction: TextExtractionConfig = field(default_factory=TextExtractionConfig)

    # Logging
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))

    # Agent settings
    max_iterations: int = 5
    verbose: bool = False

    def validate(self) -> bool:
        """Validate all configurations."""
        self.neo4j.validate()
        self.llm.validate()
        return True


# Global configuration instance
_config: Optional[AppConfig] = None


def get_config() -> AppConfig:
    """Get or create the global configuration instance."""
    global _config
    if _config is None:
        _config = AppConfig()
    return _config


def reset_config() -> None:
    """Reset the global configuration (useful for testing)."""
    global _config
    _config = None


# Convenience functions for common settings
def get_neo4j_import_dir() -> Optional[str]:
    """Get the Neo4j import directory path."""
    return get_config().neo4j.import_dir


def get_dashscope_api_key() -> str:
    """Get the DashScope API key."""
    return get_config().llm.api_key


def get_dashscope_base_url() -> str:
    """Get the DashScope API base URL."""
    return get_config().llm.api_base
