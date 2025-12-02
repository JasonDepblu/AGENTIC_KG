"""
Pipelines module for Agentic KG.

Contains workflow pipelines that orchestrate agents for end-to-end KG construction.
"""

from .kg_pipeline import KGPipeline, run_full_pipeline

__all__ = [
    "KGPipeline",
    "run_full_pipeline",
]
