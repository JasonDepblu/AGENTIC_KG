"""
Knowledge Graph Extraction Tools for Agentic KG.

Tools for extracting entities and relationships from unstructured text
using Neo4j GraphRAG library with LLM-powered entity extraction.

Based on the original architecture from reference/kg_construction_2.ipynb

This module provides two modes:
1. Full GraphRAG mode: Uses Neo4j GraphRAG's SimpleKGPipeline for complete
   knowledge graph construction with Lexical Graph (chunks) and Subject Graph
   (entities with __Entity__ label).
2. Simplified mode: Direct LLM calls for entity extraction, stores results
   in session state (fallback when GraphRAG is unavailable).
"""

import os
import re
from typing import Dict, List, Optional, Any
from pathlib import Path

from google.adk.tools import ToolContext

from .common import tool_success, tool_error
from .unstructured_extraction import APPROVED_ENTITIES, APPROVED_FACTS
from ..config import get_config

# =============================================================================
# GraphRAG Components (Neo4j GraphRAG Library)
# =============================================================================

# Flag to track if GraphRAG is available
_GRAPHRAG_AVAILABLE = False

try:
    from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
    from neo4j_graphrag.experimental.components.text_splitters.base import TextSplitter
    from neo4j_graphrag.experimental.components.types import TextChunk, TextChunks
    from neo4j_graphrag.experimental.components.pdf_loader import DataLoader
    from neo4j_graphrag.experimental.components.types import PdfDocument, DocumentInfo
    from neo4j_graphrag.llm import OpenAILLM
    from neo4j_graphrag.embeddings import OpenAIEmbeddings
    _GRAPHRAG_AVAILABLE = True
except ImportError:
    # GraphRAG not available, will use simplified mode
    SimpleKGPipeline = None
    TextSplitter = object  # Placeholder base class
    TextChunk = None
    TextChunks = None
    DataLoader = object  # Placeholder base class
    PdfDocument = None
    DocumentInfo = None
    OpenAILLM = None
    OpenAIEmbeddings = None


class RegexTextSplitter(TextSplitter):
    """
    Custom text splitter that uses regex patterns to chunk text.

    This splitter breaks documents at specified delimiters (like "---")
    to create meaningful text segments for processing.

    Based on reference/kg_construction_2.ipynb
    """

    def __init__(self, pattern: str = "---"):
        """
        Initialize the regex text splitter.

        Args:
            pattern: Regex pattern to split text on (default: "---")
        """
        self.pattern = pattern

    async def run(self, text: str) -> "TextChunks":
        """
        Split text into chunks using the regex pattern.

        Args:
            text: The text to split

        Returns:
            TextChunks: A list of text chunks with indices
        """
        if not _GRAPHRAG_AVAILABLE:
            raise ImportError("neo4j-graphrag is required for RegexTextSplitter")

        texts = re.split(self.pattern, text)
        chunks = [
            TextChunk(text=str(t).strip(), index=i)
            for i, t in enumerate(texts)
            if t and t.strip()
        ]
        return TextChunks(chunks=chunks)


class MarkdownDataLoader(DataLoader):
    """
    Custom data loader for Markdown files.

    Adapts the Neo4j GraphRAG PDF loader interface to work with markdown files.
    It reads markdown content, extracts the document title from the first H1 header,
    and wraps it in the expected document format for the pipeline.

    Based on reference/kg_construction_2.ipynb
    """

    def extract_title(self, markdown_text: str) -> str:
        """
        Extract the title from markdown text (first H1 header).

        Args:
            markdown_text: The markdown content

        Returns:
            str: The title, or "Untitled" if not found
        """
        pattern = r'^# (.+)$'
        match = re.search(pattern, markdown_text, re.MULTILINE)
        return match.group(1) if match else "Untitled"

    async def run(self, filepath: Path, metadata: Dict = None) -> "PdfDocument":
        """
        Load a markdown file and return it as a document.

        Args:
            filepath: Path to the markdown file
            metadata: Optional metadata dict

        Returns:
            PdfDocument: The loaded document with text and metadata
        """
        if not _GRAPHRAG_AVAILABLE:
            raise ImportError("neo4j-graphrag is required for MarkdownDataLoader")

        if metadata is None:
            metadata = {}

        with open(filepath, "r", encoding="utf-8") as f:
            markdown_text = f.read()

        doc_headline = self.extract_title(markdown_text)
        doc_info = DocumentInfo(
            path=str(filepath),
            metadata={
                "title": doc_headline,
                **metadata
            }
        )
        return PdfDocument(text=markdown_text, document_info=doc_info)


# =============================================================================
# GraphRAG LLM and Embedder Configuration
# =============================================================================

def get_graphrag_llm() -> "OpenAILLM":
    """
    Get the LLM configured for Neo4j GraphRAG.

    Uses the project's LLM configuration (DashScope compatible).

    Returns:
        OpenAILLM: Configured LLM instance for GraphRAG

    Raises:
        ImportError: If neo4j-graphrag is not installed
    """
    if not _GRAPHRAG_AVAILABLE:
        raise ImportError(
            "neo4j-graphrag is required for GraphRAG mode. "
            "Install it with: pip install neo4j-graphrag"
        )

    config = get_config()
    return OpenAILLM(
        model_name=config.llm.default_model,
        model_params={"temperature": 0},
        api_key=config.llm.api_key,
        base_url=config.llm.api_base,
    )


class DashScopeEmbedderAdapter:
    """
    Adapter to use src/embed_model.py embedders with Neo4j GraphRAG.

    Neo4j GraphRAG expects an embedder with `embed_query` method that
    returns a list of floats. This adapter wraps the DashScope embedder
    from src/embed_model.py.
    """

    def __init__(self, embedder=None):
        """
        Initialize the adapter.

        Args:
            embedder: LangChain-compatible embedder with embed_query/embed_documents methods.
                     If None, imports embedder_qwen from src.embed_model
        """
        if embedder is None:
            try:
                from ..embed_model import embedder_qwen
                self.embedder = embedder_qwen
            except ImportError:
                raise ImportError(
                    "Could not import embedder_qwen from src.embed_model. "
                    "Make sure the module exists and has the embedder configured."
                )
        else:
            self.embedder = embedder

    async def embed_query(self, text: str) -> List[float]:
        """
        Embed a single text query (async interface for Neo4j GraphRAG).

        Args:
            text: The text to embed

        Returns:
            List[float]: The embedding vector
        """
        # LangChain embedders are synchronous, so we just call them directly
        return self.embedder.embed_query(text)

    def embed_query_sync(self, text: str) -> List[float]:
        """
        Embed a single text query (sync interface).

        Args:
            text: The text to embed

        Returns:
            List[float]: The embedding vector
        """
        return self.embedder.embed_query(text)

    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple documents (async interface for Neo4j GraphRAG).

        Args:
            texts: List of texts to embed

        Returns:
            List[List[float]]: List of embedding vectors
        """
        return self.embedder.embed_documents(texts)


def get_graphrag_embedder(use_dashscope: bool = True):
    """
    Get the embeddings model configured for Neo4j GraphRAG.

    Args:
        use_dashscope: If True, use the DashScope embedder from src/embed_model.py.
                      If False, use OpenAIEmbeddings with config settings.

    Returns:
        Embedder instance compatible with Neo4j GraphRAG

    Raises:
        ImportError: If neo4j-graphrag is not installed
    """
    if not _GRAPHRAG_AVAILABLE:
        raise ImportError(
            "neo4j-graphrag is required for GraphRAG mode. "
            "Install it with: pip install neo4j-graphrag"
        )

    if use_dashscope:
        try:
            return DashScopeEmbedderAdapter()
        except ImportError:
            # Fall back to OpenAIEmbeddings if DashScope embedder not available
            pass

    # Fall back to OpenAIEmbeddings
    config = get_config()
    return OpenAIEmbeddings(
        model=config.llm.embedding_model,
        api_key=config.llm.api_key,
        base_url=config.llm.api_base,
    )


# =============================================================================
# Contextualized Extraction Prompt
# =============================================================================

def create_contextualized_prompt(context: str) -> str:
    """
    Create a contextualized prompt template for entity extraction.

    This template includes file context to help the LLM better understand
    the document structure when processing individual chunks.

    Args:
        context: The file context (first few lines of the file)

    Returns:
        str: The prompt template with placeholders for {schema} and {text}
    """
    return f'''You are a top-tier algorithm designed for extracting
information in structured formats to build a knowledge graph.

Extract the entities (nodes) and specify their type from the following text.
Also extract the relationships between these nodes.

Return result as JSON using the following format:
{{"nodes": [ {{"id": "0", "label": "Person", "properties": {{"name": "John"}} }}],
"relationships": [{{"type": "KNOWS", "start_node_id": "0", "end_node_id": "1", "properties": {{}} }}] }}

Use only the following node and relationship types (if provided):
{{schema}}

Assign a unique ID (string) to each node, and reuse it to define relationships.
Do respect the source and target node types for relationship and
the relationship direction.

Make sure you adhere to the following rules to produce valid JSON objects:
- Do not return any additional information other than the JSON in it.
- Omit any backticks around the JSON - simply output the JSON on its own.
- The JSON object must not wrapped into a list - it is its own JSON object.
- Property names must be enclosed in double quotes

Consider the following context to help identify entities and relationships:
<context>
{context}
</context>

Input text:

{{text}}
'''


# =============================================================================
# SimpleKGPipeline Factory
# =============================================================================

def make_kg_builder(
    file_path: str,
    entity_schema: Dict[str, Any],
    chunk_delimiter: str = "---"
) -> "SimpleKGPipeline":
    """
    Create a KG builder pipeline for a specific file.

    This creates a customized SimpleKGPipeline that:
    - Uses a contextualized prompt based on file content
    - Splits text using the specified delimiter
    - Extracts entities according to the provided schema
    - Writes results directly to Neo4j with __Entity__ labels

    Args:
        file_path: Path to the file to process
        entity_schema: The entity schema from build_entity_schema
        chunk_delimiter: Regex pattern for splitting text (default: "---")

    Returns:
        SimpleKGPipeline: Configured KG builder pipeline

    Raises:
        ImportError: If neo4j-graphrag is not installed
    """
    if not _GRAPHRAG_AVAILABLE:
        raise ImportError(
            "neo4j-graphrag is required for GraphRAG mode. "
            "Install it with: pip install neo4j-graphrag"
        )

    from ..neo4j_client import get_neo4j_client

    # Get file context for the prompt
    context = _get_file_context(file_path)
    contextualized_prompt = create_contextualized_prompt(context)

    # Get Neo4j driver
    neo4j = get_neo4j_client()
    driver = neo4j.get_driver()

    return SimpleKGPipeline(
        llm=get_graphrag_llm(),
        driver=driver,
        embedder=get_graphrag_embedder(),
        from_pdf=True,  # Use custom loader
        pdf_loader=MarkdownDataLoader(),
        text_splitter=RegexTextSplitter(chunk_delimiter),
        schema=entity_schema,
        prompt_template=contextualized_prompt,
    )


def is_graphrag_available() -> bool:
    """
    Check if Neo4j GraphRAG is available.

    Returns:
        bool: True if GraphRAG can be used, False otherwise
    """
    return _GRAPHRAG_AVAILABLE

# =============================================================================
# State Keys
# =============================================================================

ENTITY_SCHEMA = "entity_schema"
EXTRACTION_RESULTS = "extraction_results"
CORRELATION_RESULTS = "correlation_results"


# =============================================================================
# Schema Building Tools
# =============================================================================

def build_entity_schema(tool_context: ToolContext) -> Dict:
    """
    Build an entity schema from approved entities and fact types.

    This schema is used by the GraphRAG pipeline to constrain entity extraction.

    The schema contains:
    - node_types: List of approved entity types
    - relationship_types: List of predicate labels (uppercase)
    - patterns: List of [subject, predicate, object] patterns

    Returns:
        dict: The entity schema for GraphRAG
    """
    # Get approved entities
    approved_entities = tool_context.state.get(APPROVED_ENTITIES, [])
    if not approved_entities:
        return tool_error(
            "No approved entity types found. "
            "Run the NER agent first to approve entity types."
        )

    # Get approved facts
    approved_facts = tool_context.state.get(APPROVED_FACTS, {})
    if not approved_facts:
        return tool_error(
            "No approved fact types found. "
            "Run the Fact Type agent first to approve fact types."
        )

    # Build relationship types (uppercase predicates)
    relationship_types = [key.upper() for key in approved_facts.keys()]

    # Build patterns as [subject, predicate, object] lists
    patterns = [
        [fact["subject_label"], fact["predicate_label"].upper(), fact["object_label"]]
        for fact in approved_facts.values()
    ]

    # Build the schema
    entity_schema = {
        "node_types": approved_entities,
        "relationship_types": relationship_types,
        "patterns": patterns,
        "additional_node_types": False,  # Strict mode - only extract specified types
    }

    # Save to state
    tool_context.state[ENTITY_SCHEMA] = entity_schema

    return tool_success(ENTITY_SCHEMA, entity_schema)


def get_entity_schema(tool_context: ToolContext) -> Dict:
    """
    Get the current entity schema.

    Returns:
        dict: The entity schema, or error if not built
    """
    schema = tool_context.state.get(ENTITY_SCHEMA)
    if not schema:
        return tool_error(
            "Entity schema not built. "
            "Use build_entity_schema first."
        )
    return tool_success(ENTITY_SCHEMA, schema)


# =============================================================================
# Text Processing Helpers
# =============================================================================

def _get_file_context(file_path: str, num_lines: int = 5) -> str:
    """
    Extract the first few lines of a file for context.

    Args:
        file_path: Path to the file
        num_lines: Number of lines to extract

    Returns:
        str: First few lines of the file
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = []
            for _ in range(num_lines):
                line = f.readline()
                if not line:
                    break
                lines.append(line)
        return "".join(lines)
    except Exception as e:
        return f"Error reading file: {e}"


def _extract_title_from_markdown(markdown_text: str) -> str:
    """
    Extract the title from markdown text (first H1 header).

    Args:
        markdown_text: The markdown content

    Returns:
        str: The title, or "Untitled" if not found
    """
    pattern = r'^# (.+)$'
    match = re.search(pattern, markdown_text, re.MULTILINE)
    return match.group(1) if match else "Untitled"


def _chunk_text(text: str, delimiter: str = "---") -> List[str]:
    """
    Split text into chunks using a delimiter.

    Args:
        text: The text to split
        delimiter: The delimiter pattern (regex)

    Returns:
        List[str]: List of text chunks
    """
    chunks = re.split(delimiter, text)
    # Filter out empty chunks and strip whitespace
    return [chunk.strip() for chunk in chunks if chunk.strip()]


# =============================================================================
# Entity Extraction Tools
# =============================================================================

def extract_entities_from_text(
    text: str,
    entity_schema: Dict,
    tool_context: ToolContext
) -> Dict:
    """
    Extract entities and relationships from text using LLM.

    This is a simplified version that uses the project's LLM directly
    instead of Neo4j GraphRAG's SimpleKGPipeline (which requires additional setup).

    Args:
        text: The text to extract from
        entity_schema: The schema constraining extraction

    Returns:
        dict: Extracted entities and relationships
    """
    from ..llm import get_llm_client

    # Build the extraction prompt
    node_types = entity_schema.get("node_types", [])
    relationship_types = entity_schema.get("relationship_types", [])
    patterns = entity_schema.get("patterns", [])

    prompt = f"""You are a top-tier algorithm designed for extracting
information in structured formats to build a knowledge graph.

Extract the entities (nodes) and specify their type from the following text.
Also extract the relationships between these nodes.

Return result as JSON using the following format:
{{"nodes": [ {{"id": "0", "label": "Person", "properties": {{"name": "John"}} }}],
"relationships": [{{"type": "KNOWS", "start_node_id": "0", "end_node_id": "1", "properties": {{}} }}] }}

Use ONLY the following node and relationship types:
- Node types: {node_types}
- Relationship types: {relationship_types}
- Allowed patterns (subject, predicate, object): {patterns}

Rules:
- Assign a unique ID (string) to each node, and reuse it to define relationships
- Respect the source and target node types for relationships
- Do not extract entities or relationships not in the schema
- Output ONLY valid JSON, no additional text or markdown

Input text:
{text}
"""

    try:
        client = get_llm_client()
        response = client.chat.completions.create(
            model=get_config().llm.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        result_text = response.choices[0].message.content

        # Parse the JSON response
        import json
        # Clean up potential markdown code blocks
        result_text = result_text.strip()
        if result_text.startswith("```"):
            result_text = re.sub(r'^```(?:json)?\n?', '', result_text)
            result_text = re.sub(r'\n?```$', '', result_text)

        extracted = json.loads(result_text)
        return tool_success("extracted", extracted)

    except json.JSONDecodeError as e:
        return tool_error(f"Failed to parse LLM response as JSON: {e}")
    except Exception as e:
        return tool_error(f"Entity extraction failed: {e}")


def process_unstructured_file(
    file_path: str,
    chunk_delimiter: str = "---",
    tool_context: ToolContext = None
) -> Dict:
    """
    Process an unstructured text file to extract entities and relationships.

    This tool:
    1. Reads the file
    2. Chunks it using the delimiter
    3. Extracts entities from each chunk
    4. Aggregates results

    Args:
        file_path: Path to the file (relative to import directory)
        chunk_delimiter: Regex pattern to split text into chunks

    Returns:
        dict: Aggregated extraction results
    """
    # Get import directory
    config = get_config()
    import_dir = config.neo4j.import_dir

    if import_dir:
        full_path = Path(import_dir) / file_path
    else:
        full_path = Path(file_path)

    if not full_path.exists():
        return tool_error(f"File not found: {file_path}")

    # Get entity schema
    entity_schema = tool_context.state.get(ENTITY_SCHEMA)
    if not entity_schema:
        return tool_error(
            "Entity schema not built. "
            "Use build_entity_schema first."
        )

    try:
        # Read the file
        with open(full_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Get document metadata
        title = _extract_title_from_markdown(content)

        # Chunk the text
        chunks = _chunk_text(content, chunk_delimiter)

        # Extract from each chunk
        all_nodes = []
        all_relationships = []
        node_id_offset = 0

        for i, chunk in enumerate(chunks):
            if not chunk:
                continue

            result = extract_entities_from_text(chunk, entity_schema, tool_context)

            if result.get("status") == "success":
                extracted = result.get("extracted", {})
                nodes = extracted.get("nodes", [])
                relationships = extracted.get("relationships", [])

                # Offset node IDs to avoid conflicts between chunks
                id_map = {}
                for node in nodes:
                    old_id = node.get("id")
                    new_id = str(node_id_offset + int(old_id) if old_id.isdigit() else f"{i}_{old_id}")
                    id_map[old_id] = new_id
                    node["id"] = new_id
                    node["chunk_index"] = i
                    all_nodes.append(node)

                for rel in relationships:
                    rel["start_node_id"] = id_map.get(rel["start_node_id"], rel["start_node_id"])
                    rel["end_node_id"] = id_map.get(rel["end_node_id"], rel["end_node_id"])
                    rel["chunk_index"] = i
                    all_relationships.append(rel)

                node_id_offset += len(nodes) + 100  # Buffer to avoid collisions

        # Store results
        extraction_result = {
            "file_path": str(file_path),
            "title": title,
            "chunk_count": len(chunks),
            "nodes": all_nodes,
            "relationships": all_relationships,
            "node_count": len(all_nodes),
            "relationship_count": len(all_relationships),
        }

        # Add to extraction results list
        results_list = tool_context.state.get(EXTRACTION_RESULTS, [])
        results_list.append(extraction_result)
        tool_context.state[EXTRACTION_RESULTS] = results_list

        return tool_success("extraction_result", extraction_result)

    except Exception as e:
        return tool_error(f"Failed to process file: {e}")


def get_extraction_results(tool_context: ToolContext) -> Dict:
    """
    Get all extraction results.

    Returns:
        dict: List of all extraction results
    """
    results = tool_context.state.get(EXTRACTION_RESULTS, [])
    return tool_success(EXTRACTION_RESULTS, results)


# =============================================================================
# GraphRAG Entity Extraction (Full Pipeline)
# =============================================================================

async def process_unstructured_file_graphrag(
    file_path: str,
    chunk_delimiter: str = "---",
    tool_context: ToolContext = None
) -> Dict:
    """
    Process an unstructured text file using Neo4j GraphRAG SimpleKGPipeline.

    This is the full GraphRAG implementation that:
    1. Uses SimpleKGPipeline to process the file
    2. Creates Chunk nodes (Lexical Graph) with embeddings
    3. Creates Entity nodes (Subject Graph) with __Entity__ label
    4. Writes all results directly to Neo4j

    Args:
        file_path: Path to the file (relative to import directory)
        chunk_delimiter: Regex pattern to split text into chunks

    Returns:
        dict: Extraction results including node and relationship counts
    """
    if not _GRAPHRAG_AVAILABLE:
        return tool_error(
            "neo4j-graphrag is not installed. "
            "Install it with: pip install neo4j-graphrag, "
            "or use process_unstructured_file for simplified mode."
        )

    # Get import directory
    config = get_config()
    import_dir = config.neo4j.import_dir

    if import_dir:
        full_path = Path(import_dir) / file_path
    else:
        full_path = Path(file_path)

    if not full_path.exists():
        return tool_error(f"File not found: {file_path}")

    # Get entity schema
    entity_schema = tool_context.state.get(ENTITY_SCHEMA)
    if not entity_schema:
        return tool_error(
            "Entity schema not built. "
            "Use build_entity_schema first."
        )

    try:
        # Create the KG builder pipeline
        kg_builder = make_kg_builder(
            str(full_path),
            entity_schema,
            chunk_delimiter
        )

        # Run the async pipeline
        result = await kg_builder.run_async(file_path=str(full_path))

        # Extract result information
        extraction_result = {
            "file_path": str(file_path),
            "mode": "graphrag",
            "status": "success",
            "result": result.result if hasattr(result, 'result') else str(result),
        }

        # Add to extraction results list
        results_list = tool_context.state.get(EXTRACTION_RESULTS, [])
        results_list.append(extraction_result)
        tool_context.state[EXTRACTION_RESULTS] = results_list

        return tool_success("extraction_result", extraction_result)

    except Exception as e:
        return tool_error(f"GraphRAG extraction failed: {e}")


def process_unstructured_file_auto(
    file_path: str,
    chunk_delimiter: str = "---",
    tool_context: ToolContext = None,
    prefer_graphrag: bool = True
) -> Dict:
    """
    Process an unstructured file, automatically selecting the best mode.

    This function tries to use GraphRAG mode first (if available and preferred),
    falling back to simplified mode if needed.

    Args:
        file_path: Path to the file (relative to import directory)
        chunk_delimiter: Regex pattern to split text into chunks
        prefer_graphrag: If True, prefer GraphRAG mode when available

    Returns:
        dict: Extraction results
    """
    if prefer_graphrag and _GRAPHRAG_AVAILABLE:
        # Use asyncio to run the async function
        import asyncio

        try:
            # Try to get existing event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If already in async context, create a task
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run,
                        process_unstructured_file_graphrag(
                            file_path, chunk_delimiter, tool_context
                        )
                    )
                    return future.result()
            else:
                return loop.run_until_complete(
                    process_unstructured_file_graphrag(
                        file_path, chunk_delimiter, tool_context
                    )
                )
        except RuntimeError:
            # No event loop, create new one
            return asyncio.run(
                process_unstructured_file_graphrag(
                    file_path, chunk_delimiter, tool_context
                )
            )
    else:
        # Fall back to simplified mode
        return process_unstructured_file(file_path, chunk_delimiter, tool_context)


# =============================================================================
# Entity Resolution Tools
# =============================================================================

def correlate_entities(
    entity_label: str,
    entity_key: str,
    domain_key: str,
    similarity_threshold: float = 0.9,
    tool_context: ToolContext = None
) -> Dict:
    """
    Correlate extracted entities with domain graph nodes using string similarity.

    This connects the "subject graph" (extracted entities) with the "domain graph"
    (existing graph from structured data) using Jaro-Winkler distance.

    Args:
        entity_label: The label of entities to correlate (e.g., "Product")
        entity_key: The property key in extracted entities (e.g., "name")
        domain_key: The property key in domain nodes (e.g., "product_name")
        similarity_threshold: Minimum similarity for correlation (0.0-1.0)

    Returns:
        dict: Correlation results
    """
    try:
        from rapidfuzz import fuzz
    except ImportError:
        return tool_error(
            "rapidfuzz library not installed. "
            "Install it with: pip install rapidfuzz"
        )

    # Get extraction results
    extraction_results = tool_context.state.get(EXTRACTION_RESULTS, [])
    if not extraction_results:
        return tool_error("No extraction results found. Process files first.")

    # Collect all extracted entities of the specified label
    extracted_entities = []
    for result in extraction_results:
        for node in result.get("nodes", []):
            if node.get("label") == entity_label:
                props = node.get("properties", {})
                if entity_key in props:
                    extracted_entities.append({
                        "id": node.get("id"),
                        "value": props[entity_key],
                        "source_file": result.get("file_path"),
                    })

    if not extracted_entities:
        return tool_error(
            f"No extracted entities of type '{entity_label}' with key '{entity_key}' found."
        )

    # Get domain node values from Neo4j
    from ..neo4j_client import get_neo4j_client

    try:
        neo4j = get_neo4j_client()
        query = f"""
        MATCH (n:{entity_label})
        WHERE NOT n:`__Entity__`
        RETURN elementId(n) as id, n.{domain_key} as value
        """
        result = neo4j.send_query(query)
        if result["status"] == "error":
            return tool_error(f"Neo4j query failed: {result.get('message')}")

        domain_nodes = result.get("query_result", [])
    except Exception as e:
        return tool_error(f"Failed to query domain nodes: {e}")

    if not domain_nodes:
        return tool_error(
            f"No domain nodes of type '{entity_label}' with key '{domain_key}' found."
        )

    # Find correlations using Jaro-Winkler similarity
    correlations = []
    for entity in extracted_entities:
        entity_value = str(entity["value"]).lower().strip()
        best_match = None
        best_score = 0

        for domain in domain_nodes:
            domain_value = str(domain.get("value", "")).lower().strip()
            if not domain_value:
                continue

            # Calculate similarity (rapidfuzz returns 0-100, normalize to 0-1)
            similarity = fuzz.ratio(entity_value, domain_value) / 100.0

            if similarity > best_score and similarity >= similarity_threshold:
                best_score = similarity
                best_match = {
                    "domain_id": domain["id"],
                    "domain_value": domain.get("value"),
                }

        if best_match:
            correlations.append({
                "entity_id": entity["id"],
                "entity_value": entity["value"],
                "domain_id": best_match["domain_id"],
                "domain_value": best_match["domain_value"],
                "similarity": best_score,
                "source_file": entity["source_file"],
            })

    # Store correlation results
    correlation_results = tool_context.state.get(CORRELATION_RESULTS, {})
    correlation_results[entity_label] = {
        "entity_key": entity_key,
        "domain_key": domain_key,
        "threshold": similarity_threshold,
        "total_entities": len(extracted_entities),
        "correlated_count": len(correlations),
        "correlations": correlations,
    }
    tool_context.state[CORRELATION_RESULTS] = correlation_results

    return tool_success("correlation_result", correlation_results[entity_label])


def get_correlation_results(tool_context: ToolContext) -> Dict:
    """
    Get all entity correlation results.

    Returns:
        dict: All correlation results by entity label
    """
    results = tool_context.state.get(CORRELATION_RESULTS, {})
    return tool_success(CORRELATION_RESULTS, results)


def create_correspondence_relationships(
    entity_label: str,
    tool_context: ToolContext
) -> Dict:
    """
    Create CORRESPONDS_TO relationships in Neo4j between correlated entities.

    This connects the subject graph (extracted entities) to the domain graph
    (existing structured data nodes).

    Args:
        entity_label: The label of entities to connect

    Returns:
        dict: Results of relationship creation
    """
    from ..neo4j_client import get_neo4j_client

    correlation_results = tool_context.state.get(CORRELATION_RESULTS, {})
    label_results = correlation_results.get(entity_label)

    if not label_results:
        return tool_error(
            f"No correlation results for '{entity_label}'. "
            "Run correlate_entities first."
        )

    correlations = label_results.get("correlations", [])
    if not correlations:
        return tool_error(f"No correlations found for '{entity_label}'.")

    try:
        neo4j = get_neo4j_client()
        created_count = 0

        for corr in correlations:
            # Create CORRESPONDS_TO relationship
            # Note: This assumes extracted entities are stored with __Entity__ label
            query = """
            MATCH (entity), (domain)
            WHERE elementId(entity) = $entity_id AND elementId(domain) = $domain_id
            MERGE (entity)-[r:CORRESPONDS_TO]->(domain)
            ON CREATE SET r.created_at = datetime(), r.similarity = $similarity
            ON MATCH SET r.updated_at = datetime()
            RETURN count(r) as count
            """
            result = neo4j.send_query(query, {
                "entity_id": corr["entity_id"],
                "domain_id": corr["domain_id"],
                "similarity": corr["similarity"],
            })

            if result["status"] == "success":
                created_count += result.get("query_result", [{}])[0].get("count", 0)

        return tool_success("correspondence_result", {
            "entity_label": entity_label,
            "relationships_created": created_count,
        })

    except Exception as e:
        return tool_error(f"Failed to create relationships: {e}")


# =============================================================================
# GraphRAG Entity Resolution (Neo4j APOC)
# =============================================================================

def correlate_entities_graphrag(
    entity_label: str,
    entity_key: str,
    domain_key: str,
    similarity_threshold: float = 0.9,
    tool_context: ToolContext = None
) -> Dict:
    """
    Correlate extracted entities with domain graph using Neo4j APOC functions.

    This is the GraphRAG version that performs entity resolution directly
    in Neo4j using Jaro-Winkler distance, which is more efficient for
    large graphs as it avoids loading all nodes into Python.

    This function works with entities created by SimpleKGPipeline that have
    the __Entity__ label.

    Args:
        entity_label: The label of entities to correlate (e.g., "Product")
        entity_key: The property key in extracted entities (e.g., "name")
        domain_key: The property key in domain nodes (e.g., "product_name")
        similarity_threshold: Minimum similarity for correlation (0.0-1.0)

    Returns:
        dict: Correlation results and created relationships
    """
    from ..neo4j_client import get_neo4j_client

    # Convert similarity threshold to distance (Jaro-Winkler returns 0 for exact match)
    distance_threshold = 1.0 - similarity_threshold

    try:
        neo4j = get_neo4j_client()

        # Use Neo4j APOC to correlate and create relationships in one query
        query = f"""
        MATCH (entity:{entity_label}:`__Entity__`), (domain:{entity_label})
        WHERE NOT domain:`__Entity__`
        WITH entity, domain,
             apoc.text.jaroWinklerDistance(
                 toLower(toString(entity[${entity_key!r}])),
                 toLower(toString(domain[${domain_key!r}]))
             ) as distance
        WHERE distance < $threshold
        WITH entity, domain, (1.0 - distance) as similarity
        ORDER BY similarity DESC
        MERGE (entity)-[r:CORRESPONDS_TO]->(domain)
        ON CREATE SET r.created_at = datetime(), r.similarity = similarity
        ON MATCH SET r.updated_at = datetime(), r.similarity = similarity
        RETURN
            entity[${entity_key!r}] as entity_value,
            domain[${domain_key!r}] as domain_value,
            similarity,
            elementId(entity) as entity_id,
            elementId(domain) as domain_id
        """

        result = neo4j.send_query(query, {
            "threshold": distance_threshold,
        })

        if result["status"] == "error":
            # Fall back to non-APOC method if APOC is not available
            if "apoc" in result.get("message", "").lower():
                return tool_error(
                    "APOC plugin not available in Neo4j. "
                    "Use correlate_entities instead (Python-based)."
                )
            return tool_error(f"Neo4j query failed: {result.get('message')}")

        correlations = result.get("query_result", [])

        # Store correlation results
        correlation_results = tool_context.state.get(CORRELATION_RESULTS, {})
        correlation_results[entity_label] = {
            "entity_key": entity_key,
            "domain_key": domain_key,
            "threshold": similarity_threshold,
            "mode": "graphrag_apoc",
            "correlated_count": len(correlations),
            "correlations": correlations,
        }
        tool_context.state[CORRELATION_RESULTS] = correlation_results

        return tool_success("correlation_result", correlation_results[entity_label])

    except Exception as e:
        return tool_error(f"GraphRAG entity correlation failed: {e}")


def auto_correlate_all_entities(
    similarity_threshold: float = 0.9,
    tool_context: ToolContext = None
) -> Dict:
    """
    Automatically correlate all entity types between Subject and Domain graphs.

    This function:
    1. Finds all unique entity labels in the Subject Graph (__Entity__ nodes)
    2. Finds matching labels in the Domain Graph
    3. Attempts to correlate keys automatically
    4. Creates CORRESPONDS_TO relationships

    Based on the workflow in reference/kg_construction_2.ipynb

    Args:
        similarity_threshold: Minimum similarity for correlation (0.0-1.0)

    Returns:
        dict: Summary of all correlations created
    """
    from ..neo4j_client import get_neo4j_client

    try:
        neo4j = get_neo4j_client()

        # Step 1: Find unique entity labels in Subject Graph
        entity_labels_result = neo4j.send_query("""
            MATCH (n)
            WHERE n:`__Entity__`
            WITH DISTINCT labels(n) AS entity_labels
            UNWIND entity_labels AS entity_label
            WITH entity_label
            WHERE NOT entity_label STARTS WITH "__"
            RETURN collect(DISTINCT entity_label) as unique_labels
        """)

        if entity_labels_result["status"] == "error":
            return tool_error(f"Failed to get entity labels: {entity_labels_result.get('message')}")

        entity_labels = entity_labels_result.get("query_result", [{}])[0].get("unique_labels", [])

        if not entity_labels:
            return tool_error("No entity labels found in Subject Graph.")

        # Step 2: For each label, find domain nodes and correlate
        all_correlations = {}

        for label in entity_labels:
            # Get entity keys for this label
            entity_keys_result = neo4j.send_query(f"""
                MATCH (n:{label})
                WHERE n:`__Entity__`
                WITH DISTINCT keys(n) as entityKeys
                UNWIND entityKeys as entityKey
                WHERE entityKey <> 'id' AND NOT entityKey STARTS WITH '_'
                RETURN collect(DISTINCT entityKey) as unique_keys
            """)

            entity_keys = entity_keys_result.get("query_result", [{}])[0].get("unique_keys", [])

            # Get domain keys for this label
            domain_keys_result = neo4j.send_query(f"""
                MATCH (n:{label})
                WHERE NOT n:`__Entity__`
                WITH DISTINCT keys(n) as domainKeys
                UNWIND domainKeys as domainKey
                WHERE domainKey <> 'id' AND NOT domainKey STARTS WITH '_'
                RETURN collect(DISTINCT domainKey) as unique_keys
            """)

            domain_keys = domain_keys_result.get("query_result", [{}])[0].get("unique_keys", [])

            if not entity_keys or not domain_keys:
                all_correlations[label] = {
                    "status": "skipped",
                    "reason": "No matching keys found"
                }
                continue

            # Find best key match (using simple name similarity)
            best_match = None
            best_score = 0

            for e_key in entity_keys:
                for d_key in domain_keys:
                    # Normalize keys for comparison
                    e_normalized = e_key.lower().replace(label.lower(), "").strip("_")
                    d_normalized = d_key.lower().replace(label.lower(), "").strip("_")

                    if e_normalized == d_normalized:
                        best_match = (e_key, d_key)
                        best_score = 1.0
                        break
                    elif e_normalized in d_normalized or d_normalized in e_normalized:
                        if 0.8 > best_score:
                            best_match = (e_key, d_key)
                            best_score = 0.8

                if best_score == 1.0:
                    break

            if best_match:
                # Run correlation
                result = correlate_entities_graphrag(
                    label,
                    best_match[0],
                    best_match[1],
                    similarity_threshold,
                    tool_context
                )

                if result.get("status") == "success":
                    all_correlations[label] = {
                        "status": "success",
                        "entity_key": best_match[0],
                        "domain_key": best_match[1],
                        "correlated_count": result.get("correlation_result", {}).get("correlated_count", 0)
                    }
                else:
                    all_correlations[label] = {
                        "status": "failed",
                        "error": result.get("error", "Unknown error")
                    }
            else:
                all_correlations[label] = {
                    "status": "skipped",
                    "reason": "No matching key pairs found",
                    "entity_keys": entity_keys,
                    "domain_keys": domain_keys
                }

        return tool_success("auto_correlation_results", all_correlations)

    except Exception as e:
        return tool_error(f"Auto correlation failed: {e}")
