"""
LLM interface module for Agentic KG.

Provides unified LLM interfaces for both Google ADK and LangChain frameworks.
"""

import os
from typing import Optional

from dotenv import load_dotenv

from .config import get_config

load_dotenv()


def get_adk_llm(model_name: Optional[str] = None):
    """
    Create a LiteLlm instance for Google ADK.

    This function configures LiteLLM to work with DashScope's OpenAI-compatible API.

    Args:
        model_name: Model name to use. Defaults to Moonshot-Kimi-K2-Instruct.
                   Available options:
                   - Moonshot-Kimi-K2-Instruct (default)
                   - qwen3-235b-a22b-instruct-2507
                   - qwen-plus-latest

    Returns:
        LiteLlm: Google ADK compatible LLM instance

    Usage:
        from src.llm import get_adk_llm
        llm = get_adk_llm()
        agent = Agent(model=llm, ...)
    """
    from google.adk.models.lite_llm import LiteLlm
    import litellm

    config = get_config()

    # Get model name from config or use provided
    if model_name is None:
        model_name = config.llm.default_model

    # Configure LiteLLM for DashScope OpenAI-compatible API
    api_key = config.llm.api_key
    api_base = config.llm.api_base

    # Set environment variables for LiteLLM
    os.environ["OPENAI_API_KEY"] = api_key or ""
    os.environ["OPENAI_API_BASE"] = api_base

    # Also set via litellm module for reliability
    litellm.api_key = api_key
    litellm.api_base = api_base

    return LiteLlm(model=f"openai/{model_name}")


def get_langchain_llm(
    model_name: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 8192,
    streaming: bool = False,
    enable_thinking: bool = False,
):
    """
    Create a LangChain ChatOpenAI instance for DashScope.

    Args:
        model_name: Model name to use
        temperature: Sampling temperature (0.0 to 1.0)
        max_tokens: Maximum tokens in response
        streaming: Enable streaming responses
        enable_thinking: Enable thinking mode (for supported models)

    Returns:
        ChatOpenAI: LangChain compatible LLM instance
    """
    from langchain.chat_models import init_chat_model

    config = get_config()

    if model_name is None:
        model_name = config.llm.default_model

    extra_body = {}
    if not enable_thinking:
        extra_body["enable_thinking"] = False

    return init_chat_model(
        model=model_name,
        model_provider="openai",
        temperature=temperature,
        max_tokens=max_tokens,
        streaming=streaming,
        extra_body=extra_body if extra_body else None,
        openai_api_key=config.llm.api_key,
        openai_api_base=config.llm.api_base,
    )


def get_openai_llm_for_neo4j():
    """
    Create an OpenAI LLM instance for Neo4j GraphRAG.

    Returns:
        OpenAILLM: Neo4j GraphRAG compatible LLM instance
    """
    from neo4j_graphrag.llm import OpenAILLM

    config = get_config()

    # Set OpenAI environment for neo4j_graphrag
    os.environ["OPENAI_API_KEY"] = config.llm.api_key
    os.environ["OPENAI_API_BASE"] = config.llm.api_base

    return OpenAILLM(
        model_name=config.llm.default_model,
        model_params={"temperature": 0}
    )


def get_embeddings():
    """
    Create an OpenAI embeddings instance for Neo4j GraphRAG.

    Returns:
        OpenAIEmbeddings: Embeddings instance for vectorization
    """
    from neo4j_graphrag.embeddings import OpenAIEmbeddings

    config = get_config()

    os.environ["OPENAI_API_KEY"] = config.llm.api_key
    os.environ["OPENAI_API_BASE"] = config.llm.api_base

    return OpenAIEmbeddings(model="text-embedding-3-large")


def test_llm_connection(llm=None) -> bool:
    """
    Test the LLM connection with a simple query.

    Args:
        llm: Optional LLM instance to test. If None, creates a new one.

    Returns:
        bool: True if connection successful, False otherwise
    """
    if llm is None:
        llm = get_adk_llm()

    try:
        response = llm.llm_client.completion(
            model=llm.model,
            messages=[{"role": "user", "content": "Are you ready?"}],
        )
        print(f"LLM Response: {response}")
        return True
    except Exception as e:
        print(f"LLM Connection Error: {e}")
        return False
