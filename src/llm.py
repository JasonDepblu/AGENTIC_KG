"""
LLM interface module for Agentic KG.

Provides unified LLM interfaces for both Google ADK and LangChain frameworks.

Rate Limit Protection:
- DashScope has RPM (requests per minute) and TPM (tokens per minute) limits
- This module implements retry with exponential backoff
- Supports fallback to alternative models when rate limited
- See: https://help.aliyun.com/zh/model-studio/rate-limit
"""

import os
import time
import logging
from typing import Optional, List

from dotenv import load_dotenv

from .config import get_config

# Apply patches for third-party library compatibility
from .patches import apply_patches
apply_patches()

load_dotenv()

logger = logging.getLogger(__name__)

# Clear proxy environment variables to avoid network issues with DashScope API
# DashScope works better without proxy
for proxy_var in ['http_proxy', 'https_proxy', 'all_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY']:
    os.environ.pop(proxy_var, None)
os.environ['NO_PROXY'] = '*'

# Fallback models for rate limit scenarios
# When primary model is rate limited, try these in order
FALLBACK_MODELS = [
    "qwen-plus-2025-07-28",
    "qwen-plus-latest",
    "qwen-turbo-latest",  # Faster, lower limits but cheaper
]

# Rate limit retry configuration
# Increased from 3 retries / 2s base delay to handle DashScope RPM limits better
MAX_RETRIES = 5
BASE_DELAY = 5  # seconds (increased from 2s)
MAX_DELAY = 120  # seconds (increased from 60s)


def retry_with_exponential_backoff(
    func,
    max_retries: int = MAX_RETRIES,
    base_delay: float = BASE_DELAY,
    max_delay: float = MAX_DELAY,
    fallback_models: Optional[List[str]] = None,
):
    """
    Decorator/wrapper for retrying LLM calls with exponential backoff.

    Handles rate limit errors (429) by:
    1. Waiting with exponential backoff
    2. Optionally switching to fallback models

    Args:
        func: The function to retry
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        fallback_models: List of alternative model names to try

    Returns:
        The result of the function call
    """
    import functools

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        last_exception = None
        current_model = kwargs.get('model')
        models_to_try = [current_model] if current_model else []

        if fallback_models:
            models_to_try.extend([m for m in fallback_models if m != current_model])

        for attempt in range(max_retries):
            for model_idx, model in enumerate(models_to_try or [None]):
                try:
                    if model and 'model' in kwargs:
                        kwargs['model'] = model
                        if model != current_model:
                            logger.info(f"Switching to fallback model: {model}")

                    return func(*args, **kwargs)

                except Exception as e:
                    error_str = str(e).lower()
                    is_rate_limit = (
                        'rate' in error_str and 'limit' in error_str
                        or '429' in error_str
                        or 'quota' in error_str
                        or 'too many requests' in error_str
                    )

                    if is_rate_limit:
                        last_exception = e
                        delay = min(base_delay * (2 ** attempt), max_delay)

                        # Try next model if available
                        if model_idx < len(models_to_try) - 1:
                            logger.warning(
                                f"Rate limited on {model}, trying next model..."
                            )
                            continue

                        # No more models, wait and retry
                        logger.warning(
                            f"Rate limit hit (attempt {attempt + 1}/{max_retries}). "
                            f"Waiting {delay:.1f}s before retry..."
                        )
                        time.sleep(delay)
                        break  # Break inner loop to retry with first model
                    else:
                        # Non-rate-limit error, re-raise immediately
                        raise

        # All retries exhausted
        if last_exception:
            logger.error(f"All {max_retries} retries exhausted due to rate limiting")
            raise last_exception

    return wrapper


def configure_litellm_retry():
    """
    Configure LiteLLM's built-in retry mechanism for rate limits.

    LiteLLM has built-in support for retries. This function configures
    it with appropriate settings for DashScope.
    """
    import litellm

    # Enable LiteLLM's built-in retry mechanism
    litellm.num_retries = MAX_RETRIES

    # Configure retry delays (LiteLLM uses these internally)
    litellm.request_timeout = 120  # 2 minutes timeout

    # Log rate limit warnings
    litellm.set_verbose = False  # Set to True for debugging

    logger.info(
        f"LiteLLM configured with {MAX_RETRIES} retries, "
        f"{litellm.request_timeout}s timeout"
    )


def get_adk_llm(model_name: Optional[str] = None):
    """
    Create a LiteLlm instance for Google ADK.

    This function configures LiteLLM to work with DashScope's OpenAI-compatible API.
    Includes automatic rate limit protection with retry and fallback support.

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

    # Configure LiteLLM's built-in retry mechanism for rate limits
    # This helps handle DashScope's RPM/TPM limits automatically
    litellm.num_retries = MAX_RETRIES
    litellm.request_timeout = 120  # 2 minutes

    return LiteLlm(model=f"openai/{model_name}")


def get_adk_llm_flash():
    """
    Create a LiteLlm instance using qwen-flash model.

    This is a fast and cheap model from DashScope, suitable for simple tasks
    like data extraction in the preprocessing phase. Supports deep thinking.

    Returns:
        LiteLlm: Google ADK compatible LLM instance using qwen-flash

    Usage:
        from src.llm import get_adk_llm_flash
        llm = get_adk_llm_flash()
        agent = Agent(model=llm, ...)
    """
    return get_adk_llm(model_name="qwen-flash")


def get_silicon_llm(model_name: Optional[str] = None):
    """
    Create a LiteLlm instance for Silicon Cloud API.

    Optimized for specialized tasks like Cypher query generation using Qwen3-Coder.

    Args:
        model_name: Model name to use. Defaults to Qwen/Qwen3-Coder-480B-A35B-Instruct.
                   Available options:
                   - Qwen/Qwen3-Coder-480B-A35B-Instruct (default, best for code)
                   - Qwen/Qwen3-Coder-30B-A3B-Instruct (smaller, faster)
                   - Qwen/Qwen2.5-Coder-32B-Instruct

    Returns:
        LiteLlm: Google ADK compatible LLM instance for Silicon Cloud

    Usage:
        from src.llm import get_silicon_llm
        llm = get_silicon_llm()
        agent = Agent(model=llm, ...)
    """
    from google.adk.models.lite_llm import LiteLlm
    import litellm

    config = get_config()

    # Get model name from config or use provided
    if model_name is None:
        model_name = config.llm.silicon_model

    # Configure LiteLLM for Silicon Cloud OpenAI-compatible API
    api_key = config.llm.silicon_api_key
    api_base = config.llm.silicon_api_base

    if not api_key:
        raise ValueError(
            "SILICON_API_KEY is required. Please set it in your .env file."
        )

    # Use openai/ prefix with api_base parameter for custom endpoint
    # LiteLLM supports passing api_base directly in model string
    # Format: openai/<model>@<api_base>
    # Or use environment variables specific to this provider

    # Set up Silicon Cloud specific environment
    # Using SILICON_ prefix to avoid conflicts with DashScope
    os.environ["SILICON_API_KEY"] = api_key
    os.environ["SILICON_API_BASE"] = api_base

    # For LiteLLM, we can use the custom_llm_provider approach
    # or directly set the base_url in the LiteLlm constructor
    return LiteLlm(
        model=f"openai/{model_name}",
        api_key=api_key,
        api_base=api_base,
    )


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


def get_litellm_client():
    """
    Get an OpenAI client configured for LiteLLM/DashScope API.

    This client is used for direct API calls to the LLM, bypassing
    ADK or LangChain abstractions when simpler direct calls are needed.

    Returns:
        OpenAI: OpenAI client instance configured for DashScope
    """
    from openai import OpenAI

    config = get_config()

    return OpenAI(
        api_key=config.llm.api_key,
        base_url=config.llm.api_base
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
