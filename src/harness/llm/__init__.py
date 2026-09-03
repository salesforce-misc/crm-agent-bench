from harness.llm.base import ChatResponse, LLMClient, StreamChunk
from harness.llm.openai_client import OpenAIClient

__all__ = ["ChatResponse", "LLMClient", "OpenAIClient", "StreamChunk"]


def create_llm_client(
    provider: str,
    model: str,
    api_key: str | None = None,
    base_url: str | None = None,
    bedrock_compat: bool = False,
    reasoning_effort: str | None = None,
) -> LLMClient:
    """Factory: create an LLM client from provider/model strings.

    Supported providers: "openai" (any OpenAI-compatible endpoint) and
    "litellm". An unknown provider raises rather than silently defaulting to
    OpenAI — a typo in LLM_PROVIDER must not quietly switch provider behavior
    and invalidate a comparison run.
    """
    if provider == "openai":
        return OpenAIClient(
            model=model,
            api_key=api_key,
            base_url=base_url,
            bedrock_compat=bedrock_compat,
        )
    if provider == "litellm":
        from harness.llm.litellm_client import LiteLLMClient

        return LiteLLMClient(
            model=model,
            api_key=api_key,
            base_url=base_url,
            reasoning_effort=reasoning_effort,
        )
    raise ValueError(
        f"Unknown LLM provider: {provider!r}. Supported providers: 'openai', 'litellm'."
    )
