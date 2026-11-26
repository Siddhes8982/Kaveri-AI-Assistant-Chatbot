"""Backward-compatible import for the reusable client in :mod:`src.llm_client`."""
from src.llm_client import Conversation, LLMClient, LLMConfig, Reply, RunLog, tool_spec

__all__ = ["Conversation", "LLMClient", "LLMConfig", "Reply", "RunLog", "tool_spec"]
