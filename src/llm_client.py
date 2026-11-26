"""Reusable OpenAI-compatible LLM client with streaming, retries, and tools."""
from __future__ import annotations

import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Callable, Iterator

import openai
from dotenv import load_dotenv

logger = logging.getLogger(__name__)
MAX_RETRIES = 5  # total attempts, including the initial request


@dataclass
class LLMConfig:
    """Validated provider settings; credential values are never included in repr."""
    base_url: str
    api_key: str = field(repr=False)
    model: str

    @classmethod
    def from_env(cls, *, require_database: bool = False) -> "LLMConfig":
        load_dotenv()
        base_url = os.getenv("LLM_BASE_URL", "").strip()
        api_key = os.getenv("LLM_API_KEY", "").strip()
        model = os.getenv("LLM_MODEL", "").strip()
        missing = [name for name, value in (("LLM_BASE_URL", base_url),
                  ("LLM_API_KEY", api_key), ("LLM_MODEL", model)) if not value]
        if require_database and not os.getenv("DATABASE_URL", "").strip():
            missing.append("DATABASE_URL")
        if missing:
            raise ValueError("Missing required environment setting(s): " + ", ".join(missing))
        return cls(base_url, api_key, model)


@dataclass
class Reply:
    """One model response and measured request metadata."""
    content: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    model: str = ""
    streamed: bool = False
    chunks: int = 0
    latency_ms: int = 0


class RunLog:
    """Append UTF-8 JSON Lines events without ever recording credentials."""
    def __init__(self, path: str | None = None):
        self.path = path

    def write(self, **event: Any) -> None:
        if self.path:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")


def _retry_wait(error: openai.RateLimitError, attempt: int) -> float:
    """Return Retry-After when valid, otherwise bounded exponential backoff."""
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", {}) or {}
    retry_after = headers.get("retry-after") or headers.get("Retry-After")
    if retry_after:
        try:
            return max(0.0, float(retry_after))
        except (TypeError, ValueError):
            try:
                when = parsedate_to_datetime(str(retry_after))
                if when.tzinfo is None:
                    when = when.replace(tzinfo=timezone.utc)
                return max(0.0, (when - datetime.now(timezone.utc)).total_seconds())
            except (TypeError, ValueError, OverflowError):
                pass
    return min(30.0, float(2 ** attempt))


class LLMClient:
    """OpenAI SDK wrapper configured through LLM_* environment variables."""
    def __init__(self, log: RunLog | None = None, max_retries: int = MAX_RETRIES,
                 sleep: Callable[[float], None] | None = None):
        if max_retries < 1:
            raise ValueError("max_retries is the total attempt count and must be at least 1")
        config = LLMConfig.from_env()
        self.base_url, self.model = config.base_url, config.model
        self._client = openai.OpenAI(base_url=config.base_url, api_key=config.api_key,
                                     max_retries=0, timeout=float(os.getenv("LLM_TIMEOUT", "300")))
        self.create = self._client.chat.completions.create
        self.log = log or RunLog()
        self.max_retries = max_retries
        self.sleep = sleep or time.sleep

    def _call(self, **kwargs: Any) -> Any:
        for attempt in range(1, self.max_retries + 1):
            try:
                return self.create(model=self.model, **kwargs)
            except openai.RateLimitError as error:
                if attempt == self.max_retries:
                    raise
                wait = _retry_wait(error, attempt - 1)
                logger.warning("OpenRouter rate limit; retry %d/%d in %.2f seconds", attempt,
                               self.max_retries - 1, wait)
                self.log.write(event="retry", status=429, attempt=attempt, wait_s=round(wait, 2))
                self.sleep(wait)
        raise AssertionError("unreachable")

    def chat(self, messages: list[dict[str, Any]], tools: list[dict] | None = None,
             stream: bool = True, on_token: Callable[[str], None] | None = None) -> Reply:
        started = time.perf_counter()
        extra = {"tools": tools, "tool_choice": "auto"} if tools else {}
        reply = Reply(streamed=stream, model=self.model)
        if not stream:
            response = self._call(messages=messages, **extra)
            reply.model = getattr(response, "model", None) or self.model
            message = response.choices[0].message
            reply.content = message.content or ""
            reply.tool_calls = [_tool_call(tc.id, tc.function.name, tc.function.arguments)
                                for tc in (message.tool_calls or [])]
        else:
            # Retry a stream only when it failed before yielding any text/tool fragment.
            for attempt in range(1, self.max_retries + 1):
                try:
                    stream_obj = self.create(model=self.model, messages=messages, stream=True, **extra)
                    reply.model = getattr(stream_obj, "model", None) or self.model
                    self._consume_stream(stream_obj, reply, on_token)
                    break
                except openai.RateLimitError as error:
                    if attempt == self.max_retries or reply.chunks or reply.tool_calls:
                        raise
                    wait = _retry_wait(error, attempt - 1)
                    logger.warning("OpenRouter rate limit; retry %d/%d in %.2f seconds", attempt,
                                   self.max_retries - 1, wait)
                    self.log.write(event="retry", status=429, attempt=attempt, wait_s=round(wait, 2))
                    self.sleep(wait)
                except Exception:
                    # Stream fragments are assembled locally; do not replay after any delivery.
                    if reply.chunks or reply.tool_calls:
                        raise
                    raise
        reply.latency_ms = int((time.perf_counter() - started) * 1000)
        return reply

    @staticmethod
    def _consume_stream(stream_obj: Iterator[Any], reply: Reply,
                        on_token: Callable[[str], None] | None) -> None:
        content: list[str] = []
        calls: dict[int, dict[str, str]] = {}
        for chunk in stream_obj:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta.content:
                content.append(delta.content)
                reply.chunks += 1
                if on_token:
                    on_token(delta.content)
            for part in delta.tool_calls or []:
                slot = calls.setdefault(part.index, {"id": "", "name": "", "arguments": ""})
                if part.id:
                    slot["id"] += part.id
                if part.function:
                    slot["name"] += part.function.name or ""
                    slot["arguments"] += part.function.arguments or ""
        reply.content = "".join(content)
        reply.tool_calls = [_tool_call(value["id"], value["name"], value["arguments"])
                            for _, value in sorted(calls.items())]


def _tool_call(call_id: str, name: str, arguments: str | dict) -> dict[str, Any]:
    if isinstance(arguments, str):
        arguments = json.loads(arguments or "{}")
    if not isinstance(arguments, dict):
        raise ValueError("Tool arguments must be a JSON object")
    return {"id": call_id or f"call_{uuid.uuid4().hex[:8]}", "name": name,
            "arguments": arguments}


def _validate_tool_arguments(name: str, arguments: dict[str, Any], tools: list[dict]) -> None:
    """Validate a call against the advertised function schema before dispatch."""
    schema = next((item.get("function", {}) for item in tools
                   if item.get("function", {}).get("name") == name), None)
    if schema is None:
        raise ValueError(f"Tool '{name}' is not declared")
    params = schema.get("parameters", {})
    properties = params.get("properties", {})
    required = params.get("required", [])
    missing = [key for key in required if key not in arguments]
    if missing:
        raise ValueError("Tool arguments are missing required fields")
    if params.get("additionalProperties") is False and set(arguments) - set(properties):
        raise ValueError("Tool arguments contain unsupported fields")
    types = {"string": str, "integer": int, "number": (int, float), "boolean": bool,
             "object": dict, "array": list}
    for key, value in arguments.items():
        expected = types.get(properties.get(key, {}).get("type"))
        if expected and (not isinstance(value, expected) or
                         (properties[key].get("type") == "integer" and isinstance(value, bool))):
            raise ValueError(f"Tool argument '{key}' has the wrong type")


class Conversation:
    """Ordered message history and explicit registered-function tool dispatch."""
    VALID_ROLES = {"system", "user", "assistant", "tool"}

    def __init__(self, client: LLMClient, system: str | None = None,
                 tools: list[dict] | None = None,
                 functions: dict[str, Callable[..., Any]] | None = None, conversation_id: str = "c1"):
        self.client, self.tools = client, tools or []
        self.functions = functions or {}
        self.id, self.turn = conversation_id, 0
        self.history: list[dict[str, Any]] = []
        if system:
            self.add_system_message(system)

    def add_system_message(self, content: str) -> None:
        self.history.append({"role": "system", "content": content})

    def add_user_message(self, content: str) -> None:
        self.history.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str, **extra: Any) -> None:
        self.history.append({"role": "assistant", "content": content, **extra})

    def add_tool_message(self, content: str, tool_call_id: str, name: str | None = None) -> None:
        message = {"role": "tool", "tool_call_id": tool_call_id, "content": content}
        if name:
            message["name"] = name
        self.history.append(message)

    def get_messages(self) -> list[dict[str, Any]]:
        return [dict(message) for message in self.history]

    def clear(self) -> None:
        self.history.clear()
        self.turn = 0

    def ask(self, text: str, on_token: Callable[[str], None] | None = None) -> str:
        self.turn += 1
        self.add_user_message(text)
        self.client.log.write(conversation_id=self.id, turn=self.turn, role="user", content=text)
        for _ in range(4):
            reply = self.client.chat(self.get_messages(), tools=self.tools or None, on_token=on_token)
            assistant_event = {"conversation_id": self.id, "turn": self.turn, "role": "assistant",
                               "content": reply.content, "streamed": reply.streamed,
                               "chunks": reply.chunks, "model": reply.model,
                               "latency_ms": reply.latency_ms}
            if not reply.tool_calls:
                self.add_assistant_message(reply.content)
                self.client.log.write(**assistant_event)
                return reply.content
            serialized_calls = [{"id": call["id"], "type": "function", "function": {
                "name": call["name"], "arguments": json.dumps(call["arguments"])} }
                for call in reply.tool_calls]
            self.client.log.write(**assistant_event, tool_calls=reply.tool_calls)
            self.add_assistant_message(reply.content or None, tool_calls=serialized_calls)
            for call in reply.tool_calls:
                name, call_id = call["name"], call["id"]
                try:
                    if name not in self.functions:
                        raise ValueError(f"Tool '{name}' is not registered")
                    _validate_tool_arguments(name, call["arguments"], self.tools)
                    result = self.functions[name](**call["arguments"])
                    result_text = str(result)
                except Exception:
                    logger.exception("Tool execution failed for %s", name)
                    raise
                self.add_tool_message(result_text, call_id, name)
                self.client.log.write(conversation_id=self.id, turn=self.turn, role="tool",
                                      tool_call_id=call_id, name=name, content=result_text)
        raise RuntimeError("The model kept calling tools without answering")


def tool_spec(name: str, description: str, properties: dict, required: list[str]) -> dict:
    """Build an OpenAI-compatible function tool schema."""
    return {"type": "function", "function": {"name": name, "description": description,
            "parameters": {"type": "object", "properties": properties,
                           "required": required, "additionalProperties": False}}}
