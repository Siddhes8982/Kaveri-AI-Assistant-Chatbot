"""Unit tests for configuration, conversations, retries, streaming, tools, and SQL."""
import json
from types import SimpleNamespace as NS

import openai
import httpx
import pytest
from sqlalchemy import create_engine

from src import llm_client as module
from src.llm_client import (Conversation, LLMClient, LLMConfig, Reply, RunLog,
                            _validate_tool_arguments, tool_spec)
from src import tools as sql_tools


def config_env(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")
    monkeypatch.setenv("LLM_API_KEY", "test-secret")
    monkeypatch.setenv("LLM_MODEL", "test-model")


def fake_client(create, log=None, sleep=None, attempts=5):
    client = LLMClient.__new__(LLMClient)
    client.base_url, client.model, client.create = "https://api.invalid", "unit-model", create
    client.log, client.sleep, client.max_retries = log or RunLog(), sleep or (lambda _: None), attempts
    return client


def test_config_validation_and_repr_does_not_expose_secret(monkeypatch):
    config_env(monkeypatch)
    cfg = LLMConfig.from_env()
    assert cfg.model == "test-model"
    assert "test-secret" not in repr(cfg)
    monkeypatch.delenv("LLM_API_KEY")
    monkeypatch.setattr(module, "load_dotenv", lambda: None)
    with pytest.raises(ValueError, match="LLM_API_KEY"):
        LLMConfig.from_env()


def test_client_initialization(monkeypatch):
    config_env(monkeypatch)
    seen = {}
    class DummyOpenAI:
        def __init__(self, **kwargs):
            seen.update(kwargs)
            self.chat = NS(completions=NS(create=lambda **_: None))
    monkeypatch.setattr(module.openai, "OpenAI", DummyOpenAI)
    client = LLMClient()
    assert client.model == "test-model"
    assert seen["base_url"] == "https://openrouter.ai/api/v1"
    assert seen["api_key"] == "test-secret"
    assert seen["max_retries"] == 0


def test_history_roles_order_and_clear():
    conv = Conversation(fake_client(lambda **_: None), system="system")
    conv.add_user_message("user")
    conv.add_assistant_message("assistant", tool_calls=[])
    conv.add_tool_message("result", "call-1", "my_tool")
    assert [m["role"] for m in conv.get_messages()] == ["system", "user", "assistant", "tool"]
    assert conv.get_messages()[-1]["tool_call_id"] == "call-1"
    conv.clear()
    assert conv.get_messages() == []


def test_streaming_and_non_streaming(monkeypatch):
    chunk = NS(choices=[NS(delta=NS(content="hello", tool_calls=None))])
    streamed = fake_client(lambda **_: iter([chunk]))
    out = []
    reply = streamed.chat([], on_token=out.append)
    assert reply.content == "hello" and reply.chunks == 1 and out == ["hello"]
    msg = NS(content="complete", tool_calls=[])
    nonstream = fake_client(lambda **_: NS(model="actual", choices=[NS(message=msg)]))
    result = nonstream.chat([], stream=False)
    assert result.content == "complete" and result.streamed is False and result.model == "actual"


def test_streaming_tool_fragment_reconstruction():
    def delta(index, ident=None, name=None, args=None):
        fn = NS(name=name, arguments=args) if name is not None or args is not None else None
        return NS(index=index, id=ident, function=fn)
    chunks = [NS(choices=[NS(delta=NS(content=None, tool_calls=[delta(0, "call_", "get_", '{"country":')]))]),
              NS(choices=[NS(delta=NS(content=None, tool_calls=[delta(0, "1", "revenue", '"Germany"}')] ))])]
    result = fake_client(lambda **_: iter(chunks)).chat([], tools=[{}])
    assert result.tool_calls == [{"id": "call_1", "name": "get_revenue", "arguments": {"country": "Germany"}}]


def test_tool_registry_and_argument_validation():
    tools = [tool_spec("lookup", "desc", {"year": {"type": "integer"}}, ["year"])]
    _validate_tool_arguments("lookup", {"year": 2024}, tools)
    with pytest.raises(ValueError, match="not declared"):
        _validate_tool_arguments("other", {}, tools)
    with pytest.raises(ValueError, match="required"):
        _validate_tool_arguments("lookup", {}, tools)
    with pytest.raises(ValueError, match="unsupported"):
        _validate_tool_arguments("lookup", {"year": 2024, "extra": "x"}, tools)
    with pytest.raises(ValueError, match="wrong type"):
        _validate_tool_arguments("lookup", {"year": True}, tools)


def test_retry_backoff_retry_after_and_maximum(tmp_path, monkeypatch):
    delays, events = [], tmp_path / "events.jsonl"
    request = httpx.Request("POST", "https://example.invalid")
    response = httpx.Response(429, headers={"retry-after": "3"}, request=request)
    error = openai.RateLimitError("limited", response=response, body=None)
    calls = iter([error, error, NS(ok=True)])
    def create(**_):
        item = next(calls)
        if isinstance(item, Exception):
            raise item
        return item
    client = fake_client(create, RunLog(str(events)), delays.append)
    assert client._call() == NS(ok=True)
    assert delays == [3.0, 3.0]
    entries = [json.loads(line) for line in events.read_text(encoding="utf-8").splitlines()]
    assert [entry["attempt"] for entry in entries] == [1, 2]

    delays.clear()
    client = fake_client(lambda **_: (_ for _ in ()).throw(error), sleep=delays.append, attempts=3)
    with pytest.raises(openai.RateLimitError):
        client._call()
    assert delays == [3.0, 3.0]


def test_exponential_backoff_and_non_429_propagation(monkeypatch):
    delays = []
    err = openai.RateLimitError("limited", response=httpx.Response(
        429, request=httpx.Request("POST", "https://example.invalid")), body=None)
    client = fake_client(lambda **_: (_ for _ in ()).throw(err), sleep=delays.append, attempts=4)
    with pytest.raises(openai.RateLimitError):
        client._call()
    assert delays == [1.0, 2.0, 4.0]
    other = RuntimeError("bad request")
    with pytest.raises(RuntimeError, match="bad request"):
        fake_client(lambda **_: (_ for _ in ()).throw(other))._call()


def test_conversation_tool_execution_and_log(tmp_path):
    path = tmp_path / "run.jsonl"
    tool_reply = Reply(content="", tool_calls=[{"id": "real-id", "name": "lookup",
                                                   "arguments": {"key": "value"}}], model="m")
    final_reply = Reply(content="found", model="m", chunks=1, streamed=True)
    replies = iter([tool_reply, final_reply])
    client = fake_client(lambda **_: next(replies), RunLog(str(path)))
    client.chat = lambda *args, **kwargs: next(replies)
    called = []
    conv = Conversation(client, tools=[tool_spec("lookup", "desc", {"key": {"type": "string"}}, ["key"])],
                        functions={"lookup": lambda **kw: called.append(kw) or "result"})
    assert conv.ask("question") == "found"
    assert called == [{"key": "value"}]
    assert conv.history[-2]["tool_call_id"] == "real-id"
    assert conv.history[-2]["content"] == "result"
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [r["role"] for r in records] == ["user", "assistant", "tool", "assistant"]
    assert records[2]["tool_call_id"] == "real-id"
    assert "test-secret" not in path.read_text(encoding="utf-8")


def _conversation_with_failing_tool(error):
    tool_reply = Reply(content="", tool_calls=[{"id": "failure-call", "name": "lookup",
                                                  "arguments": {"key": "value"}}], model="m")
    client = fake_client(lambda **_: None)
    client.chat = lambda *args, **kwargs: tool_reply

    def fail_tool(**_kwargs):
        raise error

    return Conversation(client, tools=[tool_spec("lookup", "desc", {"key": {"type": "string"}}, ["key"])],
                       functions={"lookup": fail_tool})


def test_tool_value_error_propagates_and_no_tool_result_is_added():
    error = ValueError("Database failure")
    conversation = _conversation_with_failing_tool(error)
    with pytest.raises(ValueError) as caught:
        conversation.ask("question")
    assert caught.value is error
    assert not any(message["role"] == "tool" for message in conversation.history)


def test_tool_runtime_error_propagates_unchanged():
    error = RuntimeError("Tool failure")
    conversation = _conversation_with_failing_tool(error)
    with pytest.raises(RuntimeError) as caught:
        conversation.ask("question")
    assert caught.value is error
    assert not any(message["role"] == "tool" for message in conversation.history)


def test_sql_tool_parameterized_query_and_missing(monkeypatch):
    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as c:
        c.exec_driver_sql("CREATE TABLE customers (customer_id INTEGER, country TEXT)")
        c.exec_driver_sql("CREATE TABLE orders (order_id INTEGER, customer_id INTEGER, order_date TEXT, total_amount NUMERIC)")
        c.exec_driver_sql("INSERT INTO customers VALUES (1, 'Germany')")
        c.exec_driver_sql("INSERT INTO orders VALUES (7, 1, '2024-05-10', 1234.5)")
    monkeypatch.setattr(sql_tools, "get_engine", lambda: engine)
    assert sql_tools.get_revenue("Germany", 2024) == "USD 1234.50"
    assert "No revenue records" in sql_tools.get_revenue("France", 2024)
