import asyncio
import json

from voice_agent.providers.base import ToolCall
from voice_agent.tools import Tool, ToolExecutor, default_tools, validate_arguments


def executor(**kwargs):
    return ToolExecutor(default_tools(), **kwargs)


async def test_get_current_time_with_and_without_timezone():
    result = await executor().run(ToolCall("1", "get_current_time", {"timezone": "Asia/Kolkata"}))
    assert not result.is_error
    data = json.loads(result.content)
    assert data["timezone"] == "Asia/Kolkata"
    assert data["iso"].endswith("+05:30")
    assert " on " in data["spoken"]

    local = await executor().run(ToolCall("2", "get_current_time", {}))
    assert not local.is_error


async def test_unknown_timezone_is_a_tool_error():
    result = await executor().run(ToolCall("1", "get_current_time", {"timezone": "Mars/Olympus"}))
    assert result.is_error
    assert "Unknown time zone" in json.loads(result.content)["error"]


async def test_order_lookup_normalizes_the_id():
    result = await executor().run(ToolCall("1", "get_order_status", {"order_id": "a-1001"}))
    assert json.loads(result.content) == {
        "order_id": "A1001",
        "status": "shipped",
        "carrier": "UPS",
        "estimated_delivery": "Friday",
    }


async def test_missing_order_is_reported_to_the_model():
    result = await executor().run(ToolCall("1", "get_order_status", {"order_id": "Z999"}))
    assert result.is_error
    assert json.loads(result.content) == {"error": "No order found with ID 'Z999'."}


async def test_bad_arguments_never_reach_the_handler():
    ex = executor()
    missing = await ex.run(ToolCall("1", "get_order_status", {}))
    wrong_type = await ex.run(ToolCall("2", "get_order_status", {"order_id": 1001}))
    extra = await ex.run(ToolCall("3", "get_current_time", {"zone": "UTC"}))
    assert all(r.is_error for r in (missing, wrong_type, extra))
    assert "missing required argument 'order_id'" in missing.content


async def test_unknown_tool():
    result = await executor().run(ToolCall("1", "launch_rocket", {}))
    assert result.is_error


async def test_slow_and_crashing_tools_become_errors():
    async def slow():
        await asyncio.sleep(5)

    def crash():
        raise RuntimeError("boom")

    schema = {"type": "object", "properties": {}, "required": []}
    ex = ToolExecutor([Tool("slow", "", schema, slow), Tool("crash", "", schema, crash)], timeout_s=0.05)
    slow_result, crash_result = await ex.run_all([ToolCall("1", "slow", {}), ToolCall("2", "crash", {})])
    assert slow_result.is_error and "timed out" in slow_result.content
    assert crash_result.is_error and "RuntimeError" in crash_result.content


def test_validate_arguments_rejects_bools_as_numbers():
    schema = {"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]}
    assert validate_arguments(schema, {"n": 3}) is None
    assert validate_arguments(schema, {"n": True}) is not None
    assert validate_arguments(schema, "nope") is not None
