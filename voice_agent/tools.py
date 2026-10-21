"""Tool executor and the sample tools.

Tool arguments come from the model, so they are validated against the tool's
schema before the handler runs. Every failure (unknown tool, bad arguments,
handler error, timeout) becomes an ``is_error`` result that goes back to the
LLM, which then explains the problem to the caller.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Awaitable, Callable
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .providers.base import ToolCall, ToolResult, ToolSpec

log = logging.getLogger(__name__)


class ToolError(Exception):
    """An expected tool failure whose message is safe to show the model."""


@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[..., Any | Awaitable[Any]]

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(self.name, self.description, self.input_schema)


class ToolExecutor:
    def __init__(self, tools: list[Tool], timeout_s: float = 5.0) -> None:
        self._tools = {tool.name: tool for tool in tools}
        self._timeout_s = timeout_s

    @property
    def specs(self) -> list[ToolSpec]:
        return [tool.spec for tool in self._tools.values()]

    async def run_all(self, calls: list[ToolCall]) -> list[ToolResult]:
        return list(await asyncio.gather(*(self.run(call) for call in calls)))

    async def run(self, call: ToolCall) -> ToolResult:
        tool = self._tools.get(call.name)
        if tool is None:
            return _error(call, f"Unknown tool {call.name!r}.")
        problem = validate_arguments(tool.input_schema, call.arguments)
        if problem:
            return _error(call, f"Invalid arguments: {problem}")
        try:
            async with asyncio.timeout(self._timeout_s):
                result = tool.handler(**call.arguments)
                if inspect.isawaitable(result):
                    result = await result
        except ToolError as exc:
            return _error(call, str(exc))
        except TimeoutError:
            return _error(call, f"The {call.name} tool timed out.")
        except Exception as exc:  # a bug in a tool must not take down the call
            log.exception("tool %s crashed", call.name)
            return _error(call, f"The {call.name} tool failed ({type(exc).__name__}).")
        return ToolResult(call.id, call.name, json.dumps(result))


def _error(call: ToolCall, message: str) -> ToolResult:
    return ToolResult(call.id, call.name, json.dumps({"error": message}), is_error=True)


_JSON_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "object": (dict,),
    "array": (list,),
}


def validate_arguments(schema: dict[str, Any], args: Any) -> str | None:
    """Check ``args`` against the flat object schemas the tools use."""
    if not isinstance(args, dict):
        return "arguments must be a JSON object"
    properties: dict[str, Any] = schema.get("properties", {})
    for name in schema.get("required", []):
        if name not in args:
            return f"missing required argument {name!r}"
    for name, value in args.items():
        if name not in properties:
            if schema.get("additionalProperties", True) is False:
                return f"unexpected argument {name!r}"
            continue
        expected = properties[name].get("type")
        allowed = _JSON_TYPES.get(expected)
        if allowed is None:
            continue
        if isinstance(value, bool) and expected in {"integer", "number"}:
            return f"argument {name!r} must be {expected}"
        if not isinstance(value, allowed):
            return f"argument {name!r} must be {expected}"
    return None


# ------------------------------------------------------------------ sample tools


def make_get_current_time(default_timezone: str = "") -> Tool:
    def get_current_time(timezone: str | None = None) -> dict[str, Any]:
        tz_name = timezone or default_timezone
        if tz_name:
            try:
                now = datetime.now(ZoneInfo(tz_name))
            except (ZoneInfoNotFoundError, ValueError) as exc:
                raise ToolError(f"Unknown time zone {tz_name!r}. Use an IANA name like 'Asia/Kolkata'.") from exc
        else:
            now = datetime.now().astimezone()
        hour = now.strftime("%I").lstrip("0") or "12"
        return {
            "timezone": tz_name or str(now.tzinfo),
            "iso": now.isoformat(timespec="seconds"),
            "spoken": f"{hour}:{now:%M %p} on {now:%A, %B} {now.day}",
        }

    return Tool(
        name="get_current_time",
        description=(
            "Get the current date and time. Use this whenever the caller asks what time "
            "or day it is, anywhere in the world. Pass an IANA time zone for a specific "
            "place; omit it for the caller's local time."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string",
                    "description": "IANA time zone name, for example 'America/New_York' or 'Asia/Kolkata'.",
                }
            },
            "required": [],
            "additionalProperties": False,
        },
        handler=get_current_time,
    )


# A stand-in for an order service.
MOCK_ORDERS: dict[str, dict[str, Any]] = {
    "A1001": {"status": "shipped", "carrier": "UPS", "estimated_delivery": "Friday"},
    "A1002": {"status": "processing", "estimated_ship_date": "tomorrow"},
    "A1003": {"status": "delivered", "delivered_on": "Monday", "left_at": "front door"},
    "A1004": {"status": "cancelled", "refund": "issued to the original payment method"},
}


def get_order_status(order_id: str) -> dict[str, Any]:
    normalized = re.sub(r"[\s\-#]", "", order_id).upper()
    order = MOCK_ORDERS.get(normalized)
    if order is None:
        raise ToolError(f"No order found with ID {order_id!r}.")
    return {"order_id": normalized, **order}


ORDER_STATUS_TOOL = Tool(
    name="get_order_status",
    description=(
        "Look up the status of a customer's order by its order ID (letters and digits, "
        "for example A1001). Use this when the caller asks where their order is, when it "
        "will arrive, or whether it shipped."
    ),
    input_schema={
        "type": "object",
        "properties": {"order_id": {"type": "string", "description": "The order ID, for example 'A1001'."}},
        "required": ["order_id"],
        "additionalProperties": False,
    },
    handler=get_order_status,
)


def default_tools(default_timezone: str = "") -> list[Tool]:
    return [make_get_current_time(default_timezone), ORDER_STATUS_TOOL]
