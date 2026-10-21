"""Tool executor and the agent's tools.

Tool arguments come from the model, so they are validated against the tool's
schema before the handler runs. Every failure (unknown tool, bad arguments,
handler error, timeout) becomes an ``is_error`` result that goes back to the
LLM, which then explains the problem to the caller.

Tools: get_current_time, get_order_status (mock order service), calculate,
get_weather (Open-Meteo, optional) and end_call (per call, optional).
"""

from __future__ import annotations

import ast
import asyncio
import inspect
import json
import logging
import math
import operator
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Awaitable, Callable
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx

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
        return ToolResult(call.id, call.name, json.dumps(result, ensure_ascii=False))


def _error(call: ToolCall, message: str) -> ToolResult:
    return ToolResult(call.id, call.name, json.dumps({"error": message}, ensure_ascii=False), is_error=True)


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
        choices = properties[name].get("enum")
        if choices is not None and value not in choices:
            return f"argument {name!r} must be one of {', '.join(map(str, choices))}"
    return None


# ------------------------------------------------------------------ sample tools


def is_valid_timezone(name: str) -> bool:
    try:
        ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError):
        return False
    return True


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


_BINARY_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCTIONS: dict[str, Callable[..., float]] = {"sqrt": math.sqrt, "abs": abs, "round": round}
_MAX_MAGNITUDE = 1e100
_PERCENT = re.compile(r"(\d+(?:\.\d+)?)\s*%")


def calculate(expression: str) -> dict[str, Any]:
    """Evaluate plain arithmetic safely (no names, attributes or arbitrary calls)."""
    if len(expression) > 200:
        raise ToolError("That expression is too long.")
    text = expression.lower().replace("^", "**").replace("×", "*").replace("÷", "/").replace(",", "")
    text = _PERCENT.sub(r"(\1/100)", text).replace(" of ", " * ")  # "15% of 80"
    try:
        tree = ast.parse(text.strip(), mode="eval")
    except SyntaxError as exc:
        raise ToolError(f"I couldn't read {expression!r} as arithmetic.") from exc
    value = _evaluate(tree.body)
    if isinstance(value, float):
        value = int(value) if value.is_integer() else round(value, 10)
    return {"expression": expression, "result": value}


def _evaluate(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_evaluate(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPS:
        left, right = _evaluate(node.left), _evaluate(node.right)
        if isinstance(node.op, ast.Pow) and (abs(right) > 100 or abs(left) > 1e6):
            raise ToolError("That power is too large to compute.")
        try:
            result = _BINARY_OPS[type(node.op)](left, right)
        except ZeroDivisionError as exc:
            raise ToolError("You can't divide by zero.") from exc
        except OverflowError as exc:
            raise ToolError("That number is too large.") from exc
        if isinstance(result, complex) or abs(result) > _MAX_MAGNITUDE:
            raise ToolError("That result is out of range.")
        return result
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _FUNCTIONS
        and not node.keywords
    ):
        args = [_evaluate(arg) for arg in node.args]
        try:
            return _FUNCTIONS[node.func.id](*args)
        except (TypeError, ValueError) as exc:
            raise ToolError(f"{node.func.id} can't take those values.") from exc
    raise ToolError("Only numbers, + - * / % ** and sqrt, abs, round are supported.")


CALCULATOR_TOOL = Tool(
    name="calculate",
    description=(
        "Evaluate an arithmetic expression exactly: + - * / // % ** parentheses, percentages "
        "('15% of 80') and sqrt/abs/round. Use this instead of doing math in your head, for "
        "tips, totals, conversions and percentages."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "expression": {"type": "string", "description": "For example '(12.5 * 4) + 7' or '18% of 240'."}
        },
        "required": ["expression"],
        "additionalProperties": False,
    },
    handler=calculate,
)


GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
WEATHER_CODES = {
    0: "clear sky",
    1: "mainly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "fog",
    48: "freezing fog",
    51: "light drizzle",
    53: "drizzle",
    55: "heavy drizzle",
    56: "freezing drizzle",
    57: "heavy freezing drizzle",
    61: "light rain",
    63: "rain",
    65: "heavy rain",
    66: "freezing rain",
    67: "heavy freezing rain",
    71: "light snow",
    73: "snow",
    75: "heavy snow",
    77: "snow grains",
    80: "light rain showers",
    81: "rain showers",
    82: "violent rain showers",
    85: "snow showers",
    86: "heavy snow showers",
    95: "a thunderstorm",
    96: "a thunderstorm with hail",
    99: "a thunderstorm with heavy hail",
}


def make_get_weather(client: httpx.AsyncClient | None = None) -> Tool:
    """Current weather from Open-Meteo (free, no API key)."""

    async def get_weather(location: str, units: str = "celsius") -> dict[str, Any]:
        if units not in ("celsius", "fahrenheit"):
            raise ToolError("units must be 'celsius' or 'fahrenheit'.")
        http = client or httpx.AsyncClient(timeout=4.0)
        try:
            geo = await http.get(GEOCODE_URL, params={"name": location, "count": 1, "language": "en", "format": "json"})
            geo.raise_for_status()
            places = geo.json().get("results") or []
            if not places:
                raise ToolError(f"I couldn't find a place called {location!r}.")
            place = places[0]
            forecast = await http.get(
                FORECAST_URL,
                params={
                    "latitude": place["latitude"],
                    "longitude": place["longitude"],
                    "current": "temperature_2m,apparent_temperature,relative_humidity_2m,weather_code,wind_speed_10m",
                    "daily": "temperature_2m_max,temperature_2m_min",
                    "timezone": "auto",
                    "forecast_days": 1,
                    "temperature_unit": units,
                    "wind_speed_unit": "kmh" if units == "celsius" else "mph",
                },
            )
            forecast.raise_for_status()
            data = forecast.json()
        except httpx.HTTPError as exc:
            raise ToolError("The weather service isn't reachable right now.") from exc
        finally:
            if client is None:
                await http.aclose()
        current, daily = data["current"], data.get("daily", {})
        name = ", ".join(dict.fromkeys(p for p in (place.get("name"), place.get("admin1"), place.get("country")) if p))
        return {
            "location": name,
            "conditions": WEATHER_CODES.get(current.get("weather_code"), "unknown conditions"),
            "temperature": round(current["temperature_2m"]),
            "feels_like": round(current["apparent_temperature"]),
            "humidity_percent": current.get("relative_humidity_2m"),
            "wind": f"{round(current['wind_speed_10m'])} {'km/h' if units == 'celsius' else 'mph'}",
            "today_high": round(daily["temperature_2m_max"][0]) if daily.get("temperature_2m_max") else None,
            "today_low": round(daily["temperature_2m_min"][0]) if daily.get("temperature_2m_min") else None,
            "units": units,
        }

    return Tool(
        name="get_weather",
        description=(
            "Get the current weather and today's high and low for a city or place. Use this "
            "when the caller asks about the weather, temperature, rain or wind anywhere."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City or place name, for example 'Paris' or 'Austin, Texas'.",
                },
                "units": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "Defaults to celsius."},
            },
            "required": ["location"],
            "additionalProperties": False,
        },
        handler=get_weather,
    )


def make_end_call(on_end_call: Callable[[], None]) -> Tool:
    def end_call() -> dict[str, str]:
        on_end_call()
        return {
            "status": "The call ends once this reply has played. If you haven't said goodbye yet, say a short one now."
        }

    return Tool(
        name="end_call",
        description=(
            "Hang up the call. Use only when the caller says goodbye or clearly asks to end the "
            "call, and say a short goodbye in the same reply."
        ),
        input_schema={"type": "object", "properties": {}, "required": [], "additionalProperties": False},
        handler=end_call,
    )


def default_tools(
    default_timezone: str = "",
    *,
    weather: bool = False,
    on_end_call: Callable[[], None] | None = None,
    http_client: httpx.AsyncClient | None = None,
) -> list[Tool]:
    tools = [make_get_current_time(default_timezone), ORDER_STATUS_TOOL, CALCULATOR_TOOL]
    if weather:
        tools.append(make_get_weather(http_client))
    if on_end_call is not None:
        tools.append(make_end_call(on_end_call))
    return tools
