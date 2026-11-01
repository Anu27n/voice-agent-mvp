"""The calculator, weather and end_call tools, and the mock LLM's use of them."""

import json

import httpx
import pytest

from voice_agent.providers.base import ToolCall
from voice_agent.providers.mock import _arithmetic
from voice_agent.tools import (
    FORECAST_URL,
    GEOCODE_URL,
    ToolError,
    ToolExecutor,
    calculate,
    default_tools,
    make_get_weather,
)


@pytest.mark.parametrize(
    "expression, result",
    [
        ("2 + 2", 4),
        ("(12.5 * 4) + 7", 57),
        ("18% of 240", 43.2),
        ("15 % of 80", 12),
        ("2^10", 1024),
        ("1,000 / 8", 125),
        ("sqrt(144) + abs(-3)", 15),
        ("7 // 2", 3),
        ("10 / 4", 2.5),
        ("-3 ** 2", -9),
        ("0.1 + 0.2", 0.3),
    ],
)
def test_calculate(expression, result):
    assert calculate(expression)["result"] == result


@pytest.mark.parametrize(
    "expression, message",
    [
        ("1 / 0", "divide by zero"),
        ("9 ** 999", "too large"),
        ("10 ** 50 * 10 ** 60", "out of range"),
        ("__import__('os').system('ls')", "Only numbers"),
        ("open('x')", "Only numbers"),
        ("2 +", "couldn't read"),
        ("sqrt(-1)", "can't take"),
        ("1" * 201, "too long"),
    ],
)
def test_calculate_rejects_unsafe_or_invalid_input(expression, message):
    with pytest.raises(ToolError, match=message):
        calculate(expression)


def weather_client(geocode: dict, forecast: dict | None = None, status: int = 200) -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url).startswith(GEOCODE_URL):
            return httpx.Response(status, json=geocode)
        assert str(request.url).startswith(FORECAST_URL)
        assert request.url.params["temperature_unit"] in ("celsius", "fahrenheit")
        return httpx.Response(200, json=forecast)

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


PARIS = {
    "results": [{"name": "Paris", "admin1": "Ile-de-France", "country": "France", "latitude": 48.85, "longitude": 2.35}]
}
FORECAST = {
    "current": {
        "temperature_2m": 11.9,
        "apparent_temperature": 8.3,
        "relative_humidity_2m": 63,
        "weather_code": 2,
        "wind_speed_10m": 16.3,
    },
    "daily": {"temperature_2m_max": [14.5], "temperature_2m_min": [8.3]},
}


async def test_weather_reports_current_conditions():
    tool = make_get_weather(weather_client(PARIS, FORECAST))
    result = await ToolExecutor([tool]).run(ToolCall("1", "get_weather", {"location": "Paris"}))
    assert not result.is_error
    assert json.loads(result.content) == {
        "location": "Paris, Ile-de-France, France",
        "conditions": "partly cloudy",
        "temperature": 12,
        "feels_like": 8,
        "humidity_percent": 63,
        "wind": "16 km/h",
        "today_high": 14,
        "today_low": 8,
        "units": "celsius",
    }


async def test_weather_unknown_place_and_outage_are_tool_errors():
    unknown = await ToolExecutor([make_get_weather(weather_client({"results": []}))]).run(
        ToolCall("1", "get_weather", {"location": "Atlantis"})
    )
    assert unknown.is_error and "couldn't find" in unknown.content

    outage = await ToolExecutor([make_get_weather(weather_client({}, status=503))]).run(
        ToolCall("2", "get_weather", {"location": "Paris"})
    )
    assert outage.is_error and "isn't available" in outage.content

    bad_units = await ToolExecutor([make_get_weather(weather_client(PARIS, FORECAST))]).run(
        ToolCall("3", "get_weather", {"location": "Paris", "units": "kelvin"})
    )
    assert bad_units.is_error and "must be one of" in bad_units.content


async def test_end_call_tool_only_flags_the_hang_up():
    calls = []
    tools = {t.name: t for t in default_tools(on_end_call=lambda: calls.append(1))}
    result = await ToolExecutor(list(tools.values())).run(ToolCall("1", "end_call", {}))
    assert calls == [1] and not result.is_error
    assert "goodbye" in json.loads(result.content)["status"]


def test_default_tool_set():
    assert [t.name for t in default_tools()] == ["get_current_time", "get_order_status", "calculate"]
    names = [t.name for t in default_tools(weather=True, on_end_call=lambda: None)]
    assert names == ["get_current_time", "get_order_status", "calculate", "get_weather", "end_call"]


@pytest.mark.parametrize(
    "spoken, expression",
    [
        ("what is 18 percent of 240?", "18 % of 240"),
        ("what's 12 times 7", "12 * 7"),
        ("can you calculate 100 divided by 8", "100 / 8"),
        ("what is 3 x 4?", "3 * 4"),
        ("what time is it?", None),
        ("check order a1001", None),
    ],
)
def test_mock_llm_spots_arithmetic(spoken, expression):
    assert _arithmetic(spoken) == expression


async def test_weather_retries_a_network_blip_once():
    attempts = {"geo": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url).startswith(GEOCODE_URL):
            attempts["geo"] += 1
            if attempts["geo"] == 1:
                raise httpx.ConnectError("connection reset")
            return httpx.Response(200, json=PARIS)
        return httpx.Response(200, json=FORECAST)

    tool = make_get_weather(httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    result = await ToolExecutor([tool]).run(ToolCall("1", "get_weather", {"location": "Paris"}))
    assert not result.is_error and attempts["geo"] == 2


async def test_weather_rate_limit_is_reported_not_retried(caplog):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(429, text="Too many requests")

    tool = make_get_weather(httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    result = await ToolExecutor([tool]).run(ToolCall("1", "get_weather", {"location": "Paris"}))
    assert result.is_error and "isn't available" in result.content
    assert len(calls) == 1
    assert "HTTP 429" in caplog.text
