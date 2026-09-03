"""Tests for the tool registry."""

import json

import pytest

from harness.tools.registry import ToolRegistry


class TestToolRegistry:
    def setup_method(self):
        self.registry = ToolRegistry()

    def test_register_function(self):
        def greet(name: str) -> str:
            """Say hello."""
            return f"Hello, {name}!"

        self.registry.register(greet)
        assert "greet" in self.registry.tool_names
        assert len(self.registry) == 1

    def test_register_with_custom_name(self):
        def my_func(x: int) -> str:
            return str(x)

        self.registry.register(my_func, name="custom_tool", description="A custom tool")
        assert "custom_tool" in self.registry.tool_names

    def test_decorator_registration(self):
        @self.registry.tool("Search for things")
        def search(query: str, limit: int = 5) -> str:
            return f"Results for {query}"

        assert "search" in self.registry.tool_names
        defs = self.registry.get_definitions()
        assert len(defs) == 1
        assert defs[0].function.name == "search"
        assert defs[0].function.description == "Search for things"

    def test_get_definitions_schema(self):
        @self.registry.tool("Add numbers")
        def add(a: int, b: int) -> int:
            return a + b

        defs = self.registry.get_definitions()
        params = defs[0].function.parameters
        assert params["type"] == "object"
        assert "a" in params["properties"]
        assert "b" in params["properties"]
        assert params["properties"]["a"]["type"] == "integer"
        assert "a" in params["required"]
        assert "b" in params["required"]

    def test_optional_parameters(self):
        @self.registry.tool("Search")
        def search(query: str, limit: int = 10) -> str:
            return query

        defs = self.registry.get_definitions()
        params = defs[0].function.parameters
        assert "query" in params["required"]
        assert "limit" not in params.get("required", [])

    @pytest.mark.asyncio
    async def test_execute_sync_function(self):
        @self.registry.tool("Greet")
        def greet(name: str) -> str:
            return f"Hello, {name}!"

        result = await self.registry.execute("greet", {"name": "World"})
        assert result == "Hello, World!"

    @pytest.mark.asyncio
    async def test_execute_async_function(self):
        @self.registry.tool("Async greet")
        async def async_greet(name: str) -> str:
            return f"Hello, {name}!"

        result = await self.registry.execute("async_greet", {"name": "Async"})
        assert result == "Hello, Async!"

    @pytest.mark.asyncio
    async def test_execute_unknown_tool(self):
        result = await self.registry.execute("nonexistent", {})
        parsed = json.loads(result)
        assert "error" in parsed

    @pytest.mark.asyncio
    async def test_execute_dict_result_serialized(self):
        @self.registry.tool("Get data")
        def get_data() -> dict:
            return {"key": "value", "count": 42}

        result = await self.registry.execute("get_data", {})
        parsed = json.loads(result)
        assert parsed == {"key": "value", "count": 42}

    @pytest.mark.asyncio
    async def test_execute_error_handling(self):
        @self.registry.tool("Fail")
        def fail() -> str:
            raise ValueError("Something went wrong")

        result = await self.registry.execute("fail", {})
        parsed = json.loads(result)
        assert "error" in parsed
