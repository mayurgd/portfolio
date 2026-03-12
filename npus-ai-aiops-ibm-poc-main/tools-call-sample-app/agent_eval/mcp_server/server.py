"""
AgentTool Eval App - Databricks MCP Server

FastMCP server that registers tools **dynamically** from ``tools_config.yaml``.

Each tool entry in the YAML maps to a Databricks table query tool.
The agent generates a SQL SELECT statement and passes it to the appropriate tool.
All tools share a single handler: ``handlers.run_databricks_query(query)``.

To add, remove, or modify tools — edit ``tools_config.yaml`` only.
No Python changes are required.

Start with::

    uv run python -m agent_eval.mcp_server.server          # Start server
    uv run python -m agent_eval.mcp_server.server --test   # Run integration tests
    uv run python -m agent_eval.mcp_server.server --list   # List registered tools
"""

import logging
import sys
import asyncio
from pathlib import Path
from typing import Dict, Any, List

import yaml
from mcp.server.fastmcp import FastMCP
from agent_eval.tools import handlers
from agent_eval.config import get_mcp_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-30s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)

_TOOLS_CONFIG_PATH = Path(__file__).parent / "tools_config.yaml"


def _load_tools_config(config_path: Path = _TOOLS_CONFIG_PATH) -> dict:
    """Load and validate the tools configuration YAML.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If required fields are missing or malformed.
    """
    if not config_path.exists():
        raise FileNotFoundError(
            f"Tools config not found: {config_path}\n"
            "Create tools_config.yaml next to server.py to define your tools."
        )

    with open(config_path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    if not isinstance(raw, dict):
        raise ValueError(f"tools_config.yaml must be a YAML mapping, got: {type(raw)}")

    tools = raw.get("tools")
    if not tools or not isinstance(tools, list):
        raise ValueError("tools_config.yaml must contain a non-empty 'tools' list.")

    for i, tool_def in enumerate(tools):
        if not isinstance(tool_def, dict):
            raise ValueError(f"Tool entry [{i}] must be a mapping, got: {type(tool_def)}")
        if not tool_def.get("name"):
            raise ValueError(f"Tool entry [{i}] is missing required field 'name'.")
        if not tool_def.get("description"):
            raise ValueError(
                f"Tool '{tool_def.get('name', i)}' is missing required field 'description'."
            )

    logger.info(
        "Loaded %d tool definition(s) from %s",
        len(tools),
        config_path,
    )
    return raw


def _build_tool_description(tool_def: dict) -> str:
    """Build a description string from a tool definition dict."""
    parts: List[str] = []

    desc = str(tool_def.get("description", "")).strip()
    if desc:
        parts.append(desc)

    schema = tool_def.get("schema")
    if schema and isinstance(schema, list):
        parts.append("\nSchema:")
        for col in schema:
            col_name = col.get("column", "?")
            col_type = col.get("type", "?")
            parts.append(f"  - {col_name}: {col_type}")

    table = tool_def.get("table", "")
    if table:
        parts.append(f"\nArgs:\n    query: A valid SQL SELECT statement targeting {table}")
    else:
        parts.append("\nArgs:\n    query: A valid SQL SELECT statement")

    return "\n".join(parts)


def _register_tools(mcp_server: FastMCP, tools_config: dict) -> int:
    """Register all tools from the config onto the FastMCP server.

    Each tool closure accepts a single ``query: str`` parameter and delegates
    to ``handlers.run_databricks_query(query)``.
    """
    tool_defs = tools_config.get("tools", [])
    registered = 0

    for tool_def in tool_defs:
        tool_name: str = tool_def["name"]
        description: str = _build_tool_description(tool_def)

        def _make_tool_fn(name: str):
            def tool_fn(query: str) -> Dict[str, Any]:
                return handlers.run_databricks_query(query)

            tool_fn.__name__ = name
            tool_fn.__qualname__ = name
            tool_fn.__doc__ = description
            return tool_fn

        fn = _make_tool_fn(tool_name)

        mcp_server.add_tool(
            fn,
            name=tool_name,
            description=description,
        )

        logger.info("Registered tool: %s", tool_name)
        registered += 1

    return registered


_mcp_config = get_mcp_config()
_tools_config = _load_tools_config()
_server_name: str = _tools_config.get("server", {}).get("name", "MCPDataServer")

mcp = FastMCP(
    _server_name,
    host=_mcp_config.host,
    port=_mcp_config.port,
)

_registered_count = _register_tools(mcp, _tools_config)
logger.info("MCP server '%s' ready with %d tool(s)", _server_name, _registered_count)


async def test_mcp_integration():
    """Run MCP server and tools integration tests. Returns True if all pass."""
    from langchain_mcp_adapters.client import MultiServerMCPClient

    expected_tool_count = len(_tools_config.get("tools", []))

    print("\n" + "=" * 70)
    print("MCP SERVER & DATABRICKS TOOLS INTEGRATION TEST")
    print("=" * 70)
    print(f"\nExpecting {expected_tool_count} tool(s) from tools_config.yaml")
    print("\nNote: Make sure the MCP server is running in another terminal:")
    print("  python3 -m agent_eval.mcp_server.server")

    test_results = {
        "server_connectivity": False,
        "tool_discovery": False,
        "tool_invocation": False,
        "error_handling": False,
    }

    print("\n[1/4] Testing MCP server connectivity and tool discovery...")
    try:
        mcp_client = MultiServerMCPClient({
            "data_server": {
                "url": _mcp_config.url,
                "transport": "streamable_http",
            }
        })

        tools = await mcp_client.get_tools()
        print("   ✓ Server connected successfully")
        print(f"   ✓ Discovered {len(tools)} tool(s):")
        for tool in tools:
            tool_name = getattr(tool, "name", "unknown")
            tool_desc = getattr(tool, "description", "")
            print(f"      - {tool_name}: {tool_desc[:70]}...")

        test_results["server_connectivity"] = True
        if len(tools) == expected_tool_count:
            test_results["tool_discovery"] = True
        else:
            print(
                f"   ⚠ Expected {expected_tool_count} tool(s) "
                f"(from tools_config.yaml), found {len(tools)}"
            )

    except Exception as e:
        print(f"   ✗ Connection failed: {e}")
        print("   → Make sure the server is running: python3 -m agent_eval.mcp_server.server")
        return False

    print("\n[2/4] Testing tool invocation with SQL queries...")
    config_tools = _tools_config.get("tools", [])
    test_cases = []
    for td in config_tools[:3]:
        tname = td["name"]
        table = td.get("table", "")
        if table:
            sql = f"SELECT * FROM {table} LIMIT 2"
        else:
            sql = "SELECT 1"
        test_cases.append({"name": tname, "args": {"query": sql}, "expected_keys": ["success"]})

    passed_tests = 0
    for test_case in test_cases:
        try:
            tool = next((t for t in tools if getattr(t, "name", "") == test_case["name"]), None)
            if not tool:
                print(f"   ✗ Tool '{test_case['name']}' not found")
                continue

            result = await tool.ainvoke(test_case["args"])

            if isinstance(result, list) and len(result) > 0:
                content_item = result[0]
                if hasattr(content_item, "text"):
                    result_text = content_item.text
                elif isinstance(content_item, dict) and "text" in content_item:
                    result_text = content_item["text"]
                else:
                    result_text = str(content_item)

                import json
                result_data = json.loads(result_text)
            elif isinstance(result, str):
                import json
                result_data = json.loads(result)
            else:
                result_data = result

            if isinstance(result_data, dict) and all(
                key in result_data for key in test_case["expected_keys"]
            ):
                print(
                    f"   ✓ {test_case['name']}: "
                    f"success={result_data.get('success', 'N/A')}, "
                    f"rows={result_data.get('row_count', 'N/A')}"
                )
                passed_tests += 1
            else:
                print(f"   ✗ {test_case['name']}: Missing expected keys or invalid format")

        except Exception as e:
            print(f"   ✗ {test_case['name']}: {e}")

    if passed_tests == len(test_cases):
        test_results["tool_invocation"] = True
    else:
        print(f"   ⚠ {passed_tests}/{len(test_cases)} tool invocation tests passed")

    print("\n[3/4] Testing error handling with invalid SQL...")
    try:
        first_tool_name = config_tools[0]["name"] if config_tools else None
        query_tool = next(
            (t for t in tools if getattr(t, "name", "") == first_tool_name), None
        ) if first_tool_name else None

        if query_tool:
            result = await query_tool.ainvoke(
                {"query": "SELECT * FROM nonexistent_table_xyz LIMIT 1"}
            )

            if isinstance(result, list) and len(result) > 0:
                content_item = result[0]
                if hasattr(content_item, "text"):
                    result_text = content_item.text
                elif isinstance(content_item, dict) and "text" in content_item:
                    result_text = content_item["text"]
                else:
                    result_text = str(content_item)

                import json
                result_data = json.loads(result_text)
            elif isinstance(result, str):
                import json
                result_data = json.loads(result)
            else:
                result_data = result

            if isinstance(result_data, dict) and not result_data.get("success"):
                print(f"   ✓ Error handling works: {str(result_data.get('error', ''))[:80]}")
                test_results["error_handling"] = True
            else:
                print("   ✗ Error handling failed: Expected error response for invalid SQL")
        else:
            print("   ✗ No tools available for error handling test")

    except Exception as e:
        print(f"   ✗ Error handling test failed: {e}")

    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    all_passed = all(test_results.values())
    for test_name, passed in test_results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}  {test_name.replace('_', ' ').title()}")

    print("=" * 70)
    if all_passed:
        print("✅ ALL TESTS PASSED - MCP server and tools are working correctly!")
    else:
        print("❌ SOME TESTS FAILED - Please review the errors above")
    print("=" * 70 + "\n")

    return all_passed


if __name__ == "__main__":
    if "--list" in sys.argv:
        print(f"\nServer: {_server_name}")
        print(f"Config: {_TOOLS_CONFIG_PATH}")
        print(f"\nRegistered tools ({_registered_count}):")
        for td in _tools_config.get("tools", []):
            table = td.get("table", "")
            table_info = f"  → {table}" if table else ""
            print(f"  • {td['name']}{table_info}")
        print()
        sys.exit(0)

    elif "--test" in sys.argv:
        logger.info("Running MCP integration tests...")
        success = asyncio.run(test_mcp_integration())
        sys.exit(0 if success else 1)

    else:
        import uvicorn

        logger.info(
            "Starting MCP server '%s' on %s:%s with %d tool(s)",
            _server_name,
            _mcp_config.host,
            _mcp_config.port,
            _registered_count,
        )
        logger.info("Config: %s", _TOOLS_CONFIG_PATH)
        logger.info("Run with --list to see registered tools")
        logger.info("Run with --test flag to test the integration")

        app = mcp.streamable_http_app()
        uvicorn.run(
            app,
            host=_mcp_config.host,
            port=_mcp_config.port,
            log_level="info",
            timeout_keep_alive=120,
        )


