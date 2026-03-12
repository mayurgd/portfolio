"""Tool handler for the Bakehouse MCP server.

All Databricks table tools share a single entry point: run_databricks_query().
The agent generates a SQL SELECT statement and passes it here for execution.
"""

import logging
from typing import Dict, Any
from agent_eval.tools.db import get_databricks_client, execute_query

logger = logging.getLogger(__name__)


def run_databricks_query(query: str) -> Dict[str, Any]:
    """Execute a SQL SELECT query against Databricks and return results.

    Args:
        query: A valid SQL SELECT statement targeting a samples.bakehouse.* table.

    Returns:
        dict with keys: success, data, row_count, columns, query, error (on failure)
    """
    logger.info("run_databricks_query called with query: %s", query[:200])

    try:
        client = get_databricks_client()
        df = execute_query(client, query)

        rows = df.to_dict(orient="records")
        columns = list(df.columns)

        logger.info(
            "run_databricks_query succeeded: %d rows, %d columns",
            len(rows), len(columns),
        )

        return {
            "success": True,
            "data": rows,
            "row_count": len(rows),
            "columns": columns,
            "query": query,
        }

    except Exception as e:
        logger.error("run_databricks_query failed: %s", e)
        return {
            "success": False,
            "error": str(e),
            "query": query,
            "data": [],
            "row_count": 0,
            "columns": [],
        }


