from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementState
import pandas as pd
import logging
from agent_eval.config import get_databricks_config

logger = logging.getLogger(__name__)

databricks_config = get_databricks_config()
warehouse_id = databricks_config.warehouse_id
catalog = databricks_config.catalog
schema = databricks_config.schema
auth_type = databricks_config.auth_type


def get_databricks_client() -> WorkspaceClient:
    if auth_type == "pat":
        raise ValueError(
            "PAT authentication is disabled for this project. "
            "Set DATABRICKS_AUTH_TYPE to a non-PAT method (for example azure-client-secret)."
        )

    client = WorkspaceClient(auth_type=auth_type)
    user_info = client.current_user.me()
    logger.info(
        "Authenticated as: %s  host: %s  auth_type: %s",
        user_info.user_name,
        client.config.host,
        client.config.auth_type,
    )
    return client

def statement_to_dataframe(response):
    if response.status is None:
        raise Exception("No status returned")
    if response.status.state != StatementState.SUCCEEDED:
        raise Exception(f"Query failed: {response.status.state}")
    if response.manifest is None:
        raise Exception("No manifest returned")
    if response.result is None:
        raise Exception("No result returned")

    manifest = response.manifest
    result = response.result
    schema = manifest.schema
    if schema is None:
        raise Exception("No schema returned")

    columns = [col.name for col in schema.columns]
    rows = result.data_array or []
    return pd.DataFrame(rows, columns=columns)


def execute_query(client, sql_query):
    logger.info("Executing query: %s", sql_query[:200])
    response = client.statement_execution.execute_statement(
        statement=sql_query,
        warehouse_id=warehouse_id,
        catalog=catalog,
        schema=schema,
        wait_timeout="30s",
    )
    df = statement_to_dataframe(response)
    logger.info("Query succeeded: %d rows returned", len(df))
    return df


if __name__ == "__main__":
    client = get_databricks_client()

    for w in client.warehouses.list():
        print("Name:", w.name)
        print("ID:", w.id)
        print("State:", w.state)
        print("-----")

    query = "SELECT * FROM sales_customers WHERE email_address = 'scollier@example.org' LIMIT 1"
    df = execute_query(client, query)
    print(df)