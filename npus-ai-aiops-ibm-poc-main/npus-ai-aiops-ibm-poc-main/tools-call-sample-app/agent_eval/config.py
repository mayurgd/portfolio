"""Central configuration."""

import os
import logging
from dataclasses import dataclass
from typing import Union
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OpenAIConfig:
    api_key: str = ""
    model: str = "gpt-4o-mini"
    base_url: str = ""
    temperature: float = 0.3
    max_tokens: int = 1024


@dataclass(frozen=True)
class LangfuseConfig:
    public_key: str = ""
    secret_key: str = ""
    host: str = "https://cloud.langfuse.com"
    # True (default), False (disable verification), or a path to a CA bundle .pem file.
    ssl_verify: Union[bool, str] = True

    @property
    def enabled(self) -> bool:
        return bool(self.public_key and self.secret_key)


@dataclass(frozen=True)
class MCPConfig:
    host: str = "0.0.0.0"
    port: int = 8052

    @property
    def url(self) -> str:
        return f"http://localhost:{self.port}/mcp"


@dataclass(frozen=True)
class DatabricksConfig:
    warehouse_id: str = ""
    catalog: str = "samples"
    schema: str = "bakehouse"
    auth_type: str = "azure-client-secret"


@dataclass(frozen=True)
class NESGENConfig:
    client_id: str
    client_secret: str
    model_endpoint: str
    api_base: str
    model: str
    api_version: str
    embedding_api_base: str
    embedding_model: str
    embedding_api_version: str
    openai_api_key_fallback: str


@dataclass(frozen=True)
class Config:
    nesgen: NESGENConfig
    openai: OpenAIConfig
    langfuse: LangfuseConfig
    mcp: MCPConfig
    databricks: DatabricksConfig


def get_openai_config() -> OpenAIConfig:
    return OpenAIConfig(
        api_key=os.getenv("OPENAI_API_KEY", ""),
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        base_url=os.getenv("OPENAI_BASE_URL", ""),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
        max_tokens=int(os.getenv("LLM_MAX_TOKENS", "1024")),
    )


def get_langfuse_config() -> LangfuseConfig:
    ssl_cert = os.getenv("LANGFUSE_SSL_CERT", "").strip()
    _ssl_raw = os.getenv("LANGFUSE_SSL_VERIFY", "true").strip().lower()

    ssl_verify: Union[bool, str]
    if ssl_cert:
        ssl_verify = ssl_cert          # path to CA bundle .pem
    else:
        ssl_verify = _ssl_raw not in ("0", "false", "no")

    return LangfuseConfig(
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY", ""),
        secret_key=os.getenv("LANGFUSE_SECRET_KEY", ""),
        host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        ssl_verify=ssl_verify,
    )


def get_mcp_config() -> MCPConfig:
    return MCPConfig(
        host=os.getenv("MCP_SERVER_HOST", "0.0.0.0"),
        port=int(os.getenv("MCP_SERVER_PORT", "8052")),
    )


def get_databricks_config() -> DatabricksConfig:
    return DatabricksConfig(
        warehouse_id=os.getenv("WAREHOUSE_ID", ""),
        catalog=os.getenv("DATABRICKS_CATALOG", "samples"),
        schema=os.getenv("DATABRICKS_SCHEMA", "bakehouse"),
        auth_type=os.getenv("DATABRICKS_AUTH_TYPE", "azure-client-secret"),
    )


def get_nesgen_config() -> NESGENConfig:
    return NESGENConfig(
        client_id=os.getenv("NESTLE_CLIENT_ID", ""),
        client_secret=os.getenv("NESTLE_CLIENT_SECRET", ""),
        model_endpoint=os.getenv("NESGEN_URL", ""),
        api_base=os.getenv("NESGEN_API_BASE", "https://eur-sdr-int-pub.nestle.com/api/dv-exp-sandbox-openai-api/1/openai/deployments"),
        model=os.getenv("NESGEN_MODEL", "gpt-5.1"),
        api_version=os.getenv("NESGEN_API_VERSION", "2024-02-01"),
        embedding_api_base=os.getenv("NESGEN_EMBEDDING_API_BASE", "https://eur-sdr-int-pub.nestle.com/api/dv-exp-sandbox-openai-api/1"),
        embedding_model=os.getenv("NESGEN_EMBEDDING_MODEL", "text-embedding-3-small"),
        embedding_api_version=os.getenv("NESGEN_EMBEDDING_API_VERSION", "2024-10-21"),
        openai_api_key_fallback=os.getenv("OPENAI_API_KEY", "")
    )


def get_nestle_credentials() -> tuple[str, str]:
    """Return (client_id, client_secret) or raise if missing."""
    client_id = os.getenv("NESTLE_CLIENT_ID", "")
    client_secret = os.getenv("NESTLE_CLIENT_SECRET", "")
    if not client_id or not client_secret:
        raise ValueError(
            "NESTLE_CLIENT_ID and NESTLE_CLIENT_SECRET environment variables are required"
        )
    return client_id, client_secret


config = Config(
    nesgen=get_nesgen_config(),
    openai=get_openai_config(),
    langfuse=get_langfuse_config(),
    mcp=get_mcp_config(),
    databricks=get_databricks_config(),
)
