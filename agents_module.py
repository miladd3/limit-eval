import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

from agents import Agent
from agents.mcp import MCPServerStreamableHttp

from config import (
    LIMIT_API_BASE_URL,
    MCP_SERVER_URL,
    OPENAI_MODEL,
    mcp_agent_mod,
    tools_agent_mod,
)


def build_tools_agent() -> Agent:
    client = tools_agent_mod.LimitApiClient(
        base_url=LIMIT_API_BASE_URL,
        timeout_seconds=int(os.getenv("LIMIT_API_TIMEOUT", "15")),
    )
    tools = tools_agent_mod.build_tools(client)
    return Agent(
        name="Debit Card Limit Assistant (Tools)",
        instructions=tools_agent_mod.SYSTEM_INSTRUCTIONS,
        model=OPENAI_MODEL,
        tools=tools,
    )


@asynccontextmanager
async def build_mcp_agent() -> AsyncIterator[Agent]:
    mcp_timeout = int(os.getenv("MCP_TIMEOUT", "15"))
    mcp_sse_read_timeout = int(os.getenv("MCP_SSE_READ_TIMEOUT", "300"))
    async with MCPServerStreamableHttp(
        name=os.getenv("MCP_SERVER_LABEL", "card_limit_manager"),
        params={
            "url": MCP_SERVER_URL,
            "timeout": mcp_timeout,
            "sse_read_timeout": mcp_sse_read_timeout,
        },
        require_approval="never",
        cache_tools_list=True,
        max_retry_attempts=2,
        retry_backoff_seconds_base=2.0,
        client_session_timeout_seconds=mcp_timeout,
    ) as server:
        yield Agent(
            name="Debit Card Limit Assistant (MCP)",
            instructions=mcp_agent_mod.SYSTEM_INSTRUCTIONS,
            model=OPENAI_MODEL,
            mcp_servers=[server],
        )
