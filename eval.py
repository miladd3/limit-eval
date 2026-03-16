#!/usr/bin/env python3
"""
Compares limit-agent-tools vs limit-agent-mcp using LLM-as-a-judge via Arize Phoenix.

Prerequisites:
  - docker compose up -d          (Phoenix on :6006)
  - limit-api running on  :2010
  - limit-mcp running on  :2009
  - cp .env.example .env && fill in OPENAI_API_KEY
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import Any, Optional

import httpx
import pandas as pd
from agents import Agent, Runner, SQLiteSession, function_tool, gen_trace_id, trace
from agents.mcp import MCPServerStreamableHttp
from dotenv import load_dotenv
from openinference.instrumentation.openai_agents import OpenAIAgentsInstrumentor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

load_dotenv()

# ── Phoenix tracing setup ───────────────────────────────────────────────────────
PHOENIX_ENDPOINT = os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006")

tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(
    SimpleSpanProcessor(OTLPSpanExporter(endpoint=f"{PHOENIX_ENDPOINT}/v1/traces"))
)
OpenAIAgentsInstrumentor().instrument(tracer_provider=tracer_provider)

# ── Config ──────────────────────────────────────────────────────────────────────
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gpt-4.1")
LIMIT_API_BASE_URL = os.getenv("LIMIT_API_BASE_URL", "http://127.0.0.1:2010")
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://127.0.0.1:2009/mcp")

os.environ.setdefault("OPENAI_API_KEY", os.environ["OPENAI_API_KEY"])

# ── Shared system prompt ────────────────────────────────────────────────────────
SYSTEM_INSTRUCTIONS = """You are a helpful banking assistant that helps users manage
their debit card limits for POS (payment), ATM (withdrawal), and E-commerce transactions.
Use only your tools as the source of truth. Never hardcode values."""

# ── Test dataset ────────────────────────────────────────────────────────────────
TEST_CASES = [
    "Show my current card limits",
    "What cards do I have and what are their limits?",
    "What is my ATM withdrawal limit?",
    "What is my POS limit?",
    "What is my ecommerce limit?",
]

# ── Eval template ────────────────────────────────────────────────────────────────
EVAL_TEMPLATE = """
You are evaluating a banking assistant that helps users check and manage debit card limits.

[User question]
{input}

[Agent response]
{output}

[Evaluation criteria]
- "correct"   → Response directly answers the question using real card data (e.g. actual limit numbers, masked card numbers). No hallucination.
- "partial"   → Response is on topic but vague, incomplete, or asks a clarifying question instead of answering.
- "incorrect" → Response fails to answer, contains errors, or ignores the question entirely.

Respond with exactly one word: correct, partial, or incorrect.
"""


# ── Helpers ──────────────────────────────────────────────────────────────────────
class LimitApiClient:
    def __init__(self, base_url: str, timeout: int = 15) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    async def request(self, method: str, path: str, payload: Optional[dict] = None) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            try:
                r = await client.request(method, f"{self._base_url}{path}", json=payload)
                r.raise_for_status()
                return r.json()
            except Exception as exc:
                return {"error": str(exc)}


def build_tools(client: LimitApiClient):
    @function_tool
    async def get_payment_instruments() -> dict[str, Any]:
        """Fetch all accounts and cards with current limits."""
        return await client.request("GET", "/accounts")

    @function_tool
    async def get_current_limits(card_id: str) -> dict[str, Any]:
        """Get current limits for a specific card."""
        return await client.request("GET", f"/cards/{card_id}/limits")

    @function_tool
    async def get_default_card_limits() -> dict[str, Any]:
        """Get current limits for the default card."""
        return await client.request("GET", "/cards/default/limits")

    return [get_payment_instruments, get_current_limits, get_default_card_limits]


# ── Agent runners ─────────────────────────────────────────────────────────────────
async def run_tools_agent(prompt: str) -> str:
    """Run limit-agent-tools and return final output."""
    client = LimitApiClient(LIMIT_API_BASE_URL)
    agent = Agent(
        name="Debit Card Limit Assistant (Tools)",
        instructions=SYSTEM_INSTRUCTIONS,
        model=OPENAI_MODEL,
        tools=build_tools(client),
    )
    session = SQLiteSession(session_id=f"eval-tools-{prompt[:20].replace(' ', '_')}")
    with trace(workflow_name="eval:limit-agent-tools"):
        result = await Runner.run(agent, prompt, session=session)
    return str(result.final_output or "").strip()


async def run_mcp_agent(prompt: str) -> str:
    """Run limit-agent-mcp and return final output."""
    async with MCPServerStreamableHttp(
        name="card_limit_manager",
        params={"url": MCP_SERVER_URL, "timeout": 15, "sse_read_timeout": 60},
        require_approval="never",
        cache_tools_list=True,
    ) as server:
        agent = Agent(
            name="Debit Card Limit Assistant (MCP)",
            instructions=SYSTEM_INSTRUCTIONS,
            model=OPENAI_MODEL,
            mcp_servers=[server],
        )
        session = SQLiteSession(session_id=f"eval-mcp-{prompt[:20].replace(' ', '_')}")
        with trace(workflow_name="eval:limit-agent-mcp"):
            result = await Runner.run(agent, prompt, session=session)
    return str(result.final_output or "").strip()


# ── Main ──────────────────────────────────────────────────────────────────────────
async def collect_results() -> pd.DataFrame:
    rows = []
    for i, prompt in enumerate(TEST_CASES, 1):
        print(f"[{i}/{len(TEST_CASES)}] {prompt}")
        tools_out = await run_tools_agent(prompt)
        mcp_out = await run_mcp_agent(prompt)
        print(f"  tools → {tools_out[:80]}...")
        print(f"  mcp   → {mcp_out[:80]}...")
        rows.append({"input": prompt, "tools_output": tools_out, "mcp_output": mcp_out})
    return pd.DataFrame(rows)


def run_evals(df: pd.DataFrame) -> pd.DataFrame:
    from phoenix.evals import OpenAIModel, llm_classify

    judge = OpenAIModel(model=JUDGE_MODEL, api_key=os.environ["OPENAI_API_KEY"])

    tools_df = df[["input", "tools_output"]].rename(columns={"tools_output": "output"})
    mcp_df = df[["input", "mcp_output"]].rename(columns={"mcp_output": "output"})

    print("\nRunning judge evals...")
    tools_evals = llm_classify(tools_df, judge, EVAL_TEMPLATE, rails=["correct", "partial", "incorrect"], provide_explanation=True)
    mcp_evals = llm_classify(mcp_df, judge, EVAL_TEMPLATE, rails=["correct", "partial", "incorrect"], provide_explanation=True)

    df["tools_label"] = tools_evals["label"].values
    df["mcp_label"] = mcp_evals["label"].values
    df["tools_explanation"] = tools_evals.get("explanation", pd.Series([""] * len(df))).values
    df["mcp_explanation"] = mcp_evals.get("explanation", pd.Series([""] * len(df))).values
    return df


def print_summary(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("EVAL RESULTS")
    print("=" * 70)
    for _, row in df.iterrows():
        print(f"\nQ: {row['input']}")
        print(f"  tools → [{row['tools_label']}]  {row['tools_explanation']}")
        print(f"  mcp   → [{row['mcp_label']}]  {row['mcp_explanation']}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for agent, col in [("limit-agent-tools", "tools_label"), ("limit-agent-mcp", "mcp_label")]:
        counts = df[col].value_counts()
        correct = counts.get("correct", 0)
        total = len(df)
        print(f"  {agent}: {correct}/{total} correct  |  {counts.to_dict()}")


async def main() -> None:
    print(f"Phoenix UI: {PHOENIX_ENDPOINT}")
    print(f"Running {len(TEST_CASES)} test cases through both agents...\n")

    df = await collect_results()
    df = run_evals(df)
    print_summary(df)

    out_path = Path(__file__).parent / "results.csv"
    df.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
