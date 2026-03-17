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
from pathlib import Path
import pandas as pd
from agents import Agent, Runner, SQLiteSession, gen_trace_id, trace
from agents.mcp import MCPServerStreamableHttp
from dotenv import load_dotenv
from openinference.instrumentation.openai_agents import OpenAIAgentsInstrumentor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

# ── Import real agent modules ────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

import importlib.util

def _load_module(name: str, filepath: Path):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

tools_agent_mod = _load_module("tools_agent", PROJECT_ROOT / "limit-agent-using-tools" / "agent.py")
mcp_agent_mod = _load_module("mcp_agent", PROJECT_ROOT / "limit-agent-using-mcp" / "agent.py")

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


# ── Agent runners (using real agent modules) ─────────────────────────────────────
async def run_tools_agent(prompt: str) -> str:
    """Run limit-agent-tools with real system instructions and all 5 tools."""
    client = tools_agent_mod.LimitApiClient(
        base_url=LIMIT_API_BASE_URL,
        timeout_seconds=int(os.getenv("LIMIT_API_TIMEOUT", "15")),
    )
    tools = tools_agent_mod.build_tools(client)
    agent = Agent(
        name="Debit Card Limit Assistant (Tools)",
        instructions=tools_agent_mod.SYSTEM_INSTRUCTIONS,
        model=OPENAI_MODEL,
        tools=tools,
    )
    session = SQLiteSession(session_id=f"eval-tools-{prompt[:20].replace(' ', '_')}")
    with trace(workflow_name="eval:limit-agent-tools"):
        result = await Runner.run(agent, prompt, session=session)
    return str(result.final_output or "").strip()


async def run_mcp_agent(prompt: str) -> str:
    """Run limit-agent-mcp with real system instructions and MCP config."""
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
        agent = Agent(
            name="Debit Card Limit Assistant (MCP)",
            instructions=mcp_agent_mod.SYSTEM_INSTRUCTIONS,
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
