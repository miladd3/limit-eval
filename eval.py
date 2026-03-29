#!/usr/bin/env python3
"""
Compares limit-agent-tools vs limit-agent-mcp using LLM-as-a-judge via Arize Phoenix.

Prerequisites:
  - phoenix serve                 (Phoenix on :6006)
  - limit-api running on  :2010
  - limit-mcp running on  :2009
  - fill in OPENAI_API_KEY in .env
"""

import argparse
import asyncio
import importlib.util
import json
import os
import re
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator
from uuid import uuid4

import pandas as pd
from agents import Agent, Runner, SQLiteSession, trace
from agents.mcp import MCPServerStreamableHttp
from dotenv import load_dotenv
from openinference.instrumentation.openai_agents import OpenAIAgentsInstrumentor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TEST_CASES_PATH = Path(__file__).resolve().parent / "test_cases.json"
DEFAULT_RESULTS_PATH = Path(__file__).resolve().parent / "results.csv"
RUN_ID = uuid4().hex[:8]


def _load_module(name: str, filepath: Path):
    spec = importlib.util.spec_from_file_location(name, filepath)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {filepath}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


tools_agent_mod = _load_module(
    "tools_agent",
    PROJECT_ROOT / "limit-agent-using-tools" / "agent.py",
)
mcp_agent_mod = _load_module(
    "mcp_agent",
    PROJECT_ROOT / "limit-agent-using-mcp" / "agent.py",
)

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Missing required environment variable: OPENAI_API_KEY")
os.environ.setdefault("OPENAI_API_KEY", OPENAI_API_KEY)

# Phoenix tracing setup
PHOENIX_ENDPOINT = os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006")

tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(
    SimpleSpanProcessor(OTLPSpanExporter(endpoint=f"{PHOENIX_ENDPOINT}/v1/traces"))
)
OpenAIAgentsInstrumentor().instrument(tracer_provider=tracer_provider)

# Config
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gpt-4.1")
LIMIT_API_BASE_URL = os.getenv("LIMIT_API_BASE_URL", "http://127.0.0.1:2010")
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://127.0.0.1:2009/mcp")
JUDGE_CONCURRENCY = int(os.getenv("JUDGE_CONCURRENCY", "8"))

EVAL_TEMPLATE = """
You are evaluating a banking assistant that helps users check and manage debit card limits.

[Conversation history before the current user turn]
{conversation_history}

[Current user message]
{latest_user_input}

[Optional evaluation focus]
{evaluation_focus}

[Optional expected outcome]
{expected_outcome}

[Agent response]
{output}

[Evaluation criteria]
- "correct"   -> Response correctly addresses the current user message using the prior conversation context. It uses card data instead of hallucinating, and it follows any provided expected outcome.
- "partial"   -> Response is on topic but incomplete, vague, or asks for information that should already be inferable from the conversation context. It may only partially satisfy the expected outcome.
- "incorrect" -> Response fails to answer the current turn, ignores conversation context, contains errors or hallucinations, or contradicts the expected outcome.

Respond with exactly one word: correct, partial, or incorrect.
"""

CONVERSATION_EVAL_TEMPLATE = """
You are evaluating the overall quality of a banking assistant conversation about debit card limits.

[Case description]
{case_description}

[Conversation objectives]
{case_objectives}

[Full conversation transcript]
{transcript}

[Evaluation criteria]
- "correct"   -> The overall conversation is coherent, efficient, and accurate. The assistant tracks state across turns, avoids unnecessary repetition, and achieves the user's goal using real card data.
- "partial"   -> The conversation is somewhat helpful but has avoidable friction, repeated questions, incomplete progress, or weak state tracking.
- "incorrect" -> The conversation fails overall because it loses state, gives wrong information, loops unnecessarily, or does not complete the user's task.

Respond with exactly one word: correct, partial, or incorrect.
"""

PAIRWISE_EVAL_TEMPLATE = """
You are comparing two banking assistant conversations for the same task.

[Case description]
{case_description}

[Conversation objectives]
{case_objectives}

[Tools agent conversation]
{tools_transcript}

[MCP agent conversation]
{mcp_transcript}

[Comparison criteria]
- "tools_better" -> The tools conversation is clearly better overall because it is more accurate, coherent, efficient, or context-aware.
- "mcp_better"   -> The MCP conversation is clearly better overall because it is more accurate, coherent, efficient, or context-aware.
- "tie"          -> Both conversations are effectively similar in quality overall.

Respond with exactly one word: tools_better, mcp_better, or tie.
"""


@dataclass(frozen=True)
class EvalTurn:
    user: str
    evaluation_focus: str = ""
    expected_outcome: str = ""


@dataclass(frozen=True)
class EvalCase:
    case_id: str
    description: str
    turns: tuple[EvalTurn, ...]


@dataclass(frozen=True)
class EvalCaseRun:
    turn_rows: tuple[dict[str, Any], ...]
    transcript: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run evals for limit agents.")
    parser.add_argument(
        "--test-cases",
        default=str(DEFAULT_TEST_CASES_PATH),
        help="Path to a JSON file describing eval cases.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_RESULTS_PATH),
        help="Path for the generated CSV results file.",
    )
    return parser.parse_args()


def _sanitize_identifier(value: str) -> str:
    sanitized = re.sub(r"[^a-zA-Z0-9_-]+", "-", value.strip()).strip("-")
    return sanitized or "case"


def _preview(text: str, limit: int = 80) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return f"{compact[:limit - 3]}..."


def _normalize_optional_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def load_test_cases(path: Path) -> list[EvalCase]:
    if not path.exists():
        raise FileNotFoundError(f"Test case file not found: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        raw_cases = payload.get("test_cases")
    elif isinstance(payload, list):
        raw_cases = payload
    else:
        raise ValueError("Test case JSON must be a list or an object with a 'test_cases' array")

    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("Test case JSON must define at least one test case")

    cases: list[EvalCase] = []
    for case_index, raw_case in enumerate(raw_cases, start=1):
        if not isinstance(raw_case, dict):
            raise ValueError(f"Test case #{case_index} must be an object")

        raw_turns = raw_case.get("turns")
        if not isinstance(raw_turns, list) or not raw_turns:
            raise ValueError(f"Test case #{case_index} must include a non-empty 'turns' array")

        case_id = _sanitize_identifier(_normalize_optional_text(raw_case.get("id")) or f"case-{case_index:03d}")
        description = _normalize_optional_text(raw_case.get("description") or raw_case.get("name"))

        turns: list[EvalTurn] = []
        for turn_index, raw_turn in enumerate(raw_turns, start=1):
            if isinstance(raw_turn, str):
                user_message = raw_turn.strip()
                if not user_message:
                    raise ValueError(f"Turn {turn_index} in case '{case_id}' cannot be empty")
                turns.append(EvalTurn(user=user_message))
                continue

            if not isinstance(raw_turn, dict):
                raise ValueError(f"Turn {turn_index} in case '{case_id}' must be a string or object")

            user_message = _normalize_optional_text(raw_turn.get("user"))
            if not user_message:
                raise ValueError(f"Turn {turn_index} in case '{case_id}' must include a non-empty 'user'")

            turns.append(
                EvalTurn(
                    user=user_message,
                    evaluation_focus=_normalize_optional_text(raw_turn.get("evaluation_focus")),
                    expected_outcome=_normalize_optional_text(raw_turn.get("expected_outcome")),
                )
            )

        cases.append(EvalCase(case_id=case_id, description=description, turns=tuple(turns)))

    return cases


def _format_history(transcript: list[tuple[str, str]]) -> str:
    if not transcript:
        return "Conversation start. There is no prior history."

    lines: list[str] = []
    for turn_index, (user_text, assistant_text) in enumerate(transcript, start=1):
        lines.append(f"Turn {turn_index} user: {user_text}")
        lines.append(f"Turn {turn_index} assistant: {assistant_text}")
    return "\n".join(lines)


def _format_case_objectives(case: EvalCase) -> str:
    lines: list[str] = []
    if case.description:
        lines.append(f"Case goal: {case.description}")

    for turn_index, turn in enumerate(case.turns, start=1):
        if turn.evaluation_focus:
            lines.append(f"Turn {turn_index} focus: {turn.evaluation_focus}")
        if turn.expected_outcome:
            lines.append(f"Turn {turn_index} expected outcome: {turn.expected_outcome}")

    return "\n".join(lines) if lines else "No explicit objectives provided. Judge overall helpfulness and correctness."


def build_tools_eval_agent() -> Agent:
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
async def build_mcp_eval_agent() -> AsyncIterator[Agent]:
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


async def run_eval_case(agent: Agent, case: EvalCase, agent_key: str, workflow_name: str) -> EvalCaseRun:
    session = SQLiteSession(
        session_id=f"eval-{RUN_ID}-{agent_key}-{_sanitize_identifier(case.case_id)}"
    )
    transcript: list[tuple[str, str]] = []
    turn_rows: list[dict[str, Any]] = []

    with trace(workflow_name=workflow_name):
        for turn_index, turn in enumerate(case.turns, start=1):
            history = _format_history(transcript)
            try:
                result = await Runner.run(agent, turn.user, session=session)
                output = str(result.final_output or "").strip()
            except Exception as exc:
                output = f"Request failed: {exc}"

            if not output:
                output = "I could not generate a response."

            turn_rows.append(
                {
                    "case_id": case.case_id,
                    "case_description": case.description,
                    "turn_index": turn_index,
                    "turn_count": len(case.turns),
                    "conversation_history": history,
                    "latest_user_input": turn.user,
                    "evaluation_focus": turn.evaluation_focus or case.description,
                    "expected_outcome": turn.expected_outcome,
                    "output": output,
                }
            )
            transcript.append((turn.user, output))

    return EvalCaseRun(
        turn_rows=tuple(turn_rows),
        transcript=_format_history(transcript),
    )


def _shared_turn_metadata(tools_turn: dict[str, Any], mcp_turn: dict[str, Any]) -> dict[str, Any]:
    shared_keys = [
        "case_id",
        "case_description",
        "turn_index",
        "turn_count",
        "latest_user_input",
        "evaluation_focus",
        "expected_outcome",
    ]

    for key in shared_keys:
        if tools_turn[key] != mcp_turn[key]:
            raise RuntimeError(
                f"Mismatched shared turn field '{key}' while evaluating case '{tools_turn['case_id']}'"
            )

    return {key: tools_turn[key] for key in shared_keys}


async def collect_results(test_cases: list[EvalCase]) -> tuple[pd.DataFrame, pd.DataFrame]:
    turn_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    tools_agent = build_tools_eval_agent()

    async with build_mcp_eval_agent() as mcp_agent:
        for case_index, case in enumerate(test_cases, start=1):
            label = case.description or case.turns[0].user
            print(f"[{case_index}/{len(test_cases)}] {case.case_id} ({len(case.turns)} turns) - {label}")

            tools_case_run = await run_eval_case(
                agent=tools_agent,
                case=case,
                agent_key="tools",
                workflow_name="eval:limit-agent-tools",
            )
            mcp_case_run = await run_eval_case(
                agent=mcp_agent,
                case=case,
                agent_key="mcp",
                workflow_name="eval:limit-agent-mcp",
            )

            if len(tools_case_run.turn_rows) != len(mcp_case_run.turn_rows):
                raise RuntimeError(f"Mismatched turn counts while evaluating case '{case.case_id}'")

            for tools_turn, mcp_turn in zip(tools_case_run.turn_rows, mcp_case_run.turn_rows):
                print(f"  tools t{tools_turn['turn_index']} -> {_preview(tools_turn['output'])}")
                print(f"  mcp   t{mcp_turn['turn_index']} -> {_preview(mcp_turn['output'])}")
                shared_metadata = _shared_turn_metadata(tools_turn, mcp_turn)
                turn_rows.append(
                    {
                        **shared_metadata,
                        "tools_conversation_history": tools_turn["conversation_history"],
                        "mcp_conversation_history": mcp_turn["conversation_history"],
                        "tools_output": tools_turn["output"],
                        "mcp_output": mcp_turn["output"],
                    }
                )

            case_rows.append(
                {
                    "case_id": case.case_id,
                    "case_description": case.description,
                    "turn_count": len(case.turns),
                    "case_objectives": _format_case_objectives(case),
                    "tools_transcript": tools_case_run.transcript,
                    "mcp_transcript": mcp_case_run.transcript,
                }
            )

    return pd.DataFrame(turn_rows), pd.DataFrame(case_rows)


def run_turn_evals(df: pd.DataFrame) -> pd.DataFrame:
    from phoenix.evals import OpenAIModel, llm_classify

    judge = OpenAIModel(model=JUDGE_MODEL, api_key=OPENAI_API_KEY)

    eval_columns = [
        "case_id",
        "turn_index",
        "turn_count",
        "latest_user_input",
        "evaluation_focus",
        "expected_outcome",
    ]

    tools_df = df[eval_columns + ["tools_conversation_history", "tools_output"]].rename(
        columns={
            "tools_conversation_history": "conversation_history",
            "tools_output": "output",
        }
    )
    mcp_df = df[eval_columns + ["mcp_conversation_history", "mcp_output"]].rename(
        columns={
            "mcp_conversation_history": "conversation_history",
            "mcp_output": "output",
        }
    )

    print("\nRunning judge evals...")
    tools_evals = llm_classify(
        data=tools_df,
        model=judge,
        template=EVAL_TEMPLATE,
        rails=["correct", "partial", "incorrect"],
        provide_explanation=True,
        concurrency=JUDGE_CONCURRENCY,
    )
    mcp_evals = llm_classify(
        data=mcp_df,
        model=judge,
        template=EVAL_TEMPLATE,
        rails=["correct", "partial", "incorrect"],
        provide_explanation=True,
        concurrency=JUDGE_CONCURRENCY,
    )

    df["tools_label"] = tools_evals["label"].values
    df["mcp_label"] = mcp_evals["label"].values
    df["tools_explanation"] = tools_evals.get("explanation", pd.Series([""] * len(df))).values
    df["mcp_explanation"] = mcp_evals.get("explanation", pd.Series([""] * len(df))).values
    return df


def run_case_evals(df: pd.DataFrame) -> pd.DataFrame:
    from phoenix.evals import OpenAIModel, llm_classify

    judge = OpenAIModel(model=JUDGE_MODEL, api_key=OPENAI_API_KEY)

    case_columns = ["case_id", "case_description", "turn_count", "case_objectives"]

    tools_df = df[case_columns + ["tools_transcript"]].rename(
        columns={"tools_transcript": "transcript"}
    )
    mcp_df = df[case_columns + ["mcp_transcript"]].rename(
        columns={"mcp_transcript": "transcript"}
    )

    print("\nRunning conversation-level evals...")
    tools_evals = llm_classify(
        data=tools_df,
        model=judge,
        template=CONVERSATION_EVAL_TEMPLATE,
        rails=["correct", "partial", "incorrect"],
        provide_explanation=True,
        concurrency=JUDGE_CONCURRENCY,
    )
    mcp_evals = llm_classify(
        data=mcp_df,
        model=judge,
        template=CONVERSATION_EVAL_TEMPLATE,
        rails=["correct", "partial", "incorrect"],
        provide_explanation=True,
        concurrency=JUDGE_CONCURRENCY,
    )
    pairwise_evals = llm_classify(
        data=df,
        model=judge,
        template=PAIRWISE_EVAL_TEMPLATE,
        rails=["tools_better", "mcp_better", "tie"],
        provide_explanation=True,
        concurrency=JUDGE_CONCURRENCY,
    )

    df["tools_conversation_label"] = tools_evals["label"].values
    df["mcp_conversation_label"] = mcp_evals["label"].values
    df["tools_conversation_explanation"] = tools_evals.get(
        "explanation", pd.Series([""] * len(df))
    ).values
    df["mcp_conversation_explanation"] = mcp_evals.get(
        "explanation", pd.Series([""] * len(df))
    ).values
    df["pairwise_label"] = pairwise_evals["label"].values
    df["pairwise_explanation"] = pairwise_evals.get(
        "explanation", pd.Series([""] * len(df))
    ).values
    return df


def print_summary(turn_df: pd.DataFrame, case_df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("TURN-LEVEL EVAL RESULTS")
    print("=" * 70)
    for _, row in turn_df.iterrows():
        print(
            f"\n[{row['case_id']}] turn {row['turn_index']}/{row['turn_count']}"
        )
        print(f"User: {row['latest_user_input']}")
        print(f"  tools -> [{row['tools_label']}]  {row['tools_explanation']}")
        print(f"  mcp   -> [{row['mcp_label']}]  {row['mcp_explanation']}")

    print("\n" + "=" * 70)
    print("TURN-LEVEL SUMMARY")
    print("=" * 70)
    for agent_name, column in [("limit-agent-tools", "tools_label"), ("limit-agent-mcp", "mcp_label")]:
        counts = turn_df[column].value_counts()
        correct = counts.get("correct", 0)
        total = len(turn_df)
        print(f"  {agent_name}: {correct}/{total} correct  |  {counts.to_dict()}")

    print("\n" + "=" * 70)
    print("CASE-LEVEL EVAL RESULTS")
    print("=" * 70)
    for _, row in case_df.iterrows():
        print(f"\n[{row['case_id']}] {row['case_description']}")
        print(
            f"  tools overall -> [{row['tools_conversation_label']}]  {row['tools_conversation_explanation']}"
        )
        print(
            f"  mcp   overall -> [{row['mcp_conversation_label']}]  {row['mcp_conversation_explanation']}"
        )
        print(f"  winner -> [{row['pairwise_label']}]  {row['pairwise_explanation']}")

    print("\n" + "=" * 70)
    print("CASE-LEVEL SUMMARY")
    print("=" * 70)
    for agent_name, column in [
        ("limit-agent-tools", "tools_conversation_label"),
        ("limit-agent-mcp", "mcp_conversation_label"),
    ]:
        counts = case_df[column].value_counts()
        correct = counts.get("correct", 0)
        total = len(case_df)
        print(f"  {agent_name}: {correct}/{total} correct  |  {counts.to_dict()}")

    pairwise_counts = case_df["pairwise_label"].value_counts()
    print(f"  pairwise winners: {pairwise_counts.to_dict()}")


async def main() -> None:
    args = parse_args()
    test_cases_path = Path(args.test_cases).expanduser().resolve()
    out_path = Path(args.output).expanduser().resolve()
    test_cases = load_test_cases(test_cases_path)

    print(f"Phoenix UI: {PHOENIX_ENDPOINT}")
    print(f"Loaded {len(test_cases)} test cases from {test_cases_path}")
    print("Each turn is evaluated independently with conversation context.\n")

    turn_df, case_df = await collect_results(test_cases)
    turn_df = run_turn_evals(turn_df)
    case_df = run_case_evals(case_df)
    print_summary(turn_df, case_df)

    turn_df.to_csv(out_path, index=False)
    case_out_path = out_path.with_name(f"{out_path.stem}_case{out_path.suffix}")
    case_df.to_csv(case_out_path, index=False)
    print(f"\nTurn-level results saved to {out_path}")
    print(f"Case-level results saved to {case_out_path}")


if __name__ == "__main__":
    asyncio.run(main())
