from typing import Any

import pandas as pd
from agents import Agent, Runner, SQLiteSession, trace

from src.agents_module import build_mcp_agent, build_tools_agent
from src.config import RUN_ID
from src.models import (
    EvalCase,
    EvalCaseRun,
    format_case_objectives,
    format_history,
    preview,
    sanitize_identifier,
)


async def run_eval_case(agent: Agent, case: EvalCase, agent_key: str, workflow_name: str) -> EvalCaseRun:
    session = SQLiteSession(
        session_id=f"eval-{RUN_ID}-{agent_key}-{sanitize_identifier(case.case_id)}"
    )
    transcript: list[tuple[str, str]] = []
    turn_rows: list[dict[str, Any]] = []

    with trace(workflow_name=workflow_name):
        for turn_index, turn in enumerate(case.turns, start=1):
            history = format_history(transcript)
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
        transcript=format_history(transcript),
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
    tools_agent = build_tools_agent()

    async with build_mcp_agent() as mcp_agent:
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
                print(f"  tools t{tools_turn['turn_index']} -> {preview(tools_turn['output'])}")
                print(f"  mcp   t{mcp_turn['turn_index']} -> {preview(mcp_turn['output'])}")
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
                    "case_objectives": format_case_objectives(case),
                    "tools_transcript": tools_case_run.transcript,
                    "mcp_transcript": mcp_case_run.transcript,
                }
            )

    return pd.DataFrame(turn_rows), pd.DataFrame(case_rows)
