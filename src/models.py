import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


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


def sanitize_identifier(value: str) -> str:
    sanitized = re.sub(r"[^a-zA-Z0-9_-]+", "-", value.strip()).strip("-")
    return sanitized or "case"


def preview(text: str, limit: int = 80) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return f"{compact[:limit - 3]}..."


def _normalize_optional_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def format_history(transcript: list[tuple[str, str]]) -> str:
    if not transcript:
        return "Conversation start. There is no prior history."

    lines: list[str] = []
    for turn_index, (user_text, assistant_text) in enumerate(transcript, start=1):
        lines.append(f"Turn {turn_index} user: {user_text}")
        lines.append(f"Turn {turn_index} assistant: {assistant_text}")
    return "\n".join(lines)


def format_case_objectives(case: EvalCase) -> str:
    lines: list[str] = []
    if case.description:
        lines.append(f"Case goal: {case.description}")

    for turn_index, turn in enumerate(case.turns, start=1):
        if turn.evaluation_focus:
            lines.append(f"Turn {turn_index} focus: {turn.evaluation_focus}")
        if turn.expected_outcome:
            lines.append(f"Turn {turn_index} expected outcome: {turn.expected_outcome}")

    return "\n".join(lines) if lines else "No explicit objectives provided. Judge overall helpfulness and correctness."


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

        case_id = sanitize_identifier(_normalize_optional_text(raw_case.get("id")) or f"case-{case_index:03d}")
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
