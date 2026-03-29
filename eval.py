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
from pathlib import Path

from config import DEFAULT_RESULTS_PATH, DEFAULT_TEST_CASES_PATH, PHOENIX_ENDPOINT
from judge import print_summary, run_case_evals, run_turn_evals
from models import load_test_cases
from runner import collect_results


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
