import pandas as pd
from phoenix.evals import OpenAIModel, llm_classify

from src.config import JUDGE_CONCURRENCY, JUDGE_MODEL, OPENAI_API_KEY
from src.templates import CONVERSATION_EVAL_TEMPLATE, EVAL_TEMPLATE, PAIRWISE_EVAL_TEMPLATE


def run_turn_evals(df: pd.DataFrame) -> pd.DataFrame:
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
