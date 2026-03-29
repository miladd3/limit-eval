# limit-eval

Evaluation harness that compares two debit card limit agents — `limit-agent-tools` (direct HTTP function tools) vs `limit-agent-mcp` (MCP protocol) — using **LLM-as-a-judge** via [Arize Phoenix](https://arize.com/docs/phoenix).

## How it works

1. Both agents run against the same JSON-defined test cases
2. Each test case can contain one or more conversation turns
3. Each agent response is judged by a separate LLM (`gpt-4.1`) using a structured eval template
4. Traces are sent to Phoenix for deep inspection
5. Results are printed to stdout and saved to `results.csv`

## Prerequisites

| Service | Port | How to start |
|---|---|---|
| Phoenix | 6006 | `phoenix serve` (in this venv) |
| limit-api | 2010 | Started by `limit-agent-mcp/run.sh` |
| limit-mcp | 2009 | Started by `limit-agent-mcp/run.sh` |

> **Python 3.13 required.** `arize-phoenix` does not support Python 3.14+.

## Setup

```bash
# Create venv with Python 3.13
/opt/homebrew/bin/python3.13 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Fill in OPENAI_API_KEY in .env
```

## Running

```bash
# Terminal 1 — start Phoenix
source .venv/bin/activate
phoenix serve

# Terminal 2 — start limit-api + limit-mcp
cd ~/projects/limit-agent-mcp && ./run.sh

# Terminal 3 — run the eval
source .venv/bin/activate
python eval.py

# Or point at a different JSON dataset
python eval.py --test-cases ./test_cases.json --output ./results.csv
```

Open [http://localhost:6006](http://localhost:6006) to inspect traces in the Phoenix UI.

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | Required |
| `OPENAI_MODEL` | `gpt-4.1` | Model used by both agents |
| `JUDGE_MODEL` | `gpt-4.1` | Model used as LLM judge |
| `JUDGE_CONCURRENCY` | `8` | Parallelism for Phoenix `llm_classify` |
| `LIMIT_API_BASE_URL` | `http://127.0.0.1:2010` | limit-api base URL |
| `MCP_SERVER_URL` | `http://127.0.0.1:2009/mcp` | MCP server URL |
| `PHOENIX_COLLECTOR_ENDPOINT` | `http://localhost:6006` | Phoenix OTLP endpoint |

## Test case format

`eval.py` now loads cases from `test_cases.json` by default. The file can be either:

- a top-level array of cases
- or an object with a `test_cases` array

Each case must include `turns`, and each turn can be either a plain string or an object:

```json
{
	"id": "follow-up-atm-limit",
	"description": "Multi-turn follow-up that depends on prior context",
	"turns": [
		{
			"user": "Show my current card limits",
			"evaluation_focus": "The response should establish card and limit context."
		},
		{
			"user": "What is my ATM withdrawal limit?",
			"expected_outcome": "The assistant should answer using the already established context."
		}
	]
}
```

Notes:

- A shared agent session is reused across turns within each case.
- Each turn is judged independently, but the judge receives the prior conversation history.
- `evaluation_focus` and `expected_outcome` are optional helpers for stronger multi-turn grading.

## Output

After a run, results are printed to stdout and saved to `results.csv` with columns:

- `case_id` / `turn_index` — identifies the evaluated turn
- `conversation_history` — prior turns shown to the judge
- `latest_user_input` — the current user turn
- `tools_output` / `mcp_output` — agent responses
- `tools_label` / `mcp_label` — judge verdict: `correct`, `partial`, or `incorrect`
- `tools_explanation` / `mcp_explanation` — judge reasoning

## Agent comparison

| | limit-agent-tools | limit-agent-mcp |
|---|---|---|
| Tool dispatch | In-process function call | HTTP/SSE to MCP server |
| Dependencies | limit-api only | limit-api + limit-mcp |
| Expected latency | Lower (no MCP hop) | Higher (extra network round-trip) |
| Tool schema source | Defined in Python | Fetched from MCP server at runtime |

## Project layout

```
limit-eval/
├── eval.py             # main eval runner
├── test_cases.json     # JSON dataset, supports multi-turn conversations
├── requirements.txt
├── .env.example
├── results.csv         # generated after first run
└── llm-as-a-judge.md   # detailed research & implementation notes
```
