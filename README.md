# limit-eval

Evaluation harness that compares two debit card limit agents — `limit-agent-tools` (direct HTTP function tools) vs `limit-agent-mcp` (MCP protocol) — using **LLM-as-a-judge** via [Arize Phoenix](https://arize.com/docs/phoenix).

## How it works

1. Both agents run against the same set of test prompts
2. Each agent's response is judged by a separate LLM (`gpt-4.1`) using a structured eval template
3. Traces are sent to Phoenix for deep inspection
4. Results are printed to stdout and saved to `results.csv`

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
```

Open [http://localhost:6006](http://localhost:6006) to inspect traces in the Phoenix UI.

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | Required |
| `OPENAI_MODEL` | `gpt-4.1` | Model used by both agents |
| `JUDGE_MODEL` | `gpt-4.1` | Model used as LLM judge |
| `LIMIT_API_BASE_URL` | `http://127.0.0.1:2010` | limit-api base URL |
| `MCP_SERVER_URL` | `http://127.0.0.1:2009/mcp` | MCP server URL |
| `PHOENIX_COLLECTOR_ENDPOINT` | `http://localhost:6006` | Phoenix OTLP endpoint |

## Output

After a run, results are printed to stdout and saved to `results.csv` with columns:

- `input` — the test prompt
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
├── requirements.txt
├── .env.example
├── results.csv         # generated after first run
└── llm-as-a-judge.md   # detailed research & implementation notes
```
