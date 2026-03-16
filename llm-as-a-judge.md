# LLM-as-a-Judge: Research & Implementation Notes

---

## Agent Handoff Context

> Read this section first if you are picking up this work mid-way.

### What this project does

Compares two agents (`limit-agent-tools` vs `limit-agent-mcp`) that do the same job (manage debit card limits) via different mechanisms, using LLM-as-a-judge via Arize Phoenix to evaluate output quality.

### Full project layout

```
~/projects/
├── limit-api/              FastAPI REST backend  — port 2010
├── limit-mcp/              MCP server (wraps limit-api)  — port 2009
├── limit-agent-tools/      OpenAI agent using direct HTTP function tools
├── limit-agent-mcp/        OpenAI agent using MCP protocol
├── limit-eval/             ← YOU ARE HERE — eval harness + Phoenix
│   ├── requirements.txt    Python deps (requires Python 3.13, NOT 3.14)
│   ├── .env                API keys and endpoints (gitignored)
│   ├── .env.example        Template
│   ├── eval.py             Main eval runner (run directly)
│   └── llm-as-a-judge.md   This file
└── limit.code-workspace    VS Code workspace for all projects
```

### Required startup order

```
1. Phoenix    → cd limit-eval && phoenix serve
2. Services   → cd limit-agent-mcp && ./run.sh   (starts limit-api + limit-mcp + agent)
3. Eval       → cd limit-eval && source .venv/bin/activate && python eval.py
```

Each agent's `run.sh` is its own entry point. `limit-eval` has no `run.sh` — just run `eval.py` directly.

### Environment

- macOS, Python 3.14 is system default but **3.13 is required here** (`arize-phoenix` max is <3.14)
- Each project has its own `.venv` — do not mix them
- `limit-eval/.venv` was created with `/opt/homebrew/bin/python3.13`
- All agent projects already have `.env` files with `OPENAI_API_KEY` filled in
- `OPENAI_MODEL=gpt-4.1` across all agents
- Ports: limit-api=2010, limit-mcp=2009, Phoenix=6006

### Current state

- [x] All 4 repos cloned and venvs set up
- [x] All `.env` files filled with real `OPENAI_API_KEY`
- [x] Both agents tested and working via their own `run.sh` scripts
- [x] `limit-eval/` fully set up — venv, deps, .env, eval.py
- [ ] First eval run not yet executed — run `python eval.py`
- [ ] `results.csv` not yet generated

### Key decisions made

- Used explicit OTel setup (not `phoenix.otel.register()`) — more portable, no dependency on `arize-phoenix` in agent code
- `eval.py` contains its own minimal agent setup rather than importing from agent dirs — avoids cross-project coupling
- `results.csv` written after each run for offline inspection
- Judge model = `gpt-4.1`, same as agent model (can be overridden via `JUDGE_MODEL` env var)

### How to run

```bash
# Terminal 1 — start Phoenix
cd ~/projects/limit-eval
source .venv/bin/activate
phoenix serve

# Terminal 2 — start limit-api + limit-mcp
cd ~/projects/limit-agent-mcp && ./run.sh

# Terminal 3 — run the eval
cd ~/projects/limit-eval
source .venv/bin/activate
python eval.py

# Open http://localhost:6006 to inspect traces
```

---

## What is LLM-as-a-Judge?

An evaluation approach that uses language models to assess the quality of other LLM outputs, replacing traditional metrics (ROUGE, BLEU) with human-like evaluation at scale.

**Core problem it solves:** Rule-based assessment and similarity metrics fail to handle nuanced LLM outputs. LLM-as-a-judge scales quality evaluation without manual annotation.

**The 4-step process:**
1. **Define Criteria** — what to evaluate (correctness, faithfulness, toxicity, relevance, etc.)
2. **Create Evaluation Prompt** — a template specifying which variables from the request/response are needed
3. **Choose Evaluator LLM** — the judge model (typically a stronger model than the one being evaluated)
4. **Execute & Analyze** — run evals across a dataset and review results

---

## Platform: Arize Phoenix

**Docs:** https://arize.com/docs/phoenix/evaluation/concepts-evals/llm-as-a-judge

Open source, fully self-hostable. No paid plan needed.

### Running Phoenix locally

`arize-phoenix` is installed in the `limit-eval` venv. No Docker needed.

```bash
cd ~/projects/limit-eval
source .venv/bin/activate
phoenix serve
# UI at http://localhost:6006
```

**Client-side env var:**
```
PHOENIX_COLLECTOR_ENDPOINT=http://localhost:6006
```

---

## Instrumentation

### Package

```bash
pip install openinference-instrumentation-openai-agents opentelemetry-sdk opentelemetry-exporter-otlp
```

- Package: `openinference-instrumentation-openai-agents` (real, v1.4.0+, Apache 2.0)
- Instruments the `openai-agents` SDK transparently — no changes needed to agent code
- Captures: agent runs, LLM calls, tool calls, MCP calls — all as OTel spans
- Works with `Runner.run()` and `Runner.run_sync()`

### Setup (confirmed working pattern)

```python
from openinference.instrumentation.openai_agents import OpenAIAgentsInstrumentor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(
    SimpleSpanProcessor(OTLPSpanExporter(endpoint="http://localhost:6006/v1/traces"))
)
OpenAIAgentsInstrumentor().instrument(tracer_provider=tracer_provider)
```

> Note: `phoenix.otel.register()` is an alternative shortcut but requires the `arize-phoenix` package at runtime. The explicit OTel setup above is more portable.

---

## How Tool Calling and MCP Calling Work (for tracing purposes)

### Function Tool Calling (`limit-agent-tools`)

```
Runner.run(agent, prompt)
    └── LLM call  [span: LLM]
          └── decides to call tool
    └── function_tool() executed locally  [span: Tool]
          └── httpx → limit-api HTTP request
    └── LLM call with tool result  [span: LLM]
          └── produces final answer
```

- Tools are plain Python async functions decorated with `@function_tool`
- The tool call and result are **in-process** — no network hop for the tool dispatch itself (only the API call inside the tool)
- Phoenix captures each tool invocation as a child span under the agent run
- You can see: tool name, input args, return value, duration

### MCP Tool Calling (`limit-agent-mcp`)

```
Runner.run(agent, prompt)
    └── MCPServerStreamableHttp.list_tools()  [span: MCP list_tools]
          └── HTTP GET  http://localhost:2009/mcp  (SSE)
    └── LLM call with tool schemas injected  [span: LLM]
          └── decides to call MCP tool
    └── MCPServerStreamableHttp.call_tool()  [span: MCP call_tool]
          └── HTTP POST http://localhost:2009/mcp  (SSE)
                └── limit-mcp → limit-api HTTP request
    └── LLM call with tool result  [span: LLM]
          └── produces final answer
```

- Tools are **remote** — the agent sends JSON-RPC over HTTP/SSE to the MCP server
- There is an extra network hop: agent → MCP server → API
- `cache_tools_list=True` means `list_tools()` is only called once per session
- Phoenix captures the MCP call as a child span, including tool name, args, and response

### Key difference in traces

| | limit-agent-tools | limit-agent-mcp |
|---|---|---|
| Tool dispatch | In-process function call | HTTP/SSE to MCP server |
| Span types | `LLM`, `Tool` | `LLM`, `MCP list_tools`, `MCP call_tool` |
| Extra latency | None for dispatch | Network round-trip to MCP server |
| Tool schema source | Defined in Python code | Fetched from MCP server at runtime |
| Visibility in Phoenix | Tool name + args + result | Same + MCP server URL + SSE session |

---

## Evaluator Types

### LLM-Based Evaluators
- Use a judge model to assess subjective criteria
- Return: label, score, explanation
- Used in `eval.py` via `phoenix.evals.llm_classify()`

### Code Evaluators
- Deterministic logic (exact match, regex, Levenshtein)
- Fast, no LLM calls

### Built-in Templates
- `correctness` — accuracy against a reference
- `relevance` — output pertinence to input
- `faithfulness` — output alignment with provided context

### Third-party Integrations
- [Ragas](https://docs.ragas.io/) — RAG-specific evaluation
- [Deepeval](https://docs.deepeval.com/) — general LLM evaluation

---

## Comparing Two Agents (limit-agent-tools vs limit-agent-mcp)

| Agent | Mechanism | Dependencies |
|---|---|---|
| `limit-agent-tools` | Calls `limit-api` directly via HTTP function tools | `limit-api` on :2010 |
| `limit-agent-mcp` | Calls `limit-mcp` via MCP/SSE protocol | `limit-api` + `limit-mcp` on :2010/:2009 |

### What to Measure

| Dimension | How |
|---|---|
| **Correctness** | LLM judge — did it return the right limits? |
| **Helpfulness** | LLM judge — was the response clear and complete? |
| **Tool use accuracy** | LLM judge — did it call the right tool? |
| **Latency** | Span duration in Phoenix (not judge) |
| **LLM call count** | Number of LLM spans per trace in Phoenix |

### Expected hypothesis
- `limit-agent-tools` — lower latency (no MCP network hop)
- `limit-agent-mcp` — tool schema flexibility (no redeployment to add tools)
- Judge determines if output quality actually differs

### Full Workflow

```
docker compose up -d          # start Phoenix
limit-api + limit-mcp running
    ↓
python eval.py
    ↓
Both agents run against TEST_CASES → traces sent to Phoenix
    ↓
llm_classify() judges each output
    ↓
Results printed + saved to results.csv
    ↓
Phoenix UI (http://localhost:6006) for deep trace inspection
```

---

## Project Structure

```
limit-eval/
├── docker-compose.yml       # Phoenix self-hosted
├── requirements.txt
├── .env.example
├── eval.py                  # main eval runner
├── results.csv              # output (generated after first run)
└── llm-as-a-judge.md        # this file
```

---

## TODO / Next Steps

- [x] Create `limit-eval/` project folder
- [x] Docker Compose for Phoenix
- [x] Confirmed `openinference-instrumentation-openai-agents` works with `openai-agents` SDK
- [x] Documented how tool calling vs MCP calling appear in traces
- [x] Setup venv with Python 3.13 (3.14 not supported by arize-phoenix)
- [x] Installed all deps: `arize-phoenix[evals]`, `openinference-instrumentation-openai-agents`, etc.
- [x] `.env` filled with real API key
- [x] `run.sh` starts Phoenix + limit-api + limit-mcp + eval in one command
- [ ] Run first eval: `./run.sh`
- [ ] Inspect traces in Phoenix UI at http://localhost:6006
- [ ] Log baseline results and compare
