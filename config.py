import importlib.util
import os
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv
from openinference.instrumentation.openai_agents import OpenAIAgentsInstrumentor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TEST_CASES_PATH = Path(__file__).resolve().parent / "test_cases.json"
DEFAULT_RESULTS_PATH = Path(__file__).resolve().parent / "results.csv"
RUN_ID = uuid4().hex[:8]

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Missing required environment variable: OPENAI_API_KEY")
os.environ.setdefault("OPENAI_API_KEY", OPENAI_API_KEY)

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gpt-4.1")
LIMIT_API_BASE_URL = os.getenv("LIMIT_API_BASE_URL", "http://127.0.0.1:2010")
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://127.0.0.1:2009/mcp")
JUDGE_CONCURRENCY = int(os.getenv("JUDGE_CONCURRENCY", "8"))

# Phoenix tracing
PHOENIX_ENDPOINT = os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006")

tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(
    SimpleSpanProcessor(OTLPSpanExporter(endpoint=f"{PHOENIX_ENDPOINT}/v1/traces"))
)
OpenAIAgentsInstrumentor().instrument(tracer_provider=tracer_provider)


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
