"""Shared helpers for the Agentic AI Systems Engineering labs.

What lives here (and what does not):
- provider configuration: OpenRouter (default) and Hugging Face, plus any OpenAI-compatible endpoint;
- `tool_schema()`: a Python function → a tool definition for Chat Completions, Responses or Anthropic Messages;
- `chat()`: one cached, retried Chat Completions call that records tokens and cost in `LEDGER`;
- `run_agent()`: the reference agent loop (the solution of Lab 1.4) with budgets, hooks, permissions and a trace,
  so later labs can start from a working loop instead of depending on your Day 1 notebook;
- small lab utilities: `check()`, `lab_dir()`, `show()`.

The labs teach the raw APIs first; this module only removes repetition once you have written a thing yourself.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import re
import shutil
import sys
import time
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

for _stream in (sys.stdout, sys.stderr):  # Windows consoles default to a legacy code page: Arabic and emoji would crash
    try:
        if (getattr(_stream, "encoding", "") or "").lower() not in ("utf-8", "utf8"):
            _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

LABS = Path(__file__).resolve().parent
DATA = LABS / "data"
ARTIFACTS = LABS / "artifacts"  # per-lab outputs (git-ignored)
CACHE = LABS / "cache"  # cached model responses (git-ignored)

try:
    from dotenv import load_dotenv

    load_dotenv(LABS / ".env")
except ImportError:  # pragma: no cover
    pass

if "google.colab" in sys.modules:  # Colab secrets → env vars (Secrets panel, "Notebook access" on)
    try:
        from google.colab import userdata  # type: ignore

        for _k in ("OPENROUTER_API_KEY", "HF_TOKEN"):
            if not os.getenv(_k):
                try:
                    os.environ[_k] = userdata.get(_k)
                except Exception:  # noqa: BLE001 - secret not defined
                    pass
    except ImportError:
        pass

# ---------------------------------------------------------------- providers and models
PROVIDERS = {
    "openrouter": {"base_url": "https://openrouter.ai/api/v1", "key_env": "OPENROUTER_API_KEY",
                   "anthropic_base_url": "https://openrouter.ai/api"},
    "hf": {"base_url": "https://router.huggingface.co/v1", "key_env": "HF_TOKEN"},
    # any other engine: vLLM, SGLang, llama.cpp, Ollama, LM Studio, LiteLLM (they serve all three API shapes)
    "custom": {"base_url": os.getenv("LLM_BASE_URL", ""), "key_env": "LLM_API_KEY",
               "anthropic_base_url": os.getenv("LLM_ANTHROPIC_BASE_URL", "")},
}
PROVIDER = os.getenv("AGENT_PROVIDER", "openrouter")

# Free models with tool calling (checked October 2026 on openrouter.ai/models?max_price=0&supported_parameters=tools).
# Free lists change every few weeks: if a model disappears, pick another from that page and set AGENT_MODEL.
DEFAULT_MODELS = {
    "openrouter": "google/gemma-4-31b-it:free",
    "hf": "Qwen/Qwen3.6-35B-A3B",  # Inference Providers need credits (HF PRO includes a small monthly amount)
    "custom": os.getenv("LLM_MODEL", ""),
}
MODEL = os.getenv("AGENT_MODEL") or DEFAULT_MODELS.get(PROVIDER, "")
SMALL_MODEL = os.getenv("AGENT_SMALL_MODEL") or ("google/gemma-4-26b-a4b-it:free" if PROVIDER == "openrouter" else MODEL)
# OpenRouter tries these in order when the first model is rate-limited or down (the `models` routing parameter)
FALLBACKS = [m for m in os.getenv("AGENT_FALLBACKS", "google/gemma-4-26b-a4b-it:free,nvidia/nemotron-3-super-120b-a12b:free").split(",") if m]

QUICK = os.getenv("AGENT_QUICK", "0") == "1"  # smaller workloads (fewer eval tasks, fewer samples)
USE_CACHE = os.getenv("AGENT_CACHE", "1") == "1"  # replay identical requests from disk: saves free quota on re-runs


def pick(full, quick):
    return quick if QUICK else full


def api_key(provider: str | None = None) -> str:
    provider = provider or PROVIDER
    key = os.getenv(PROVIDERS[provider]["key_env"], "")
    if not key and provider != "custom":
        raise RuntimeError(f"Set {PROVIDERS[provider]['key_env']} in labs/.env (or Colab secrets). See labs/README.md.")
    return key or "none"


def headers(provider: str | None = None) -> dict:
    """OpenRouter app attribution headers (optional; they name the app on openrouter.ai)."""
    if (provider or PROVIDER) != "openrouter":
        return {}
    return {"HTTP-Referer": "https://sdaiaacademy.github.io/agentic-ai/", "X-OpenRouter-Title": "SDAIA Academy Agentic AI labs"}


def client(provider: str | None = None, **kw):
    """An OpenAI SDK client for Chat Completions and Responses, pointed at OpenRouter, HF or a custom endpoint."""
    from openai import OpenAI

    provider = provider or PROVIDER
    p = PROVIDERS[provider]
    return OpenAI(base_url=p["base_url"], api_key=api_key(provider), default_headers=headers(provider),
                  max_retries=kw.pop("max_retries", 4), timeout=kw.pop("timeout", 120), **kw)


def anthropic_client(provider: str | None = None, **kw):
    """An Anthropic SDK client for the Messages API shape, pointed at OpenRouter's Anthropic-compatible endpoint."""
    import anthropic

    provider = provider or PROVIDER
    base = PROVIDERS[provider].get("anthropic_base_url")
    if not base:
        raise RuntimeError(f"No Anthropic-compatible endpoint configured for provider {provider!r}.")
    return anthropic.Anthropic(base_url=base, auth_token=api_key(provider), api_key=None,
                               default_headers=headers(provider), max_retries=kw.pop("max_retries", 4), **kw)


# ---------------------------------------------------------------- cost ledger
@dataclass
class Ledger:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    cost: float = 0.0
    cache_hits: int = 0

    def add(self, usage: Any, cached: bool = False) -> None:
        self.calls += 1
        self.cache_hits += cached
        if usage is None:
            return
        u = usage if isinstance(usage, dict) else usage.model_dump()
        self.input_tokens += u.get("prompt_tokens") or u.get("input_tokens") or 0
        self.output_tokens += u.get("completion_tokens") or u.get("output_tokens") or 0
        details = u.get("prompt_tokens_details") or u.get("input_tokens_details") or {}
        self.cached_tokens += (details or {}).get("cached_tokens") or u.get("cache_read_input_tokens") or 0
        if not cached:
            self.cost += float(u.get("cost") or 0)

    def __str__(self) -> str:
        return (f"{self.calls} calls ({self.cache_hits} from cache) · {self.input_tokens:,} in / {self.output_tokens:,} out tokens"
                f" · {self.cached_tokens:,} cached · ${self.cost:.4f}")

    def reset(self) -> None:
        self.__init__()


LEDGER = Ledger()


# ---------------------------------------------------------------- response cache
def _cache_path(kind: str, payload: dict) -> Path:
    key = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()[:24]
    return CACHE / kind / f"{key}.json"


def cached_call(kind: str, payload: dict, call: Callable[[], Any], use_cache: bool | None = None):
    """Run `call()` once per identical payload; later identical requests come from disk (returns obj, from_cache)."""
    use_cache = USE_CACHE if use_cache is None else use_cache
    path = _cache_path(kind, payload)
    if use_cache and path.exists():
        return json.loads(path.read_text(encoding="utf-8")), True
    obj = call()
    data = obj.model_dump(mode="json") if hasattr(obj, "model_dump") else obj
    if use_cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return data, False


def rebuild(cls, data: dict):
    """Turn cached JSON back into an SDK object, leniently (gateways often omit optional fields)."""
    if cls.__module__.startswith("anthropic"):
        from anthropic._models import construct_type
    else:
        from openai._models import construct_type
    return construct_type(type_=cls, value=data)


def chat(messages: list[dict], model: str | None = None, tools: list | None = None, *, provider: str | None = None,
         fallbacks: bool = True, use_cache: bool | None = None, **kw):
    """One Chat Completions call → `ChatCompletion`. Cached, retried, recorded in LEDGER.

    `tools` may be Python functions (converted with `tool_schema`) or ready tool dicts.
    On OpenRouter, `fallbacks=True` adds the FALLBACKS model list so a rate-limited free model is replaced.
    """
    from openai.types.chat import ChatCompletion

    provider = provider or PROVIDER
    req = dict(model=model or MODEL, messages=messages, **kw)
    if tools:
        req["tools"] = [t if isinstance(t, dict) else tool_schema(t) for t in tools]
    extra = dict(req.pop("extra_body", {}) or {})
    if provider == "openrouter" and fallbacks and FALLBACKS and "models" not in extra:
        extra["models"] = [req["model"]] + [m for m in FALLBACKS if m != req["model"]]
    payload = dict(provider=provider, base=PROVIDERS[provider]["base_url"], **req, extra_body=extra)
    data, hit = cached_call("chat", payload, lambda: client(provider).chat.completions.create(**req, extra_body=extra or None),
                            use_cache)
    resp = rebuild(ChatCompletion, data)
    LEDGER.add(resp.usage, cached=hit)
    return resp


# ---------------------------------------------------------------- tools: Python function → schema
_DOC_ARG = re.compile(r"^\s{2,}(\w+)(?:\s*\([^)]*\))?:\s*(.+)$")


def _parse_doc(fn: Callable) -> tuple[str, dict[str, str]]:
    doc = inspect.getdoc(fn) or ""
    head, _, rest = doc.partition("\nArgs:")
    args, current = {}, None
    for line in rest.splitlines():
        m = re.match(r"^\s*(\w+)(?:\s*\([^)]*\))?:\s*(.*)$", line)
        if m and (line.startswith("    ") or line.startswith("  ")) and not line.startswith("        "):
            current = m.group(1)
            args[current] = m.group(2).strip()
        elif current and line.strip():
            args[current] += " " + line.strip()
    return head.strip(), args


def json_schema_for(fn: Callable) -> dict:
    """JSON Schema for a function's parameters, from its type hints (via Pydantic) and the docstring's Args section."""
    from pydantic import TypeAdapter

    _, arg_docs = _parse_doc(fn)
    hints = typing.get_type_hints(fn)
    props, required = {}, []
    for name, param in inspect.signature(fn).parameters.items():
        if name in ("self", "cls") or param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        ann = hints.get(name, str)
        schema = TypeAdapter(ann).json_schema()
        if "anyOf" in schema and any(s.get("type") == "null" for s in schema["anyOf"]):  # Optional[X] → X, not required
            non_null = [s for s in schema["anyOf"] if s.get("type") != "null"]
            schema = non_null[0] if len(non_null) == 1 else {"anyOf": non_null}
        schema.pop("title", None)
        if name in arg_docs:
            schema["description"] = arg_docs[name]
        if param.default is inspect.Parameter.empty:
            required.append(name)
        elif param.default is not None:
            schema["default"] = param.default
        props[name] = schema
    return {"type": "object", "properties": props, "required": required, "additionalProperties": False}


def tool_schema(fn: Callable, shape: str = "chat") -> dict:
    """A tool definition for one Python function.

    shape: "chat" (Chat Completions), "responses" (Responses API), "anthropic" (Messages API) or "mcp".
    """
    spec = getattr(fn, "__tool_schema__", None)  # tools that aren't Python functions (e.g. MCP tools) carry their schema
    if spec:
        name, description, params = spec["name"], spec["description"], spec["parameters"]
    else:
        description, _ = _parse_doc(fn)
        params = json_schema_for(fn)
        name = fn.__name__
    if shape == "chat":
        return {"type": "function", "function": {"name": name, "description": description, "parameters": params}}
    if shape == "responses":
        return {"type": "function", "name": name, "description": description, "parameters": params}
    if shape == "anthropic":
        return {"name": name, "description": description, "input_schema": params}
    if shape == "mcp":
        return {"name": name, "description": description, "inputSchema": params}
    raise ValueError(f"unknown shape {shape!r}")


def call_tool(tools: dict[str, Callable], name: str, arguments: str | dict) -> str:
    """Run one tool call and return what the model should see: the JSON result, or an error it can act on."""
    from madar import ToolError  # local import keeps agentkit usable without the case study

    if name not in tools:
        return json.dumps({"error": f"Unknown tool {name!r}. Available: {sorted(tools)}"})
    try:
        args = json.loads(arguments) if isinstance(arguments, str) else dict(arguments or {})
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"Arguments are not valid JSON ({e}). Send a JSON object."})
    try:
        result = tools[name](**args)
    except ToolError as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)
    except TypeError as e:  # wrong or missing arguments
        return json.dumps({"error": f"Bad arguments for {name}: {e}"})
    return result if isinstance(result, str) else json.dumps(result, ensure_ascii=False, default=str)


# ---------------------------------------------------------------- token estimates
TOKENIZER_ID = os.getenv("AGENT_TOKENIZER", "Qwen/Qwen3-0.6B")  # an open tokenizer (a few MB, no model weights)
_tokenizer: Any = None


def count_tokens(obj: Any) -> int:
    """Estimate the tokens of a string, or of any JSON-able object as it would be serialized into a request.

    Uses an open LLM tokenizer from the Hugging Face Hub (downloaded once); offline it falls back to chars / 3.5.
    Every model's tokenizer differs, so use this to compare parts of one request, and `usage` for exact counts.
    """
    global _tokenizer
    text = obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False, default=str)
    if _tokenizer is None:
        try:
            from tokenizers import Tokenizer

            _tokenizer = Tokenizer.from_pretrained(TOKENIZER_ID)
        except Exception:  # noqa: BLE001 - offline or blocked: estimate
            _tokenizer = False
    return len(_tokenizer.encode(text).ids) if _tokenizer else round(len(text) / 3.5)


# ---------------------------------------------------------------- MCP: use a server's tools from a sync loop
class MCPBridge:
    """Run an MCP client on a background event loop so the synchronous `run_agent` loop can call MCP tools.

    `server` is anything `mcp.Client` accepts: an `MCPServer` object (in-process), `StdioServerParameters`
    (launch a subprocess, e.g. `uv run madar_mcp.py`) or a Streamable HTTP URL.

        with MCPBridge(server) as mcp:
            mcp.list_tools()               # mcp.types.Tool objects
            mcp.call("compute_fee", {...})  # CallToolResult
            tools = mcp.agent_tools()       # callables for run_agent
    """

    def __init__(self, server: Any, **client_kw):
        self.server, self.client_kw = server, client_kw

    def __enter__(self) -> "MCPBridge":
        import asyncio
        import threading

        from concurrent.futures import FIRST_COMPLETED, Future, wait

        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self.thread.start()
        ready: Future = Future()
        # one long-lived task opens and closes the client (anyio scopes must exit in the task that entered them)
        self._main = asyncio.run_coroutine_threadsafe(self._session(ready), self.loop)
        wait([ready, self._main], timeout=120, return_when=FIRST_COMPLETED)
        if not ready.done():
            self._stop_loop()
            self._main.result(0) if self._main.done() else None  # re-raise the connection error, if any
            raise TimeoutError("MCP server did not start within 120 s")
        self.client = ready.result()
        return self

    async def _session(self, ready) -> None:
        import asyncio

        from mcp import Client

        self._closed = asyncio.Event()
        async with Client(self.server, **self.client_kw) as client:
            ready.set_result(client)
            await self._closed.wait()

    def run(self, coro, timeout: float = 120):
        import asyncio

        return asyncio.run_coroutine_threadsafe(coro, self.loop).result(timeout)

    def _stop_loop(self) -> None:
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(timeout=5)

    def __exit__(self, *exc) -> None:
        try:
            self.loop.call_soon_threadsafe(self._closed.set)
            self._main.result(30)
        finally:
            self._stop_loop()

    open = __enter__  # notebooks: `mcp = MCPBridge(server).open()` … `mcp.close()`

    def close(self) -> None:
        self.__exit__(None, None, None)

    def list_tools(self) -> list:
        return self.run(self.client.list_tools()).tools

    def call(self, name: str, arguments: dict) -> Any:
        return self.run(self.client.call_tool(name, arguments))

    def __getattr__(self, name: str):  # list_resources, read_resource, list_prompts, get_prompt, ... (sync)
        attr = getattr(self.client, name)
        return (lambda *a, **kw: self.run(attr(*a, **kw))) if callable(attr) else attr

    def agent_tools(self) -> list[Callable]:
        """Each MCP tool as a callable with `__tool_schema__`, usable in `run_agent` and `tool_schema`."""
        return [mcp_to_tool(t, self.call) for t in self.list_tools()]


def mcp_result_text(result: Any) -> str:
    """The text an agent should see for an MCP CallToolResult (errors are prefixed so the loop flags them)."""
    text = "\n".join(getattr(c, "text", "") or "" for c in result.content)
    return json.dumps({"error": text}, ensure_ascii=False) if result.is_error else text


def mcp_to_tool(tool: Any, call: Callable[[str, dict], Any]) -> Callable:
    """Wrap one MCP tool definition as a Python callable the agent loop can use."""

    def fn(**arguments):
        return mcp_result_text(call(tool.name, arguments))

    fn.__name__ = tool.name
    fn.__doc__ = tool.description or ""
    fn.__tool_schema__ = {"name": tool.name, "description": (tool.description or "").split("\nArgs:")[0].strip(),
                          "parameters": tool.input_schema}
    return fn


# ---------------------------------------------------------------- lab utilities
def check(cond: bool, msg: str) -> None:
    print(("✅ " if cond else "❌ ") + msg)


def lab_dir(name: str, reset: bool = False) -> Path:
    path = ARTIFACTS / name
    if reset and path.exists():
        def writable(func, p, _exc):  # Windows: git object files are read-only and refuse deletion
            os.chmod(p, 0o700)
            func(p)

        if sys.version_info >= (3, 12):
            shutil.rmtree(path, onexc=writable)
        else:
            shutil.rmtree(path, onerror=writable)
    path.mkdir(parents=True, exist_ok=True)
    return path


def show(obj: Any, width: int = 110) -> None:
    """Print a model response, message list or tool result compactly."""
    if hasattr(obj, "model_dump"):
        obj = obj.model_dump(exclude_none=True)
    text = json.dumps(obj, ensure_ascii=False, indent=1, default=str)
    for line in text.splitlines():
        print(line if len(line) <= width else line[: width - 1] + "…")


def text_of(resp: Any) -> str:
    """The assistant text of a ChatCompletion, Responses response or Anthropic message."""
    if hasattr(resp, "choices"):
        return resp.choices[0].message.content or ""
    if hasattr(resp, "output_text"):
        return resp.output_text
    if hasattr(resp, "content"):
        return "".join(getattr(b, "text", "") for b in resp.content)
    return str(resp)


def structured(prompt: str | list[dict], schema: type, *, system: str | None = None, model: str | None = None,
               repairs: int = 1, **kw):
    """One Chat Completions call whose answer must validate as the Pydantic model `schema` (Day 1.2's pattern).

    Sends `response_format` with the JSON Schema; models and gateways that ignore it still get the schema in the
    prompt. An invalid answer is sent back once with the validation error ("repair"). Returns a `schema` instance.
    """
    from pydantic import ValidationError

    messages = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else list(prompt)
    js = schema.model_json_schema()
    guide = f"Reply with one JSON object that matches this JSON Schema, and nothing else:\n{json.dumps(js, ensure_ascii=False)}"
    messages = [{"role": "system", "content": f"{system}\n\n{guide}" if system else guide}] + messages
    fmt = {"type": "json_schema", "json_schema": {"name": schema.__name__, "schema": js}}
    for attempt in range(repairs + 1):
        text = text_of(chat(messages, model=model, response_format=fmt, temperature=0, **kw)).strip()
        start, end = text.find("{"), text.rfind("}")
        try:
            return schema.model_validate_json(text[start: end + 1] if start >= 0 else text)
        except ValidationError as e:
            if attempt == repairs:
                raise
            messages += [{"role": "assistant", "content": text},
                         {"role": "user", "content": f"That JSON is invalid: {str(e)[:600]}\nReply with the corrected JSON only."}]


def serve(cmd: list[str], ready_url: str | None = None, timeout: float = 180):
    """Start a background server (e.g. `uv run servers/x.py --port N`) and wait until `ready_url` answers.

    Returns the Popen handle; stop it with `stop(proc)`, which ends the whole process tree (`uv run` starts a child
    Python process that a plain `terminate()` would leave running). Servers are also stopped when the kernel exits.
    """
    import atexit
    import subprocess

    import httpx

    kw = ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {"start_new_session": True})
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **kw)
    atexit.register(stop, proc)
    deadline = time.time() + timeout
    while ready_url and time.time() < deadline:  # the first `uv run` installs the script's dependencies
        if proc.poll() is not None:
            raise RuntimeError(f"Server exited with code {proc.returncode}: {' '.join(cmd)}")
        try:
            httpx.get(ready_url, timeout=2)
            return proc
        except httpx.HTTPError:
            time.sleep(1)
    if ready_url:
        stop(proc)
        raise TimeoutError(f"{ready_url} did not answer within {timeout} s")
    return proc


def stop(proc) -> None:
    """Stop a server started with `serve`, including its child processes."""
    import signal
    import subprocess

    if proc.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True)
    else:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


def timed(fn: Callable, *a, **kw):
    t = time.perf_counter()
    out = fn(*a, **kw)
    return out, time.perf_counter() - t


# ---------------------------------------------------------------- one loop, three API shapes
@dataclass
class Call:
    id: str
    name: str
    arguments: str  # JSON text


@dataclass
class Turn:
    text: str
    calls: list[Call]
    usage: Any = None
    raw: Any = None


class ChatShape:
    """Chat Completions: a list of role messages; tool calls on the assistant message; results as role="tool"."""

    name = "chat"

    def __init__(self, provider: str | None = None):
        self.provider = provider or PROVIDER

    def start(self, system: str, task: str, history: list | None = None) -> list:
        return [{"role": "system", "content": system}, *(history or []), {"role": "user", "content": task}]

    def step(self, state: list, tools: list[Callable], model: str, use_cache: bool | None = None, **kw) -> Turn:
        resp = chat(state, model=model, tools=tools or None, provider=self.provider, use_cache=use_cache, **kw)
        msg = resp.choices[0].message
        state.append(msg.model_dump(exclude_none=True))  # keep reasoning fields: some models need them back
        calls = [Call(c.id, c.function.name, c.function.arguments or "{}") for c in (msg.tool_calls or [])]
        return Turn(msg.content or "", calls, resp.usage, resp)

    def add_results(self, state: list, results: list[tuple[Call, str]]) -> None:
        state.extend({"role": "tool", "tool_call_id": c.id, "content": out} for c, out in results)


class ResponsesShape:
    """Responses API, stateless: replay every item (OpenRouter, Ollama and most engines reject server-side state)."""

    name = "responses"

    def __init__(self, provider: str | None = None):
        self.provider = provider or PROVIDER
        self.system = ""

    def start(self, system: str, task: str, history: list | None = None) -> list:
        self.system = system
        return [*(history or []), {"role": "user", "content": task}]

    def step(self, state: list, tools: list[Callable], model: str, use_cache: bool | None = None, **kw) -> Turn:
        from openai.types.responses import Response

        req = dict(model=model, instructions=self.system, input=state, store=False, **kw)
        if tools:
            req["tools"] = [tool_schema(t, "responses") for t in tools]
        payload = dict(provider=self.provider, **req)
        data, hit = cached_call("responses", payload, lambda: client(self.provider).responses.create(**req), use_cache)
        resp = rebuild(Response, data)
        LEDGER.add(resp.usage, cached=hit)
        state.extend(item.model_dump(exclude_none=True) for item in resp.output)
        calls = [Call(o.call_id, o.name, o.arguments or "{}") for o in resp.output if o.type == "function_call"]
        return Turn(resp.output_text or "", calls, resp.usage, resp)

    def add_results(self, state: list, results: list[tuple[Call, str]]) -> None:
        state.extend({"type": "function_call_output", "call_id": c.id, "output": out} for c, out in results)


class MessagesShape:
    """Anthropic Messages: content blocks; tool_use blocks in the assistant turn; tool_result blocks in one user turn."""

    name = "anthropic"

    def __init__(self, provider: str | None = None, max_tokens: int = 4096):
        self.provider = provider or PROVIDER
        self.system = ""
        self.max_tokens = max_tokens

    def start(self, system: str, task: str, history: list | None = None) -> list:
        self.system = system
        return [*(history or []), {"role": "user", "content": task}]

    def step(self, state: list, tools: list[Callable], model: str, use_cache: bool | None = None, **kw) -> Turn:
        from anthropic.types import Message

        req = dict(model=model, max_tokens=self.max_tokens, system=self.system, messages=state, **kw)
        if tools:
            req["tools"] = [tool_schema(t, "anthropic") for t in tools]
        payload = dict(provider=self.provider, shape="messages", **req)
        data, hit = cached_call("messages", payload, lambda: anthropic_client(self.provider).messages.create(**req), use_cache)
        msg = rebuild(Message, data)
        LEDGER.add(msg.usage, cached=hit)
        state.append({"role": "assistant", "content": [b.model_dump(exclude_none=True) for b in msg.content]})
        calls = [Call(b.id, b.name, json.dumps(b.input, ensure_ascii=False)) for b in msg.content if b.type == "tool_use"]
        text = "".join(b.text for b in msg.content if b.type == "text")
        return Turn(text, calls, msg.usage, msg)

    def add_results(self, state: list, results: list[tuple[Call, str]]) -> None:
        # every result of one turn goes back in ONE user message (splitting them discourages parallel calls)
        state.append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": c.id, "content": out, **({"is_error": True} if out.startswith('{"error"') else {})}
            for c, out in results]})


SHAPES = {"chat": ChatShape, "responses": ResponsesShape, "anthropic": MessagesShape}


@dataclass
class Event:
    kind: str  # user | model | call | result | final | error | guard
    text: str
    data: dict = field(default_factory=dict)


@dataclass
class Run:
    final: str
    state: list
    trace: list[Event]
    steps: int
    stopped: str  # done | max_steps | max_cost
    cost: float = 0.0
    tool_calls: list[tuple[str, dict]] = field(default_factory=list)


ICONS = {"user": "🧑", "model": "🤖", "call": "🔧", "result": "📦", "final": "✅", "error": "⚠️", "guard": "🛡️"}


def print_event(e: Event, width: int = 160) -> None:
    text = " ".join(e.text.split())
    print(f"{ICONS.get(e.kind, '·')} {e.kind:<6} {text[:width]}{'…' if len(text) > width else ''}")


def run_agent(task: str, tools: list[Callable], *, system: str = "You are a helpful assistant.", model: str | None = None,
              shape: str | Any = "chat", history: list | None = None, max_steps: int = 10, max_cost: float | None = None,
              permission: Callable[[str, dict], str] | None = None, approve: Callable[[str, dict], bool] | None = None,
              before_tool: Callable[[str, dict], str | None] | None = None,
              after_tool: Callable[[str, dict, str], str | None] | None = None,
              verbose: bool = True, use_cache: bool | None = None, **model_kw) -> Run:
    """The reference agent loop (Lab 1.4), with the harness hooks added on Day 3.

    - shape: "chat", "responses" or "anthropic" (or an adapter instance).
    - permission(name, args) → "allow" | "ask" | "deny"; "ask" calls approve(name, args) → bool (default: input()).
    - before_tool(name, args) may return a string to short-circuit the call (e.g. a blocked-by-policy message).
    - after_tool(name, args, result) may return a replacement result (e.g. a trimmed or redacted one).
    """
    adapter = SHAPES[shape]() if isinstance(shape, str) else shape
    model = model or MODEL
    by_name = {t.__name__: t for t in tools}
    state = adapter.start(system, task, history)
    trace: list[Event] = []

    def emit(e: Event) -> None:
        trace.append(e)
        if verbose:
            print_event(e)

    emit(Event("user", task))
    start_cost, calls_made = LEDGER.cost, []
    for step in range(1, max_steps + 1):
        turn = adapter.step(state, tools, model, use_cache=use_cache, **model_kw)
        if turn.text and turn.calls:
            emit(Event("model", turn.text))
        if not turn.calls:
            emit(Event("final", turn.text))
            return Run(turn.text, state, trace, step, "done", LEDGER.cost - start_cost, calls_made)
        results = []
        for c in turn.calls:
            try:
                args = json.loads(c.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            emit(Event("call", f"{c.name}({json.dumps(args, ensure_ascii=False)})", {"name": c.name, "args": args}))
            calls_made.append((c.name, args))
            decision = permission(c.name, args) if permission else "allow"
            if decision == "ask":
                ok = approve(c.name, args) if approve else input(f"Allow {c.name}({args})? [y/N] ").strip().lower() == "y"
                decision = "allow" if ok else "deny"
            if decision == "deny":
                out = json.dumps({"error": f"The harness denied {c.name}: not permitted here. Tell the person what you "
                                           "would have done, or hand over to an officer."})
                emit(Event("guard", f"denied {c.name}"))
            else:
                out = before_tool(c.name, args) if before_tool else None
                if out is not None:
                    emit(Event("guard", f"before_tool blocked {c.name}"))
                else:
                    out = call_tool(by_name, c.name, c.arguments or "{}")
                if after_tool:
                    out = after_tool(c.name, args, out) or out
            emit(Event("error" if out.startswith('{"error"') else "result", out, {"name": c.name}))
            results.append((c, out))
        adapter.add_results(state, results)
        if max_cost is not None and LEDGER.cost - start_cost > max_cost:
            emit(Event("guard", f"cost budget ${max_cost} exceeded"))
            return Run("", state, trace, step, "max_cost", LEDGER.cost - start_cost, calls_made)
    emit(Event("guard", f"stopped after {max_steps} steps"))
    return Run("", state, trace, max_steps, "max_steps", LEDGER.cost - start_cost, calls_made)
