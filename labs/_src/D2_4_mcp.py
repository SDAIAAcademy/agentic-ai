# %% [markdown]
# # Lab 2.4 · Madar as an MCP server
#
# Day 2 · Session 4 (Model Context Protocol) · 25 min. So far Rafeeq's tools are Python functions in the same
# process. The **Model Context Protocol** (MCP) puts tools, resources and prompts behind a standard protocol, so
# any MCP host (Claude Code, Codex, Cursor, VS Code, OpenClaw, Pi with an extension, your own loop) can use them.
# In this lab you:
# - read a complete MCP server written as a single-file **PEP 723** script (`servers/madar_mcp.py`);
# - connect to it in-process and over **stdio** (`uv run`), and list its tools, resources and prompts;
# - speak the protocol **by hand**: stateless JSON-RPC over Streamable HTTP (MCP 2026-07-28);
# - wire MCP tools and resources into your agent loop.
#
# **Tasks**
# - [ ] 1 · Read the server (3')
# - [ ] 2 · Connect and discover (6')
# - [ ] 3 · The protocol by hand (6')
# - [ ] 4 · MCP tools in your agent loop (10')
# - [ ] ✱ Stretch · Turn on the write tools, safely

# %%
import os, sys, subprocess
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    REPO_URL = "https://github.com/SDAIAAcademy/agentic-ai.git"
    if not Path("/content/agentic-ai").exists():
        subprocess.run(["git", "clone", "-q", REPO_URL, "/content/agentic-ai"], check=True)
    os.chdir("/content/agentic-ai/labs")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements-colab.txt"], check=True)
else:
    while not Path("agentkit.py").exists() and Path.cwd() != Path.cwd().parent:
        os.chdir("..")  # solutions/ notebooks run from labs/
sys.path.insert(0, os.getcwd())

import agentkit as K
print("provider:", K.PROVIDER, "| model:", K.MODEL, "| cache:", K.USE_CACHE)

# %%
import json
import shutil
import time

import httpx
import pandas as pd
from mcp import StdioServerParameters

sys.path.insert(0, str(K.LABS / "servers"))
import madar
import madar_mcp

SERVER_FILE = K.LABS / "servers" / "madar_mcp.py"

# %% [markdown]
# ## 1 · Read the server (3')
#
# The whole server is one file. The `# /// script` block at the top is **PEP 723** inline metadata: `uv run` reads
# it, builds an isolated environment with `mcp>=2.3`, and runs the file. Nothing to install, nothing to activate.
#
# Read it below and find: how a Madar method becomes an MCP **tool**, how a policy becomes a **resource** with a URI
# template, and what the **prompt** is for. Why does `anticipated()` exist?

# %%
print(SERVER_FILE.read_text(encoding="utf-8")[:2600])

# %% [markdown]
# ## 2 · Connect and discover (6')
#
# `K.MCPBridge` wraps the official client (`mcp.Client`) so the rest of the notebook can stay synchronous. It accepts
# the same targets as `mcp.Client`: an `MCPServer` object (in-process, fast, good for tests), stdio parameters (a
# subprocess, how desktop hosts launch local servers) or a URL (Streamable HTTP, for remote servers).
#
# ✏️ **Exercise** · Connect in-process to `madar_mcp.build(...)` and build `catalog`: a DataFrame with one row per
# tool (`kind="tool"`, `name`, the first line of its `description`), per resource (`kind="resource"`, `uri`), per
# resource template (`uri_template`) and per prompt.

# %%
mcp = K.MCPBridge(madar_mcp.build(K.lab_dir("D2_4/server", reset=True))).open()
# BEGIN SOLUTION
rows = [{"kind": "tool", "name": t.name, "about": (t.description or "").splitlines()[0]} for t in mcp.list_tools()]
rows += [{"kind": "resource", "name": str(r.uri), "about": r.description or ""} for r in mcp.list_resources().resources]
rows += [{"kind": "template", "name": t.uri_template, "about": t.description or ""}
         for t in mcp.list_resource_templates().resource_templates]
rows += [{"kind": "prompt", "name": p.name, "about": p.description or ""} for p in mcp.list_prompts().prompts]
catalog = pd.DataFrame(rows)
# END SOLUTION
K.check(set(catalog.kind) == {"tool", "resource", "template", "prompt"}, "found tools, resources, templates and prompts")
catalog

# %% [markdown]
# Call a tool, read a resource, render a prompt. Note the shape of each result.

# %%
r = mcp.call("compute_fee", {"permit_type": "SGN", "sign_m2": 3, "illuminated": True})
print("tool   →", r.is_error, r.content[0].text.replace("\n", " "))
r = mcp.call("get_permit", {"permit_id": "REN-10482", "national_id_last4": "0000", "mobile": "0500000000"})
print("error  →", r.is_error, r.content[0].text[:110])
print("read   →", mcp.read_resource("policy://POL-09").contents[0].text[:90].replace("\n", " "))
p = mcp.get_prompt("triage_email", {"subject": "Charged twice", "body": "Please refund SGN-40388."})
print("prompt →", p.messages[0].role, p.messages[0].content.text[:90])

# %% [markdown]
# Now the same server as a **subprocess over stdio**, exactly how Claude Code or Cursor would launch it from a config
# entry like `{"command": "uv", "args": ["run", "servers/madar_mcp.py"]}`. The first run builds the script's
# environment, so it takes a few seconds.

# %%
uv = shutil.which("uv") or shutil.which("uv", path=str(Path(sys.executable).parent))
t0 = time.perf_counter()
with K.MCPBridge(StdioServerParameters(command=uv, args=["run", "--quiet", str(SERVER_FILE)])) as remote:
    names = [t.name for t in remote.list_tools()]
    fee = remote.call("compute_fee", {"permit_type": "BLD", "area_m2": 420}).content[0].text
print(f"stdio: {time.perf_counter() - t0:.1f}s · tools {names} · {json.loads(fee)['fee_sar']} SAR")
K.check(json.loads(fee)["fee_sar"] == 840, "the stdio server answers")

# %% [markdown]
# ## 3 · The protocol by hand (6')
#
# Under the SDK, MCP is **JSON-RPC 2.0**. Since the 2026-07-28 revision it is **stateless**: no `initialize`
# handshake and no session. Every request carries its protocol version and client capabilities in `_meta`, and over
# Streamable HTTP it repeats the method in an `Mcp-Method` header (so gateways can route without parsing bodies).
# `server/discover` replaces the old handshake for clients that want to know what the server offers.
#
# Start the server over HTTP in the background:

# %%
PORT = 8791
URL = f"http://127.0.0.1:{PORT}/mcp"
http_server = K.serve([uv, "run", "--quiet", str(SERVER_FILE), "--http", str(PORT)], ready_url=URL)
print("server pid", http_server.pid, "at", URL)

# %% [markdown]
# ✏️ **Exercise** · Write `rpc(method, params, id=1)`: POST a JSON-RPC request to `URL` with the headers
# `Content-Type: application/json`, `Accept: application/json, text/event-stream`, `MCP-Protocol-Version:
# 2026-07-28` and `Mcp-Method: <method>` (plus `Mcp-Name: <params["name"]>` for `tools/call`), and with
# `params["_meta"] = META`. Return the decoded JSON. Then call `server/discover`, `tools/list` and `tools/call`.

# %%
META = {"io.modelcontextprotocol/protocolVersion": "2026-07-28", "io.modelcontextprotocol/clientCapabilities": {}}


def rpc(method: str, params: dict | None = None, id: int = 1) -> dict:
    # BEGIN SOLUTION
    params = {**(params or {}), "_meta": META}
    headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream",
               "MCP-Protocol-Version": "2026-07-28", "Mcp-Method": method}
    if method == "tools/call":
        headers["Mcp-Name"] = params["name"]
    resp = httpx.post(URL, headers=headers, json={"jsonrpc": "2.0", "id": id, "method": method, "params": params}, timeout=30)
    return resp.json()
    # END SOLUTION


disc = rpc("server/discover")
print(json.dumps(disc, indent=1)[:500], "…")
listed = rpc("tools/list", id=2)
print([t["name"] for t in listed["result"]["tools"]])
called = rpc("tools/call", {"name": "compute_fee", "arguments": {"permit_type": "EVT", "event_days": 3}}, id=3)
print(called["result"]["content"][0]["text"])
K.check("result" in called and not called["result"]["isError"], "a tool call by hand")

# %%
K.stop(http_server)

# %% [markdown]
# ## 4 · MCP tools in your agent loop (10')
#
# A host turns each MCP tool into a tool definition for its model and forwards the model's calls to the server.
# Resources are different: the **host** decides when to read them (attach them to context, or expose a reading tool).
#
# ✏️ **Exercise**
# 1. Write `to_agent_tool(tool)`: a Python callable for `K.run_agent` whose `__name__` is the tool's name and whose
#    `__tool_schema__` is `{"name", "description", "parameters": tool.input_schema}`; calling it with keyword
#    arguments forwards to `mcp.call` and returns `K.mcp_result_text(result)`.
# 2. Write `read_policy(policy_id)`, a tool that reads the resource `policy://{policy_id}`.
# 3. Run Noura's question with the MCP tools plus `read_policy`, and check what the trace shows.

# %%
# BEGIN SOLUTION
def to_agent_tool(tool):
    def fn(**arguments):
        return K.mcp_result_text(mcp.call(tool.name, arguments))

    fn.__name__ = tool.name
    fn.__tool_schema__ = {"name": tool.name, "description": tool.description or "", "parameters": tool.input_schema}
    return fn


def read_policy(policy_id: str) -> str:
    """Read one Madar policy in full (resource policy://POL-xx).

    Args:
        policy_id: e.g. "POL-05".
    """
    try:
        return mcp.read_resource(f"policy://{policy_id}").contents[0].text
    except Exception as e:  # noqa: BLE001 - a missing resource is a recoverable error for the model
        return json.dumps({"error": str(e)})
# END SOLUTION


MCP_TOOLS = [to_agent_tool(t) for t in mcp.list_tools()] + [read_policy]
print([K.tool_schema(t)["function"]["name"] for t in MCP_TOOLS])
NOURA = ("My food truck permit FTR-30311 is approved. Which health-inspection slots are free on Tuesday 3 or "
         "Wednesday 4 November in the morning? ID ends 6654, mobile 0532236654. What happens if I miss it?")
r = K.run_agent(NOURA, MCP_TOOLS, system=madar.SYSTEM_PROMPT, max_steps=8)
K.check(any(n == "list_slots" for n, _ in r.tool_calls), "the agent found slots through MCP")
K.check("09:00" in r.final or "9:00" in r.final or "9 am" in r.final.lower(), "it offers the free slot on 4 Nov at 09:00")

# %% [markdown]
# 🔍 The model can't tell an MCP tool from a local one: same schema, same loop. What changed is **where the code
# runs** and **who wrote it**. Installing someone else's MCP server means running their code with your credentials,
# and putting their tool descriptions into your model's context. Day 4 comes back to that supply chain.

# %% [markdown]
# ## ✱ Stretch · Turn on the write tools, safely
#
# Start a second in-process server with `madar_mcp.build(..., write=True)`. MCP tools can carry **annotations**
# (`readOnlyHint`, `destructiveHint`) that hosts use to ask for approval. Wrap the write tools so the agent loop asks
# before running them: pass `permission=lambda name, args: "ask" if name in m.WRITE_TOOLS else "allow"` and an
# `approve` function that prints the call and returns `True` only for `book_inspection`. Book Noura's slot.

# %%
# BEGIN SOLUTION
m = madar.Madar(K.lab_dir("D2_4/client_view"))
with K.MCPBridge(madar_mcp.build(K.lab_dir("D2_4/server_rw", reset=True), write=True)) as rw:
    tools = [K.mcp_to_tool(t, rw.call) for t in rw.list_tools()]
    approvals = []

    def approve(name: str, args: dict) -> bool:
        approvals.append(name)
        print(f"  approval requested: {name}({args}) →", name == "book_inspection")
        return name == "book_inspection"

    r = K.run_agent(NOURA + " Please book 4 November at 09:00.", tools, system=madar.SYSTEM_PROMPT, max_steps=8,
                    permission=lambda name, args: "ask" if name in m.WRITE_TOOLS else "allow", approve=approve)
print("approvals:", approvals)
# END SOLUTION

# %%
mcp.close()

# %% [markdown]
# ## Checkpoint
#
# 1. Name the three MCP primitives and who decides when each one is used (model, host or user).
# 2. What did the 2026-07-28 revision remove from the protocol, and why does that help servers scale?
# 3. Why does the server turn Madar's `ToolError` into MCP's `ToolError` but hide every other exception?
# 4. You find a popular MCP server for "municipal permits" on a public registry. What do you check before connecting
#    it to Rafeeq?
