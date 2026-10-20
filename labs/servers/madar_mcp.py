# /// script
# requires-python = ">=3.11"
# dependencies = ["mcp>=2.3"]
# ///
"""Madar Municipal Services as an MCP server (Lab 2.4).

Run it with uv; the dependencies above are installed into an isolated environment on first run:

    uv run servers/madar_mcp.py                    # stdio: an MCP host launches this as a subprocess
    uv run servers/madar_mcp.py --http 8765        # Streamable HTTP on http://127.0.0.1:8765/mcp
    uv run servers/madar_mcp.py --write            # also expose the write tools (off by default)

It exposes:
- tools: Madar's read tools (and, with --write, the write tools), straight from madar.py;
- resources: the policy index `madar://policies` and each policy as `policy://POL-02`;
- a prompt: `triage_email`, the triage instructions from Lab 1.2.
"""

from __future__ import annotations

import argparse
import functools
import inspect
import sys
from pathlib import Path

LABS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LABS))  # madar.py lives in labs/ and only uses the standard library

import madar  # noqa: E402
from mcp.server.mcpserver import MCPServer  # noqa: E402
from mcp.server.mcpserver.exceptions import ResourceNotFoundError, ToolError  # noqa: E402


def anticipated(fn):
    """Madar's ToolError → MCP's ToolError, so the model reads the message (other exceptions stay hidden)."""

    @functools.wraps(fn)  # keeps the signature, so the input schema comes from fn's type hints
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except madar.ToolError as e:
            raise ToolError(str(e)) from e

    return wrapper


def build(workdir: str | Path | None = None, write: bool = False) -> MCPServer:
    """Create the server. `workdir` holds this server's private copy of the Madar database."""
    m = madar.Madar(workdir or LABS / "artifacts" / "mcp_server")
    server = MCPServer(
        "madar",
        title="Madar Municipal Services",
        instructions="Tools and policies of Madar Municipal Services (a fictional municipality). "
                     "Verify identity with get_permit before disclosing anything about an application.",
        version="1.0.0",
        log_level="WARNING",
    )

    names = m.READ_TOOLS + (m.WRITE_TOOLS if write else ())
    for name in names:
        fn = getattr(m, name)
        # the model should see the summary and argument docs, not Python's raw docstring layout
        server.tool(name=name, description=inspect.getdoc(fn))(anticipated(fn))

    @server.resource("madar://policies", mime_type="text/markdown")
    def policy_index() -> str:
        """The list of Madar policies with their IDs and titles."""
        return "\n".join(f"- {p['id']}: {p['title']} ({p['title_ar']})" for p in madar.load_policies())

    @server.resource("policy://{policy_id}", mime_type="text/markdown")
    def policy(policy_id: str) -> str:
        """One Madar policy document (English with an Arabic summary), e.g. policy://POL-02."""
        for p in madar.load_policies():
            if p["id"].lower() == policy_id.lower():
                return p["text"]
        raise ResourceNotFoundError(f"No policy {policy_id!r}. Read madar://policies for the list.")

    @server.prompt()
    def triage_email(subject: str, body: str) -> str:
        """Classify one email sent to Madar (label, permit ID, language, needs_human)."""
        return (
            "You triage messages sent to Madar Municipal Services. Reply with JSON only: "
            '{"label": one of status|escalation|inspection|appeal|needs_info|refund|fees|rules|other, '
            '"permit_id": string or null, "lang": "ar" or "en", "needs_human": boolean}.\n\n'
            f"Subject: {subject}\n\n{body}"
        )

    return server


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--http", type=int, metavar="PORT", help="serve Streamable HTTP on this port instead of stdio")
    ap.add_argument("--write", action="store_true", help="also expose write tools (bookings, refunds, emails)")
    ap.add_argument("--workdir", help="folder for this server's copy of the database")
    a = ap.parse_args()
    srv = build(a.workdir, write=a.write)
    if a.http:
        srv.run(transport="streamable-http", host="127.0.0.1", port=a.http)
    else:
        srv.run()  # stdio
