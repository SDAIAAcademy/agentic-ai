# /// script
# requires-python = ">=3.11"
# dependencies = ["a2a-sdk[http-server]>=1.2", "fastapi>=0.150", "uvicorn>=0.40", "openai>=3.30", "python-dotenv>=1.1"]
# ///
"""Madar's policy worker as an A2A agent (Lab 4.2).

    uv run servers/madar_a2a.py                 # http://127.0.0.1:8792, agent card at /.well-known/agent-card.json
    uv run servers/madar_a2a.py --port 8800

The worker is a small agent of its own (agentkit's loop with one tool, `search_policies`) behind the A2A protocol
(v1.0, JSON-RPC binding). Any A2A client can discover it from its agent card and send it a message; it answers with
a completed task whose artifact is the answer, citing policy IDs. It uses the same provider settings as the labs
(labs/.env).
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

LABS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LABS))

import uvicorn  # noqa: E402
from a2a.helpers import new_task_from_user_message, new_text_message, new_text_part  # noqa: E402
from a2a.server.agent_execution import AgentExecutor, RequestContext  # noqa: E402
from a2a.server.events import EventQueue  # noqa: E402
from a2a.server.request_handlers import DefaultRequestHandler  # noqa: E402
from a2a.server.routes import add_a2a_routes_to_fastapi, create_agent_card_routes, create_jsonrpc_routes  # noqa: E402
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater  # noqa: E402
from a2a.types import AgentCapabilities, AgentCard, AgentInterface, AgentSkill  # noqa: E402
from fastapi import FastAPI  # noqa: E402

import agentkit as K  # noqa: E402
import madar  # noqa: E402

SYSTEM = """You are Madar's policy specialist. Answer the question from Madar's policy documents only, using
search_policies, and cite the policy ID for every rule, e.g. (POL-06). If the policies don't cover it, say so.
You don't know anything about individual applications. Reply in the language of the question, in at most 5 sentences."""


class PolicyWorker(AgentExecutor):
    def __init__(self) -> None:
        self.m = madar.Madar(K.lab_dir("a2a_worker"), actor="policy-worker")

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        task = context.current_task or new_task_from_user_message(context.message)
        await event_queue.enqueue_event(task)
        updater = TaskUpdater(event_queue, task.id, task.context_id)
        await updater.start_work()
        question = context.get_user_input()
        try:
            run = await asyncio.to_thread(K.run_agent, question, self.m.tools(["search_policies"]), system=SYSTEM,
                                          max_steps=5, verbose=False)
        except Exception as e:  # noqa: BLE001 - report the failure to the caller as a failed task
            await updater.failed(new_text_message(f"Policy worker error: {type(e).__name__}: {e}",
                                                  context_id=task.context_id, task_id=task.id))
            return
        await updater.add_artifact([new_text_part(run.final or "(no answer)")], name="answer")
        await updater.complete()

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError("The policy worker answers in seconds; cancel is not supported.")


def card(port: int) -> AgentCard:
    return AgentCard(
        name="madar-policy-worker",
        description="Answers questions about Madar Municipal Services' permit policies, citing policy IDs (POL-01..10).",
        version="1.0.0",
        supported_interfaces=[AgentInterface(url=f"http://127.0.0.1:{port}/", protocol_binding="JSONRPC",
                                             protocol_version="1.0")],
        capabilities=AgentCapabilities(streaming=False),
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        skills=[AgentSkill(id="policy-answers", name="Policy answers",
                           description="Fees, deadlines, documents, appeals, refunds, inspections, food-truck zones.",
                           tags=["permits", "policy", "arabic", "english"],
                           examples=["What is the refund if I withdraw during review?", "ما مهلة الاعتراض على الرفض؟"])],
    )


def build_app(port: int) -> FastAPI:
    agent_card = card(port)
    handler = DefaultRequestHandler(agent_executor=PolicyWorker(), task_store=InMemoryTaskStore(), agent_card=agent_card)
    app = FastAPI(title="Madar policy worker (A2A)")
    add_a2a_routes_to_fastapi(app, agent_card_routes=create_agent_card_routes(agent_card),
                              jsonrpc_routes=create_jsonrpc_routes(handler, rpc_url="/"))
    return app


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8792)
    args = ap.parse_args()
    uvicorn.run(build_app(args.port), host="127.0.0.1", port=args.port, log_level="warning")
