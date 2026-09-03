"""Burr-based ReAct agent loop.

Implements a per-node ReAct control flow — reason with the LLM, execute any
requested tools, feed the results back, and loop — as an explicit
`Apache Burr <https://burr.dagworks.io/>`_ state machine.

State machine::

    reason ──(no tool calls / budget exhausted)──▶ finish
      │
      └──(tool calls requested)──▶ execute_tools ──▶ reason  (loop)

``reason`` is the only action that calls the LLM (one call per iteration, the
"tool planner" step). ``execute_tools`` runs the requested tools and makes no
LLM call. The loop repeats until the model stops requesting tools or the
per-node iteration budget is hit — identical semantics to :class:`AgentLoop`,
but structured as a Burr graph so the flow is inspectable, and (optionally)
traceable/checkpointable via Burr's tooling.

The interface (``run`` / ``stream`` returning :class:`AgentResponse` /
:class:`AgentEvent`) matches :class:`harness.agent.AgentLoop` so the two engines
are drop-in interchangeable behind the graph agent.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator

from burr.core import ApplicationBuilder, State, action, when

from harness.exceptions import HallucinatedToolError
from harness.llm.base import LLMClient
from harness.models import AgentEvent, AgentResponse, ChatMessage, Role
from harness.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class AgentLoopBurr:
    """ReAct agent loop implemented as an Apache Burr state machine.

    Drop-in replacement for :class:`harness.agent.AgentLoop`. A "turn" is one
    call to :meth:`run` (or :meth:`stream`); inside a turn the Burr app cycles
    ``reason -> execute_tools`` until the model returns a final answer or the
    ``max_iterations`` LLM-call budget is exhausted.
    """

    def __init__(
        self,
        llm: LLMClient,
        tools: ToolRegistry,
        system_prompt: str,
        max_iterations: int = 10,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> None:
        self.llm = llm
        self.tools = tools
        self.system_prompt = system_prompt
        self.max_iterations = max_iterations
        self.temperature = temperature
        self.max_tokens = max_tokens

    # -- Burr graph construction ------------------------------------------------

    def _build_app(self, user_message: str, history: list[ChatMessage] | None):
        """Construct a fresh Burr application for a single turn.

        Actions are defined as closures so they capture this instance's LLM
        client, tool registry, and sampling config without threading them
        through Burr state (which is reserved for per-turn conversation data).
        """
        llm = self.llm
        tools = self.tools
        tool_defs = tools.get_definitions() if len(tools) > 0 else None
        temperature = self.temperature
        max_tokens = self.max_tokens
        max_iterations = self.max_iterations

        @action(
            reads=["messages", "llm_calls"],
            writes=[
                "messages",
                "pending_tool_calls",
                "final_content",
                "done",
                "llm_calls",
                "iterations",
            ],
        )
        async def reason(state: State) -> State:
            """The tool-planner step: one LLM call to plan tools or finish."""
            llm_calls = state["llm_calls"] + 1
            response = await llm.chat(
                messages=state["messages"],
                tools=tool_defs,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            if not response.tool_calls:
                # Model produced a final answer — no tools requested.
                return state.update(
                    llm_calls=llm_calls,
                    iterations=llm_calls,
                    final_content=response.content or "",
                    pending_tool_calls=[],
                    done=True,
                )

            # Record the assistant turn (with its tool calls) before executing.
            messages = state["messages"] + [
                ChatMessage(
                    role=Role.ASSISTANT,
                    tool_calls=response.tool_calls,
                    content=response.content,
                )
            ]

            if llm_calls >= max_iterations:
                # Budget exhausted: we can't afford another reasoning call to
                # consume the tool results, so stop here (mirrors AgentLoop).
                logger.warning("Max iterations (%d) reached", max_iterations)
                return state.update(
                    llm_calls=llm_calls,
                    iterations=llm_calls,
                    messages=messages,
                    final_content=response.content
                    or "I've reached the maximum number of reasoning steps.",
                    pending_tool_calls=[],
                    done=True,
                )

            return state.update(
                llm_calls=llm_calls,
                messages=messages,
                pending_tool_calls=response.tool_calls,
                done=False,
            )

        @action(
            reads=["messages", "pending_tool_calls", "tool_calls_made"],
            writes=["messages", "tool_calls_made", "pending_tool_calls"],
        )
        async def execute_tools(state: State) -> State:
            """Run the planned tool calls and append their results."""
            messages = list(state["messages"])
            made = list(state["tool_calls_made"])
            for tc in state["pending_tool_calls"]:
                # A call to a tool that isn't wired to this node is a model
                # hallucination, not an infra fault. Surface it as a structured
                # error the analyzer grades as a task failure.
                if tc.name not in tools._tools:
                    raise HallucinatedToolError(
                        f"Tool '{tc.name}' not found in available tools. "
                        f"Available tools: {', '.join(sorted(tools.tool_names))}"
                    )
                logger.info("Executing tool: %s(%s)", tc.name, tc.arguments)
                result = await tools.execute(tc.name, tc.arguments)
                messages.append(
                    ChatMessage(
                        role=Role.TOOL, content=result, tool_call_id=tc.id, name=tc.name
                    )
                )
                made.append(
                    {"name": tc.name, "arguments": tc.arguments, "result": result}
                )
            return state.update(
                messages=messages, tool_calls_made=made, pending_tool_calls=[]
            )

        @action(reads=["final_content", "iterations", "tool_calls_made"], writes=[])
        async def finish(state: State) -> State:
            """Terminal action — the halt point that packages the result."""
            return state

        messages: list[ChatMessage] = [
            ChatMessage(role=Role.SYSTEM, content=self.system_prompt)
        ]
        if history:
            messages.extend(history)
        messages.append(ChatMessage(role=Role.USER, content=user_message))

        return (
            ApplicationBuilder()
            .with_actions(reason=reason, execute_tools=execute_tools, finish=finish)
            .with_transitions(
                ("reason", "finish", when(done=True)),
                ("reason", "execute_tools", when(done=False)),
                ("execute_tools", "reason"),
            )
            .with_entrypoint("reason")
            .with_state(
                messages=messages,
                pending_tool_calls=[],
                tool_calls_made=[],
                final_content="",
                llm_calls=0,
                iterations=0,
                done=False,
            )
            .build()
        )

    # -- Public API -------------------------------------------------------------

    async def run(
        self,
        user_message: str,
        history: list[ChatMessage] | None = None,
    ) -> AgentResponse:
        """Run the Burr ReAct loop to completion and return the final response."""
        app = self._build_app(user_message, history)
        _, _, state = await app.arun(halt_after=["finish"])
        return AgentResponse(
            content=state["final_content"] or "",
            tool_calls_made=list(state["tool_calls_made"]),
            iterations=state["iterations"],
        )

    async def stream(
        self,
        user_message: str,
        history: list[ChatMessage] | None = None,
    ) -> AsyncIterator[AgentEvent]:
        """Stream the loop, yielding tool_call / tool_result / final events.

        Because the Burr decomposition plans all of a step's tool calls in
        ``reason`` and then runs them in ``execute_tools``, tool_call events for
        an iteration are emitted before their tool_result events (rather than
        strictly interleaved). The set of events is identical to
        :meth:`AgentLoop.stream`.
        """
        app = self._build_app(user_message, history)
        emitted_results = 0
        async for act, _result, state in app.aiterate(halt_after=["finish"]):
            if act.name == "reason":
                for tc in state["pending_tool_calls"]:
                    yield AgentEvent(type="tool_call", tool_call=tc, tool_name=tc.name)
            elif act.name == "execute_tools":
                made = state["tool_calls_made"]
                for entry in made[emitted_results:]:
                    yield AgentEvent(
                        type="tool_result",
                        tool_name=entry["name"],
                        tool_result=entry["result"],
                    )
                emitted_results = len(made)
            elif act.name == "finish":
                yield AgentEvent(type="final", content=state["final_content"] or "")
