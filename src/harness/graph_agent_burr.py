"""Multi-node graph agent whose per-node ReAct loop runs on Apache Burr.

It reuses the harness topic router, per-node tool isolation, and refusal
handling from :class:`~harness.graph_agent.GraphAgent` unchanged, and swaps only
the per-node execution engine to :class:`~harness.agent_burr.AgentLoopBurr` — a
ReAct loop expressed as a Burr state machine.

The result has no proprietary runtime dependencies: routing is plain harness
code and each node's reasoning/tool loop is open-source Burr driving the
harness's own ``LLMClient`` and ``ToolRegistry``.
"""

from __future__ import annotations

from harness.agent_burr import AgentLoopBurr
from harness.graph_agent import GraphAgent, NodeConfig


class GraphAgentBurr(GraphAgent):
    """Topic-routed multi-node agent with a Burr-backed per-node ReAct loop.

    Identical routing/refusal behavior to :class:`GraphAgent`; the only
    difference is that each selected node runs its reason→tools→loop cycle as an
    Apache Burr application instead of the plain Python loop.
    """

    def _make_agent(self, node_cfg: NodeConfig, node_prompt: str) -> AgentLoopBurr:
        return AgentLoopBurr(
            llm=self.llm,
            tools=node_cfg.tools,
            system_prompt=node_prompt,
            max_iterations=node_cfg.max_iterations,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
