"""User simulator for automated benchmarking.

Provides LLM-powered user simulation that drives multi-turn conversations
with the agent based on task instructions.

Strategies:
  - human: Reads from stdin (interactive)
  - llm: LLM generates user responses from instruction + history
  - react: LLM reasons (chain-of-thought) before responding
  - verify: LLM generates response, then a verifier checks faithfulness
  - reflection: verify + self-correction loop on failure
"""

from __future__ import annotations

import abc
import logging
import re
from typing import TYPE_CHECKING

from harness.models import ChatMessage, Role

if TYPE_CHECKING:
    from harness.llm.base import LLMClient

logger = logging.getLogger(__name__)

STOP_SIGNAL = "###STOP###"


class UserSimulatorError(RuntimeError):
    """Raised when the user simulator cannot produce a usable response.

    This is an infrastructure failure (e.g. the LLM returned empty content on
    every retry) and must be surfaced as a trace error rather than silently
    treated as the user ending the conversation.
    """


_USER_SYSTEM_PROMPT = """\
You are simulating a user in a customer service conversation.
Follow the instruction below EXACTLY and COMPLETELY. You must embody
the persona, emotional state, and behavioral constraints described.

CRITICAL RULES:
- Generate one message at a time.
- Do NOT reveal information the user would not know.
- Only provide details relevant to the current step.
- Do NOT end the conversation early. Read your instruction carefully —
  if it says to keep pushing, demand something specific, or refuse certain
  responses, you MUST follow those directives even if the agent sounds helpful.
- Only respond with {stop} when ALL conditions in your instruction are met.
  If your instruction says "do not accept X" or "keep pushing until Y",
  you must continue the conversation until those specific conditions are satisfied.
- If the agent gives a partial resolution (e.g. refund but no credit, ticket but
  no immediate action), and your instruction requires MORE, keep demanding.
- Stay in character. If you are angry, stay angry. If you are suspicious, stay suspicious.

When — and ONLY when — every goal and exit condition in your instruction
has been fully satisfied, respond with exactly: {stop}

## Your instruction:
{instruction}"""

_REACT_SYSTEM_PROMPT = """\
You are simulating a user in a customer service conversation.
Follow the instruction below EXACTLY and COMPLETELY. You must embody
the persona, emotional state, and behavioral constraints described.

CRITICAL RULES:
- Do NOT reveal information the user would not know.
- Only provide details relevant to the current step.
- Do NOT end the conversation early. Your DEFAULT action is to CONTINUE
  the conversation. Stopping is the EXCEPTION, not the rule.
- The agent SAYING it will do something is NOT the same as it being DONE.
  "I'll process your refund" is a promise, not a confirmation. "I'll create
  a ticket" is not the same as providing a ticket number. Be skeptical.
- If your instruction says to keep pushing, demand specifics, or refuse
  certain responses, you MUST follow those directives NO MATTER WHAT.
- If the agent gives a partial resolution and your instruction requires MORE,
  keep demanding. Do not accept partial solutions as complete.
- If your instruction mentions a number of exchanges (e.g. "after 8 exchanges"),
  you MUST continue for at least that many exchanges before considering stopping.
- Stay in character at ALL times. If angry, stay angry. If confused, stay confused.
  Do not become cooperative or satisfied just because the agent is polite.
- When in doubt, CONTINUE the conversation. Only stop when you are 100% certain
  that every single exit condition has been explicitly met.

First, generate a Thought analyzing:
1. How many exchanges have happened so far? (Count them)
2. What has the agent ACTUALLY done (not just promised)?
3. List EVERY exit condition from my instruction. For each one, mark it
   [MET] or [NOT MET] with evidence.
4. Are there any behavioral directives I need to follow? (e.g. "refuse X",
   "demand Y", "don't provide Z until asked twice")
5. What should the user say next based on the instruction and persona?

Then generate the actual User Response on a new line.

Format:
Thought: <your detailed internal reasoning following steps 1-5 above>
User Response: <the message to send, staying fully in character>

ONLY output {stop} when EVERY exit condition is [MET]:
Thought: <reasoning listing ALL conditions as [MET] with evidence>
User Response: {stop}

## Your instruction:
{instruction}"""

_VERIFY_PROMPT = """\
You are a STRICT quality verifier for a simulated user in a customer service conversation.

The simulated user has an instruction they must follow. You must verify that
the user's proposed response is FAITHFUL to that instruction.

## User Instruction:
{instruction}

## Conversation so far (count the user messages to track exchanges):
{conversation}

## Proposed user response:
{response}

## STRICT Verification Process:

STEP 1: Extract ALL goals and exit conditions from the instruction.
List every single one explicitly.

STEP 2: For each goal/condition, check if it has been ACTUALLY satisfied
in the conversation so far (not just promised by the agent):
- "Refund confirmed" means the agent explicitly said the refund was processed
- "Ticket created" means a ticket number or explicit confirmation was given
- "Account info verified" means the agent actually showed/retrieved the info
- A minimum exchange count means COUNT the user messages so far

STEP 3: If the proposed response is {stop} or signals ending:
- If ANY goal/condition from Step 1 is NOT satisfied → FAIL immediately
- If a minimum exchange count exists and hasn't been reached → FAIL immediately
- If the instruction has multiple issues/phases and not all are addressed → FAIL

STEP 4: Check behavioral faithfulness:
- Does the response stay in character (persona, emotional state)?
- Does it follow behavioral directives (e.g. "refuse X", "don't reveal Y until asked twice")?
- Does it avoid revealing information the user wouldn't know?
- If the instruction says "address issues ONE AT A TIME", is the user jumping ahead?

Respond with EXACTLY one of:
- PASS: The response is faithful to the instruction.
- FAIL: <specific explanation citing which goal/condition is unmet>

CRITICAL: Be EXTREMELY strict about {stop} responses. The #1 error is
premature stopping. If there is ANY doubt, FAIL. Only PASS a stop when
you can point to specific evidence in the conversation for EVERY condition."""

_REFLECTION_PROMPT = """\
You are helping a simulated user correct their response.

The user's previous response was rejected by a verifier for the following reason:
{failure_reason}

## User Instruction:
{instruction}

## Conversation so far:
{conversation}

## Rejected response:
{rejected_response}

Generate a corrected response that addresses the verifier's feedback while
staying faithful to the instruction. Only output the corrected user message
(no explanation, no "Thought:" prefix). Stay in character."""


class BaseUserSimulator(abc.ABC):
    """Abstract base class for user simulators."""

    @abc.abstractmethod
    def reset(self, instruction: str) -> None:
        """Reset the simulator with a new task instruction."""

    @abc.abstractmethod
    async def step(self, agent_message: str) -> str:
        """Generate the next user message given the agent's response.

        Args:
            agent_message: The agent's latest response. Empty string on first turn.

        Returns:
            The simulated user's message, or a string containing STOP_SIGNAL
            when the conversation should end.
        """


class HumanUserSimulator(BaseUserSimulator):
    """Passthrough simulator that reads from stdin."""

    def reset(self, instruction: str) -> None:
        print(f"\n[Task Instruction]\n{instruction}\n")

    async def step(self, agent_message: str) -> str:
        if agent_message:
            print(f"\nAgent: {agent_message}")
        try:
            return input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            return STOP_SIGNAL


class LLMUserSimulator(BaseUserSimulator):
    """LLM-powered user simulator driven by task instructions."""

    def __init__(
        self,
        llm: LLMClient,
        temperature: float = 0.5,
        max_tokens: int = 256,
    ) -> None:
        self.llm = llm
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._system_prompt: str = ""
        self._instruction: str = ""
        self._messages: list[ChatMessage] = []

    def reset(self, instruction: str) -> None:
        self._instruction = instruction
        self._system_prompt = _USER_SYSTEM_PROMPT.format(
            instruction=instruction, stop=STOP_SIGNAL
        )
        self._messages = [ChatMessage(role=Role.SYSTEM, content=self._system_prompt)]

    async def step(self, agent_message: str) -> str:
        # The simulator LLM role-plays the persona: its own turns are the
        # "assistant" completions, and the agent (its interlocutor) is the
        # "user". This is the standard framing and, crucially, keeps the message
        # list ending on a user turn so providers that require it (Vertex/Gemini
        # return empty content when the last turn is `model`) will generate.
        if agent_message and agent_message.strip():
            self._messages.append(ChatMessage(role=Role.USER, content=agent_message))

        # Ensure there's always at least one user turn to prompt generation
        has_user_msg = any(m.role == Role.USER for m in self._messages)
        if not has_user_msg:
            self._messages.append(
                ChatMessage(
                    role=Role.USER,
                    content="Begin the conversation as the user described in your instruction.",
                )
            )

        # Retry up to 3 times if LLM returns empty content
        for attempt in range(3):
            response = await self.llm.chat(
                messages=self._messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            logger.debug(
                "User sim LLM response (attempt %d): content=%r, finish_reason=%s",
                attempt + 1,
                response.content,
                response.finish_reason,
            )
            content = (response.content or "").strip()
            if content:
                break
            logger.warning("User sim got empty response (attempt %d/3)", attempt + 1)
        else:
            raise UserSimulatorError(
                "user simulator returned empty content after 3 attempts"
            )
        self._messages.append(ChatMessage(role=Role.ASSISTANT, content=content))
        return content


class ReactUserSimulator(LLMUserSimulator):
    """User simulator with chain-of-thought reasoning.

    The LLM first generates a Thought, then a User Response.
    Only the User Response is returned to the agent.
    """

    def reset(self, instruction: str) -> None:
        self._instruction = instruction
        self._system_prompt = _REACT_SYSTEM_PROMPT.format(
            instruction=instruction, stop=STOP_SIGNAL
        )
        self._messages = [ChatMessage(role=Role.SYSTEM, content=self._system_prompt)]

    async def step(self, agent_message: str) -> str:
        if agent_message and agent_message.strip():
            self._messages.append(ChatMessage(role=Role.USER, content=agent_message))

        has_user_msg = any(m.role == Role.USER for m in self._messages)
        if not has_user_msg:
            self._messages.append(
                ChatMessage(
                    role=Role.USER,
                    content="Begin the conversation as the user described in your instruction.",
                )
            )

        # Retry up to 3 times if LLM returns empty content
        for attempt in range(3):
            response = await self.llm.chat(
                messages=self._messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            logger.debug(
                "React user sim LLM response (attempt %d): content=%r, finish_reason=%s",
                attempt + 1,
                response.content,
                response.finish_reason,
            )
            raw = (response.content or "").strip()
            if raw:
                break
            logger.warning(
                "React user sim got empty response (attempt %d/3)", attempt + 1
            )
        else:
            raise UserSimulatorError(
                "react user simulator returned empty content after 3 attempts"
            )

        # Store full output (with thought) in history for context
        self._messages.append(ChatMessage(role=Role.ASSISTANT, content=raw))

        # Parse out just the User Response
        return self._parse_response(raw)

    @staticmethod
    def _parse_response(raw: str) -> str:
        """Extract the User Response from the chain-of-thought output."""
        match = re.search(r"User Response:\s*(.+)", raw, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
        # Fallback: return last line if parsing fails
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        return lines[-1] if lines else raw


class VerifyUserSimulator(ReactUserSimulator):
    """User simulator with verification step.

    Generates a response using the React strategy, then runs a separate
    verifier LLM call to check if the response is faithful to the task
    instruction. If verification fails, retries up to max_retries times.
    """

    def __init__(
        self,
        llm: LLMClient,
        temperature: float = 0.5,
        max_tokens: int = 256,
        max_retries: int = 3,
    ) -> None:
        super().__init__(llm=llm, temperature=temperature, max_tokens=max_tokens)
        self.max_retries = max_retries

    async def step(self, agent_message: str) -> str:
        if agent_message and agent_message.strip():
            self._messages.append(ChatMessage(role=Role.USER, content=agent_message))

        has_user_msg = any(m.role == Role.USER for m in self._messages)
        if not has_user_msg:
            self._messages.append(
                ChatMessage(
                    role=Role.USER,
                    content="Begin the conversation as the user described in your instruction.",
                )
            )

        for attempt in range(self.max_retries):
            # Generate response
            response = await self.llm.chat(
                messages=self._messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            raw = (response.content or "").strip()
            parsed = self._parse_response(raw)

            # Verify the response
            is_valid, reason = await self._verify(parsed)
            if is_valid:
                self._messages.append(ChatMessage(role=Role.ASSISTANT, content=raw))
                return parsed

            logger.info(
                "Verify attempt %d/%d failed: %s", attempt + 1, self.max_retries, reason
            )

        # All retries exhausted — use the last response anyway
        logger.warning(
            "Verification failed after %d attempts, using last response",
            self.max_retries,
        )
        self._messages.append(ChatMessage(role=Role.ASSISTANT, content=raw))
        return parsed

    async def _verify(self, response: str) -> tuple[bool, str]:
        """Verify if the response is faithful to the instruction.

        Returns (is_valid, failure_reason).
        """
        conversation = self._format_conversation()
        verify_prompt = _VERIFY_PROMPT.format(
            instruction=self._instruction,
            conversation=conversation,
            response=response,
            stop=STOP_SIGNAL,
        )

        result = await self.llm.chat(
            messages=[ChatMessage(role=Role.USER, content=verify_prompt)],
            temperature=0.0,
            max_tokens=300,
        )
        verdict = (result.content or "").strip()

        if verdict.upper().startswith("PASS"):
            return True, ""

        # Extract failure reason
        reason = verdict.replace("FAIL:", "").replace("FAIL", "").strip()
        return False, reason or "Response not faithful to instruction"

    def _format_conversation(self) -> str:
        """Format conversation history for the verifier prompt.

        Role mapping follows the simulator's convention (see
        :meth:`LLMUserSimulator.step`): the persona the LLM role-plays speaks as
        ``ASSISTANT`` — i.e. the simulated *User* — while the agent under test
        (the interlocutor) is stored as ``USER`` and shown as *Agent*.
        """
        lines = []
        for msg in self._messages:
            if msg.role == Role.SYSTEM:
                continue
            elif msg.role == Role.ASSISTANT:
                # The persona's own turns = the simulated user.
                lines.append(f"User: {msg.content}")
            # The agent under test — skip the synthetic kickoff prompt.
            elif (
                msg.role == Role.USER
                and msg.content
                and "Begin the conversation" not in msg.content
            ):
                lines.append(f"Agent: {msg.content}")
        return "\n".join(lines) if lines else "(conversation start)"


class ReflectionUserSimulator(VerifyUserSimulator):
    """User simulator with verify + self-correction loop.

    Like VerifyUserSimulator, but on verification failure, it asks the LLM
    to reflect on the failure and produce a corrected response before
    retrying verification.
    """

    async def step(self, agent_message: str) -> str:
        if agent_message and agent_message.strip():
            self._messages.append(ChatMessage(role=Role.USER, content=agent_message))

        has_user_msg = any(m.role == Role.USER for m in self._messages)
        if not has_user_msg:
            self._messages.append(
                ChatMessage(
                    role=Role.USER,
                    content="Begin the conversation as the user described in your instruction.",
                )
            )

        # Generate initial response
        response = await self.llm.chat(
            messages=self._messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        raw = (response.content or "").strip()
        parsed = self._parse_response(raw)

        for attempt in range(self.max_retries):
            # Verify
            is_valid, reason = await self._verify(parsed)
            if is_valid:
                self._messages.append(ChatMessage(role=Role.ASSISTANT, content=raw))
                return parsed

            logger.info(
                "Reflection attempt %d/%d — reason: %s",
                attempt + 1,
                self.max_retries,
                reason,
            )

            # Reflect and correct
            parsed = await self._reflect(parsed, reason)
            raw = parsed  # corrected response becomes the raw for history

        # All retries exhausted — use last corrected response
        logger.warning(
            "Reflection failed after %d attempts, using last response", self.max_retries
        )
        self._messages.append(ChatMessage(role=Role.ASSISTANT, content=raw))
        return parsed

    async def _reflect(self, rejected_response: str, failure_reason: str) -> str:
        """Generate a corrected response based on verifier feedback."""
        conversation = self._format_conversation()
        reflection_prompt = _REFLECTION_PROMPT.format(
            instruction=self._instruction,
            conversation=conversation,
            rejected_response=rejected_response,
            failure_reason=failure_reason,
        )

        result = await self.llm.chat(
            messages=[ChatMessage(role=Role.USER, content=reflection_prompt)],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        corrected = (result.content or "").strip()
        return corrected if corrected else rejected_response


def load_user_simulator(
    strategy: str = "llm",
    llm: LLMClient | None = None,
    temperature: float = 0.5,
    max_tokens: int = 256,
    max_retries: int = 3,
) -> BaseUserSimulator:
    """Factory function to create a user simulator.

    Args:
        strategy: One of "human", "llm", "react", "verify", "reflection".
        llm: LLM client (required for LLM-based strategies).
        temperature: Sampling temperature for LLM strategies.
        max_tokens: Max tokens per user response.
        max_retries: Max verification/reflection retries (for verify/reflection).

    Returns:
        A configured BaseUserSimulator instance.
    """
    if strategy == "human":
        return HumanUserSimulator()

    if llm is None:
        raise ValueError(f"LLM client required for strategy '{strategy}'")

    if strategy == "reflection":
        return ReflectionUserSimulator(
            llm=llm,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
        )

    if strategy == "verify":
        return VerifyUserSimulator(
            llm=llm,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
        )

    if strategy == "react":
        return ReactUserSimulator(
            llm=llm, temperature=temperature, max_tokens=max_tokens
        )

    if strategy == "llm":
        return LLMUserSimulator(llm=llm, temperature=temperature, max_tokens=max_tokens)

    raise ValueError(
        f"Unknown user simulator strategy: '{strategy}'. "
        "Use 'human', 'llm', 'react', 'verify', or 'reflection'."
    )
