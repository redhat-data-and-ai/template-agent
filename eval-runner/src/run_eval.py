#!/usr/bin/env python3
r"""Standalone live-eval runner for lightspeed-evaluation.

Calls a running agent for each query in eval_data.yaml via the LangGraph API,
collects real response + tool calls + contexts from the SSE stream, writes a
populated YAML, then invokes lightspeed-eval to score and produce a report.

Usage:
    python run_eval.py                                         # local agent
    python run_eval.py --agent-url https://agent.example.com  # deployed agent
    python run_eval.py \\
        --agent-url http://localhost:5002 \\
        --eval-data custom_data.yaml \\
        --output-dir ./results

Requires:
    GOOGLE_API_KEY  or  GOOGLE_APPLICATION_CREDENTIALS_CONTENT  (Vertex AI service account JSON)
    pip install "lightspeed-evaluation @ git+https://github.com/lightspeed-core/lightspeed-evaluation.git@v0.7.0"

Agent API (standard LangGraph Platform — no custom endpoints needed):
    POST /threads                              → create a thread
    POST /threads/{thread_id}/runs/stream      → stream a run / resume interrupt
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

import httpx
import yaml

SCRIPT_DIR = Path(__file__).parent
DEFAULT_SYSTEM = SCRIPT_DIR / "system.yaml"
DEFAULT_EVAL_DATA = SCRIPT_DIR / "eval_data.yaml"
DEFAULT_OUTPUT_DIR = Path("eval_output_live")
DEFAULT_AGENT_URL = "http://localhost:5002"
AGENT_SSL_VERIFY = os.environ.get("AGENT_SSL_VERIFY", "true").lower() not in (
    "false",
    "0",
)
ASSISTANT_ID = "agent"
MAX_HITL_APPROVALS = 50  # safety cap on auto-approvals per turn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stderr,
)
log = logging.getLogger("run_eval")


# ── SSE parsing ──────────────────────────────────────────────────────────────


def _parse_sse_stream(lines: list[str]) -> list[tuple[str, Any]]:
    """Parse LangGraph SSE stream lines into (event_type, data) pairs.

    LangGraph SSE format per event block:
        event: <type>
        data: <json>
        id: <id>
        <blank line>
    """
    events: list[tuple[str, Any]] = []
    current_event = "message"
    current_data: str | None = None

    for line in lines:
        line = line.rstrip()
        if line.startswith("event:"):
            current_event = line[len("event:") :].strip()
        elif line.startswith("data:"):
            current_data = line[len("data:") :].strip()
        elif line == "" and current_data is not None:
            try:
                events.append((current_event, json.loads(current_data)))
            except json.JSONDecodeError:
                pass
            current_data = None
            current_event = "message"

    if current_data is not None:
        try:
            events.append((current_event, json.loads(current_data)))
        except json.JSONDecodeError:
            pass

    return events


def _extract_text(content: Any) -> str:
    """Extract plain text from LangChain message content (str or block list)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            (
                b.get("text", "")
                if isinstance(b, dict) and b.get("type") == "text"
                else b
            )
            for b in content
            if isinstance(b, (str, dict))
        ]
        return "\n".join(str(p) for p in parts if p)
    return ""


def _resolve_tool_name(tc: dict[str, Any]) -> str:
    """Unwrap 'task' → actual subagent name for readability."""
    name = tc.get("name", "")
    if name == "task":
        return tc.get("args", {}).get("subagent_type", "task")  # type: ignore[no-any-return]
    return name  # type: ignore[no-any-return]


def _unwrap_update_data(data: Any) -> dict[str, Any]:
    """Unwrap [namespace, update_dict] from stream_subgraphs=True, or return as-is."""
    if isinstance(data, list) and len(data) == 2 and isinstance(data[1], dict):
        return data[1]
    if isinstance(data, dict):
        return data
    return {}


def _has_interrupt(events: list[tuple[str, Any]]) -> bool:
    """Return True if the stream contains a HITL interrupt event."""
    for event_type, data in events:
        if event_type == "updates":
            update = _unwrap_update_data(data)
            if "__interrupt__" in update:
                return True
    return False


def _count_interrupted_tool_calls(events: list[tuple[str, Any]]) -> int:
    """Return number of tool calls pending approval in the interrupt payload."""
    for event_type, data in events:
        if event_type == "updates":
            update = _unwrap_update_data(data)
            interrupt_list = update.get("__interrupt__", [])
            if interrupt_list:
                value = interrupt_list[0].get("value", {})
                return len(value.get("action_requests", [1]))
    return 1


def _extract_interrupt_response(events: list[tuple[str, Any]]) -> str:
    """Extract the HITL interrupt description as the pre-approval response.

    The __interrupt__ payload has action_requests with:
      - description: already-formatted human-readable approval request
      - name / args: tool name and arguments pending approval
    We use description if present, otherwise build from name + args.
    """
    for event_type, data in events:
        if event_type != "updates":
            continue
        update = _unwrap_update_data(data)
        interrupt_list = update.get("__interrupt__", [])
        if not interrupt_list:
            continue
        value = interrupt_list[0].get("value", {})
        requests = value.get("action_requests", [])
        if not requests:
            continue
        parts = []
        for req in requests:
            if req.get("description"):
                parts.append(req["description"])
            else:
                name = req.get("name", "unknown_tool")
                args = req.get("args", {})
                parts.append(
                    f"Tool execution requires approval\n\nTool: {name}\nArgs: {args}"
                )
        return "\n\n".join(parts)
    return ""


# ── Agent interaction ────────────────────────────────────────────────────────


def _headers(auth_token: str | None) -> dict[str, str]:
    h = {"Content-Type": "application/json"}
    if auth_token:
        h["Authorization"] = f"Bearer {auth_token}"
    return h


def _create_thread(
    agent_url: str, auth_token: str | None, timeout: int
) -> str:  # pragma: no cover
    url = f"{agent_url.rstrip('/')}/threads"
    with httpx.Client(timeout=timeout) as c:
        resp = c.post(url, json={}, headers=_headers(auth_token))
        resp.raise_for_status()
        return resp.json()["thread_id"]  # type: ignore[no-any-return]


def _stream_one_request(  # pragma: no cover
    agent_url: str,
    thread_id: str,
    body: dict[str, Any],
    auth_token: str | None,
    timeout: int,
) -> tuple[list[tuple[str, Any]], bool]:
    """Execute one streaming run request and return (events, interrupted).

    `interrupted` is True when the stream ended with a HITL __interrupt__ node.
    """
    url = f"{agent_url.rstrip('/')}/threads/{thread_id}/runs/stream"
    raw: list[str] = []
    # Use a separate read timeout: SSE streams can stay open for the full turn
    # duration (especially with HITL auto-approvals + slow LLM under parallel load).
    stream_timeout = httpx.Timeout(connect=10.0, read=timeout, write=10.0, pool=5.0)
    with httpx.Client(timeout=stream_timeout, verify=AGENT_SSL_VERIFY) as c:
        with c.stream("POST", url, json=body, headers=_headers(auth_token)) as resp:
            resp.raise_for_status()
            try:
                for line in resp.iter_lines():
                    raw.append(line)
            except Exception:
                # Partial read (e.g. peer closed connection mid-stream).
                # Use whatever lines arrived — HITL interrupts are emitted
                # early so they're usually already in `raw`.
                pass

    events = _parse_sse_stream(raw)
    return events, _has_interrupt(events)


def _extract_node_updates(data: Any) -> Iterable[Any]:
    """Return node-update dicts from a LangGraph updates event (iterable, no copy).

    Handles two formats emitted by the LangGraph Platform:
      - Top-level graph:  {"node_name": {messages: [...]}, ...}
      - Subgraph (stream_subgraphs=True): [["ns1", "ns2", ...], {"node_name": {...}}]
        The first element is the namespace tuple; the second is the actual update dict.
    """
    if isinstance(data, list) and len(data) == 2 and isinstance(data[1], dict):
        return data[1].values()
    if isinstance(data, dict):
        return data.values()
    return ()


_INTERNAL_TOOLS = frozenset(
    {
        "write_todos",
        "task",
        "read_file",
        "write_file",
        "ls",
        "glob",
        "grep",
        "execute_command",
        "compact_conversation",
    }
)


def _collect_from_events(
    events: list[tuple[str, Any]],
) -> tuple[list[str], list[dict[str, Any]], list[str], bool]:
    """Extract (ai_text_blocks, tool_calls, tool_result_texts, ai_before_contexts) from events.

    ai_before_contexts is True when the only AI text appeared BEFORE the first tool
    result — the delegation pattern where the orchestrator greeted/handed off before
    the subagent produced the actual answer. When False, the AI text came after tool
    results and IS the real response.

    Uses two complementary event sources:
    - "updates" events  → final AI response text and tool-result contexts
    - "events" events   → on_tool_start captures every MCP tool call at any subgraph
                          depth (e.g. calculate_bmi_value inside the analyst subagent)
    Internal housekeeping tools (write_todos, task) are excluded from tool_calls.
    """
    ai_texts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    contexts: list[str] = []
    seen_tool_runs: set[str] = set()  # deduplicate by run_id
    last_ai_text_idx: int = -1  # position of last AI text in the event sequence
    first_context_idx: int = -1  # position of first external tool result
    event_pos: int = 0

    for event_type, raw_data in events:
        event_pos += 1
        # When stream_mode is a list, LangGraph wraps data as ["mode", payload]
        data = (
            raw_data[1]
            if (isinstance(raw_data, list) and len(raw_data) == 2)
            else raw_data
        )

        # ── on_tool_start / on_tool_end: captures MCP tool calls at any depth ──
        if event_type == "events" and isinstance(data, dict):
            event_name = data.get("event", "")
            name = data.get("name", "")
            run_id = data.get("run_id", "")

            if event_name == "on_tool_start":
                if name in _INTERNAL_TOOLS or (run_id and run_id in seen_tool_runs):
                    pass
                else:
                    if run_id:
                        seen_tool_runs.add(run_id)
                    args = data.get("data", {}).get("input", {})
                    if isinstance(args, dict):
                        tool_calls.append({"tool_name": name, "arguments": args})

            elif event_name == "on_tool_end":
                # Collect tool outputs as contexts for ragas:faithfulness.
                # on_tool_end output may be a plain string OR a LangChain ToolMessage
                # dict like {"content": [{"type": "text", "text": "..."}], ...}.
                # Extract just the text content so ragas gets clean output, not metadata.
                if name and name not in _INTERNAL_TOOLS:
                    output = data.get("data", {}).get("output", "")
                    if output is not None and output != "":
                        if isinstance(output, dict):
                            # LangChain ToolMessage: extract text from content list
                            content = output.get("content", "")
                            if isinstance(content, list):
                                text = " ".join(
                                    c.get("text", "")
                                    for c in content
                                    if isinstance(c, dict) and c.get("type") == "text"
                                )
                            elif isinstance(content, str):
                                text = content
                            else:
                                text = ""  # don't stringify unknown content types
                        elif isinstance(output, list):
                            # list of content blocks — extract text items
                            text = " ".join(
                                c.get("text", "")
                                for c in output
                                if isinstance(c, dict) and c.get("type") == "text"
                            )
                        elif isinstance(output, str):
                            text = output
                        elif isinstance(output, (int, float)):
                            text = str(output)
                        else:
                            text = ""  # don't stringify unknown output types
                        if text.strip():
                            if first_context_idx == -1:
                                first_context_idx = event_pos
                            contexts.append(text.strip())

            continue

        # ── updates: AI response text, tool calls, tool-result contexts ─────
        if event_type != "updates":
            continue
        for update in _extract_node_updates(data):  # data already unwrapped above
            if not isinstance(update, dict):
                continue
            for msg in update.get("messages", []):
                if not isinstance(msg, dict):
                    continue
                msg_type = msg.get("type", "")
                if msg_type == "ai":
                    text = _extract_text(msg.get("content", ""))
                    if text:
                        ai_texts.append(text)
                        last_ai_text_idx = event_pos
                    for tc in msg.get("tool_calls", []):
                        resolved = _resolve_tool_name(tc)
                        if resolved not in _INTERNAL_TOOLS:
                            tool_calls.append(
                                {
                                    "tool_name": resolved,
                                    "arguments": tc.get("args", {}),
                                }
                            )
                elif msg_type == "tool":
                    # Skip results from internal housekeeping tools — their outputs
                    # (todo updates, task status, etc.) are not user-facing context.
                    if msg.get("name", "") in _INTERNAL_TOOLS:
                        continue
                    content = _extract_text(msg.get("content", ""))
                    if content:
                        if first_context_idx == -1:
                            first_context_idx = event_pos
                        contexts.append(content)

    # Delegation pattern: AI text appeared BEFORE any tool result in the stream.
    # True → orchestrator greeted/delegated before the subagent answered.
    # False → AI text came after tool results → it IS the real response.
    ai_before_contexts = (
        last_ai_text_idx != -1
        and first_context_idx != -1
        and last_ai_text_idx < first_context_idx
    )
    return ai_texts, tool_calls, contexts, ai_before_contexts


def _last_nonempty(texts: list[str]) -> str:
    for t in reversed(texts):
        if t.strip():
            return t
    return ""


def _call_agent(  # pragma: no cover
    agent_url: str,
    query: str,
    thread_id: str,
    auth_token: str | None,
    timeout: int,
) -> dict[str, Any]:
    """Run one conversation turn and collect response/tool_calls/contexts.

    Always auto-approves HITL interrupts so the run completes end-to-end.

    Returns both the pre-approval response (what the agent said when it paused
    to ask for human approval) and the final post-approval response, so callers
    can choose which one to score depending on what they are testing.
    """
    all_ai_texts: list[str] = []
    all_tool_calls: list[dict[str, Any]] = []
    all_contexts: list[str] = []
    pre_approval_response: str = ""
    was_interrupted: bool = False

    # Initial run
    body: dict[str, Any] = {
        "assistant_id": ASSISTANT_ID,
        "input": {"messages": [{"role": "human", "content": query}]},
        "stream_mode": ["updates", "events"],
        "stream_subgraphs": True,
    }
    events, interrupted = _stream_one_request(
        agent_url, thread_id, body, auth_token, timeout
    )
    ai_texts, tcs, ctxs, ai_before_ctx = _collect_from_events(events)
    all_ai_texts.extend(ai_texts)
    all_tool_calls.extend(tcs)
    all_contexts.extend(ctxs)
    delegation_pattern = ai_before_ctx  # greeting before tool results

    if interrupted:
        was_interrupted = True
        # Prefer any text the agent emitted before pausing; fall back to the
        # interrupt payload (tool name + args) which is what actually triggered
        # the approval gate — this gives intent_eval something to score.
        pre_approval_response = _last_nonempty(ai_texts) or _extract_interrupt_response(
            events
        )

    # Auto-approve HITL interrupts until the run completes.
    approvals = 0
    while interrupted and approvals < MAX_HITL_APPROVALS:
        n_decisions = _count_interrupted_tool_calls(events)
        approvals += 1
        resume_body: dict[str, Any] = {
            "assistant_id": ASSISTANT_ID,
            "command": {"resume": {"decisions": [{"type": "approve"}] * n_decisions}},
            "stream_mode": ["updates", "events"],
            "stream_subgraphs": True,
        }
        events, interrupted = _stream_one_request(
            agent_url, thread_id, resume_body, auth_token, timeout
        )
        ai_texts, tcs, ctxs, ai_before_ctx = _collect_from_events(events)
        all_ai_texts.extend(ai_texts)
        all_tool_calls.extend(tcs)
        all_contexts.extend(ctxs)
        # Update based on the most recent non-empty stream: if this stream
        # produced AI text AFTER contexts (real final response), cancel any
        # earlier delegation flag so we don't override with raw tool output.
        if ai_texts:
            delegation_pattern = ai_before_ctx

    if interrupted:
        log.warning(
            "HITL approval cap (%d) reached — run may be incomplete", MAX_HITL_APPROVALS
        )

    final_response = _last_nonempty(all_ai_texts)

    # Delegation pattern: orchestrator greeted/handed off BEFORE the tool results
    # arrived. The subagent's output is in contexts, not a second AI message.
    # Only apply when ordering confirms the AI text came before any tool result
    # (greeting → delegate → subagent answers via context).
    # Not applied when AI text came AFTER tool results (= real human-readable response).
    if delegation_pattern and all_contexts:
        last_ctx = all_contexts[-1].strip()
        if last_ctx:
            log.debug(
                "Delegation pattern (AI before contexts) — using last context as scored response"
            )
            final_response = last_ctx

    return {
        "response": final_response,
        "pre_approval_response": pre_approval_response,  # what agent said when it paused
        "was_interrupted": was_interrupted,
        "tool_calls_made": all_tool_calls,
        "contexts": all_contexts,
    }


# ── Dataset population ───────────────────────────────────────────────────────

_AGENT_ERROR_RESPONSE = "[agent error: no response collected]"


def _strip_args_for_no_arg_expected(
    actual_calls: list[dict],
    expected_tool_calls: list[list[dict]],
) -> list[dict]:
    """Strip arguments from actual calls for tools whose expected has no arguments.

    lightspeed-eval's _compare_tool_arguments fails when actual has extra keys
    not present in expected (even when expected args is empty {}). Stripping
    arguments from the actual call for those tools makes the comparison {}=={},
    implementing "match tool name only, any arguments" semantics from the UI.
    """
    # Collect tool names where expected arguments are empty / absent
    no_arg_tools: set[str] = set()
    for pattern in expected_tool_calls:
        for tc in pattern:
            if not tc.get("arguments"):  # None, {}, or missing
                no_arg_tools.add(tc.get("tool_name", ""))

    if not no_arg_tools:
        return actual_calls

    modified = False
    result = []
    for tc in actual_calls:
        if tc.get("tool_name") in no_arg_tools:
            result.append({"tool_name": tc["tool_name"]})
            modified = True
        else:
            result.append(tc)
    return result if modified else actual_calls  # skip copy when nothing stripped


def _dedup_tool_calls(tool_calls: list[dict]) -> list[dict]:
    """Remove duplicate tool calls keeping the first occurrence.

    Duplicates arise when the orchestrator SSE stream and the subagent
    checkpoint_blobs both surface the same call (e.g. calculate_bmi).
    Two calls are considered equal when tool_name and arguments match.
    """
    seen: set[str] = set()
    unique: list[dict] = []
    for tc in tool_calls:
        # Use tool_name + null-separator + serialised args; avoids a throwaway dict
        key = (
            tc.get("tool_name", "")
            + "\x00"
            + json.dumps(tc.get("arguments", {}), sort_keys=True)
        )
        if key not in seen:
            seen.add(key)
            unique.append(tc)
    return unique


def _fetch_subagent_tool_calls(  # pragma: no cover
    agent_url: str,
    thread_id: str,
    auth_token: str | None,
    timeout: int,
    *,
    retries: int = 4,
    retry_delay: float = 3.0,
) -> list[dict[str, Any]]:
    """Fetch subagent tool calls (calculate_bmi, send_email, etc.) from the agent.

    Calls GET /v1/eval/thread-tool-calls/{thread_id} which reads directly from
    Postgres checkpoint_blobs across all subagent namespaces — bypassing the
    LangGraph HTTP API limitation that only exposes subgraph state during interrupts.

    Retries are necessary in parallel eval runs: LangGraph Platform dispatches
    the analyst as a background task and its checkpoint_blobs reach Postgres
    slightly after the main graph's SSE stream closes. Without retries the fetch
    races the checkpoint write and returns an empty list.
    """
    url = f"{agent_url.rstrip('/')}/v1/eval/thread-tool-calls/{thread_id}"
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            with httpx.Client(timeout=timeout, verify=AGENT_SSL_VERIFY) as c:
                resp = c.get(url, headers=_headers(auth_token))
                resp.raise_for_status()
                tool_calls = resp.json().get("tool_calls", [])
            if tool_calls:
                return tool_calls  # type: ignore[no-any-return]
        except Exception as exc:
            last_exc = exc
        # Sleep before retry regardless of whether the attempt failed or returned empty.
        if attempt < retries - 1:
            time.sleep(retry_delay * (attempt + 1))
    if last_exc:
        log.warning(
            "subagent_tool_calls_fetch_failed after %d attempts (%s)",
            retries,
            type(last_exc).__name__,
        )
    return []


def _populate_group(  # pragma: no cover
    group: dict[str, Any],
    agent_url: str,
    auth_token: str | None,
    timeout: int,
) -> dict[str, Any]:
    """Run all turns in one conversation group and return the populated group."""
    group_id = group.get("conversation_group_id", "unknown")
    log.info("[%s] starting", group_id)

    try:
        thread_id = _create_thread(agent_url, auth_token, timeout)
    except Exception as exc:
        log.error(
            "[%s] thread_create_failed (%s): %s", group_id, type(exc).__name__, exc
        )
        for turn in group.get("turns", []):
            turn["response"] = _AGENT_ERROR_RESPONSE
        return group

    for turn in group.get("turns", []):
        turn_id = turn.get("turn_id", "?")
        query = turn["query"]
        log.info("[%s] turn=%s running", group_id, turn_id)

        try:
            result = _call_agent(agent_url, query, thread_id, auth_token, timeout)
        except Exception as exc:
            log.error(
                "[%s] turn=%s agent_call_failed (%s)",
                group_id,
                turn_id,
                type(exc).__name__,
            )
            result = {
                "response": _AGENT_ERROR_RESPONSE,
                "tool_calls_made": [],
                "contexts": [],
            }

        # For HITL turns, score the pre-approval response (what the agent said
        # when it paused) rather than the final post-approval response.
        is_hitl_turn = turn.pop("hitl", False)
        if is_hitl_turn and result.get("was_interrupted"):
            scored_response = (
                result.get("pre_approval_response") or _AGENT_ERROR_RESPONSE
            )
            hint = "pre-approval"
        else:
            scored_response = result["response"] or _AGENT_ERROR_RESPONSE
            hint = "post-approval" if result.get("was_interrupted") else "direct"

        # Normalize whitespace — collapse \n and extra spaces so the LLM judge
        # scores on content correctness, not on line-break formatting differences.
        turn["response"] = " ".join(scored_response.split())

        # Merge orchestrator + subagent tool calls (subagent fetch only for tool_use evals)
        all_tool_calls = list(result["tool_calls_made"])
        if group.get("tag") == "tool_use":
            subagent_tcs = _fetch_subagent_tool_calls(
                agent_url, thread_id, auth_token, timeout
            )
            all_tool_calls.extend(subagent_tcs)

        # Deduplicate: orchestrator stream and subagent checkpoint can both surface
        # the same call (e.g. calculate_bmi appears in the SSE trace AND in blobs).
        all_tool_calls = _dedup_tool_calls(all_tool_calls)

        # Strip arguments from actual calls for tools whose expected has no arguments.
        # This makes tool-name-only matching work without breaking strict arg checks.
        expected_tc = turn.get("expected_tool_calls") or []
        if expected_tc:
            all_tool_calls = _strip_args_for_no_arg_expected(
                all_tool_calls, expected_tc
            )

        if all_tool_calls:
            # Each tool call in its own sequence so partial matching works correctly.
            # [[tc1], [tc2], ...] lets tool_eval find a single expected tool among many.
            turn["tool_calls"] = [[tc] for tc in all_tool_calls]
        if result["contexts"]:
            turn["contexts"] = result["contexts"]
        else:
            # No tool results — ragas:faithfulness requires at least 1 context item.
            # Remove it from turn_metrics so lightspeed-eval doesn't error.
            if "ragas:faithfulness" in turn.get("turn_metrics", []):
                turn["turn_metrics"] = [
                    m for m in turn["turn_metrics"] if m != "ragas:faithfulness"
                ]

        log.info(
            "[%s] turn=%s done mode=%s tool_calls=%d",
            group_id,
            turn_id,
            hint,
            len(all_tool_calls),
        )

    return group


def _populate_dataset(  # pragma: no cover
    eval_data: list[dict[str, Any]],
    agent_url: str,
    auth_token: str | None,
    timeout: int,
    max_workers: int = 1,
) -> list[dict[str, Any]]:
    """Run conversation groups in parallel and fill in response/tool/context."""
    groups = copy.deepcopy(eval_data)
    results: dict[int, dict[str, Any]] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_populate_group, group, agent_url, auth_token, timeout): i
            for i, group in enumerate(groups)
        }
        for future in as_completed(futures):
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception as exc:
                log.error("group[%d] unhandled (%s)", idx, type(exc).__name__)
                results[idx] = groups[idx]

    return [results[i] for i in range(len(groups))]


# ── Lightspeed invocation ────────────────────────────────────────────────────


def _find_lightspeed_cmd() -> list[str] | None:
    """Return the command to invoke lightspeed-eval, or None if not found."""
    import shutil  # noqa: PLC0415 — stdlib, no install needed

    venv_bin = Path(sys.executable).parent
    for name in ["lightspeed-eval", "lightspeed_eval"]:
        candidate = venv_bin / name
        if candidate.exists():
            return [str(candidate)]
        found = shutil.which(name)
        if found:
            return [found]
    return None


def _subprocess_env() -> tuple[dict[str, str], list[Path]]:
    """Return env overrides and temp files needed by the lightspeed-eval subprocess.

    Writes GOOGLE_APPLICATION_CREDENTIALS_CONTENT to a temp file so that
    Vertex AI / ADC works without a GOOGLE_API_KEY.
    """
    extra_env: dict[str, str] = {}
    tmp_files: list[Path] = []

    # Google / Vertex AI service-account credentials
    content = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_CONTENT")
    if content and not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        try:
            tf = tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", prefix="gcp_sa_", delete=False
            )
            tf.write(content)
            tf.flush()
            tf.close()
            extra_env["GOOGLE_APPLICATION_CREDENTIALS"] = tf.name
            tmp_files.append(Path(tf.name))
            # Extract project_id so LiteLLM can route vertex_ai requests correctly
            try:
                sa = json.loads(content)
                project_id = sa.get("project_id", "")
                if project_id:
                    extra_env.setdefault("GOOGLE_CLOUD_PROJECT", project_id)
                    extra_env.setdefault("VERTEXAI_PROJECT", project_id)
                    extra_env.setdefault("VERTEXAI_LOCATION", "us-central1")
            except json.JSONDecodeError as exc:
                log.warning("could not parse GCP service account JSON: %s", exc)
        except Exception as exc:
            log.warning("could not write GCP credentials temp file: %s", exc)

    return extra_env, tmp_files


def _run_lightspeed(
    system_path: Path,
    populated_yaml: Path,
    output_dir: Path,
    lightspeed_cmd: list[str],
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    full_cmd = lightspeed_cmd + [
        "--system-config",
        str(system_path),
        "--eval-data",
        str(populated_yaml),
        "--output-dir",
        str(output_dir),
    ]
    log.info("invoking lightspeed-eval scorer")

    extra_env, tmp_files = _subprocess_env()
    env = {**os.environ, **extra_env} if extra_env else None
    try:
        return subprocess.run(full_cmd, check=False, env=env).returncode
    finally:
        for p in tmp_files:
            p.unlink(missing_ok=True)


# ── CLI ──────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Live eval: call agent → collect data → score with Lightspeed."
    )
    p.add_argument(
        "--agent-url",
        default=os.environ.get("AGENT_HOST", DEFAULT_AGENT_URL),
        help="Agent base URL (default: $AGENT_HOST or http://localhost:5002)",
    )
    p.add_argument(
        "--eval-data",
        type=Path,
        nargs="+",
        default=[DEFAULT_EVAL_DATA],
        metavar="FILE",
        help="One or more eval-data YAML files (merged in order)",
    )
    p.add_argument("--system", type=Path, default=DEFAULT_SYSTEM)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument(
        "--auth-token",
        default=os.environ.get("AGENT_AUTH_TOKEN"),
        help="Bearer token (or set AGENT_AUTH_TOKEN env var)",
    )
    p.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Per-turn timeout in seconds (increase for parallel runs or HITL-heavy evals)",
    )
    return p.parse_args()


def main() -> None:
    """Entry point for the standalone eval runner."""
    args = _parse_args()

    for path in args.eval_data:
        if not path.exists():
            log.error("eval data not found: %s", path)
            sys.exit(1)
    if not args.system.exists():
        log.error("system config not found: %s", args.system)
        sys.exit(1)

    if (
        not os.environ.get("GOOGLE_API_KEY")
        and not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_CONTENT")
        and not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    ):
        log.warning("no Google credentials found — scoring may fail")

    lightspeed_cmd = _find_lightspeed_cmd()
    if lightspeed_cmd is None:
        log.error(
            "lightspeed-eval not found. Install: "
            'pip install "lightspeed-evaluation @ git+https://github.com/lightspeed-core/lightspeed-evaluation.git@v0.7.0"'
        )
        sys.exit(1)

    log.info(
        "starting eval agent=%s eval_files=%d system=%s output=%s",
        args.agent_url,
        len(args.eval_data),
        args.system.name,
        args.output_dir,
    )

    eval_data: list[dict[str, Any]] = []
    for path in args.eval_data:
        eval_data.extend(yaml.safe_load(path.read_text()))

    log.info("collecting live agent responses for %d group(s)", len(eval_data))
    populated = _populate_dataset(
        eval_data,
        agent_url=args.agent_url,
        auth_token=args.auth_token,
        timeout=args.timeout,
        max_workers=10,
    )

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", prefix="lightspeed_live_", delete=False
    ) as tmp:
        yaml.dump(populated, tmp, allow_unicode=True, default_flow_style=False)
        tmp_path = Path(tmp.name)

    try:
        log.info("running lightspeed-eval scorer")
        exit_code = _run_lightspeed(
            args.system, tmp_path, args.output_dir, lightspeed_cmd
        )
    finally:
        tmp_path.unlink(missing_ok=True)

    if exit_code != 0:
        log.error("eval FAILED — lightspeed-eval exited %d", exit_code)
    else:
        log.info("eval PASSED — all metric thresholds met")

    sys.exit(exit_code)


if __name__ == "__main__":  # pragma: no cover
    main()
