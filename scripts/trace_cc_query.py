"""Read-only repro/trace harness for the failing June City Council query.

Reproduces ONE agent invocation for
    question = "what was on the june city council agenda", history = []
and prints, in order, every tool call (name + full arguments), every tool
result (truncated, with date strings surfaced), and the final answer text.

It does NOT modify api/. It imports the agent's existing internals
(``_build_agent`` / ``_to_messages`` / ``_CITATION_STATE``) and re-runs the same
``.invoke`` that ``api.agent.ask`` runs, then walks ``result["messages"]`` —
which already contains the AIMessage.tool_calls and ToolMessage results, so no
logging needs to be added inside the agent.

No index writes, no pipeline, no reindex. Read-only.
"""

import json
import re

from langchain_core.messages import HumanMessage

# Import the agent's own internals UNMODIFIED. _build_agent compiles the same
# LangGraph agent ask() uses; _CITATION_STATE is the contextvar doc_search reads
# (it MUST be seeded or the tool raises LookupError); _to_messages rebuilds the
# history (empty here, but used for fidelity with ask()).
from api.agent import _build_agent, _to_messages, _CITATION_STATE

QUESTION = "what was on the june city council agenda"
HISTORY: list[dict] = []

DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}|[A-Z][a-z]+ \d{1,2},? \d{4}")


def _truncate_with_dates(text: str, limit: int = 300) -> str:
    """First ~`limit` chars, plus every date-like string found in the FULL text.

    Truncating from the front would hide dates on later source headers, so we
    additionally surface all YYYY-MM-DD / 'Month D, YYYY' strings from the whole
    result — exactly the values needed to see which meeting dates were retrieved.
    """
    head = text[:limit]
    suffix = "…" if len(text) > limit else ""
    dates = list(dict.fromkeys(DATE_RE.findall(text)))  # de-dup, keep order
    dates_line = ("\n    dates present in full result: " + ", ".join(dates)) if dates else ""
    return f"{head}{suffix}{dates_line}"


def main() -> None:
    # Seed the per-request citation state EXACTLY as ask() does, so doc_search's
    # _CITATION_STATE.get() succeeds and global [n] numbering works.
    _CITATION_STATE.set({"pairs": [], "next_n": 1})

    messages = _to_messages(HISTORY) + [HumanMessage(QUESTION)]

    print("=" * 80)
    print(f"QUESTION: {QUESTION!r}")
    print(f"HISTORY:  {HISTORY!r}")
    print("=" * 80)

    # Same invoke ask() performs; result["messages"] is the full transcript,
    # including AIMessage.tool_calls and ToolMessage results.
    result = _build_agent().invoke({"messages": messages})
    transcript = result["messages"]

    step = 0
    for msg in transcript:
        # An AIMessage carrying tool_calls = the agent's decision(s) to call a tool.
        tool_calls = getattr(msg, "tool_calls", None) or []
        for tc in tool_calls:
            step += 1
            name = tc.get("name")
            args = tc.get("args", {})
            print(f"\n--- step {step}: TOOL CALL -> {name}")
            print("    arguments:")
            print("    " + json.dumps(args, indent=2, default=str).replace("\n", "\n    "))

        # A ToolMessage = the result returned from a tool call.
        if msg.__class__.__name__ == "ToolMessage" or getattr(msg, "type", None) == "tool":
            step += 1
            name = getattr(msg, "name", "?")
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            print(f"\n--- step {step}: TOOL RESULT <- {name}")
            print("    " + _truncate_with_dates(content).replace("\n", "\n    "))

    # Final answer = content of the last message (an AIMessage with no tool_calls).
    last = transcript[-1]
    answer = last.content if isinstance(last.content, str) else str(last.content)
    print("\n" + "=" * 80)
    print("FINAL ANSWER:")
    print("=" * 80)
    print(answer)


if __name__ == "__main__":
    main()
