"""Read-only retrieval probe to validate the fix direction for the June CC query.

Calls ``doc_search`` DIRECTLY (bypassing the agent's argument selection) three
ways, to isolate whether dropping the fabricated ``target_date`` recovers the
June City Council agenda chunks:

  A (decisive): a focused query + body, NO target_date  -> does plain hybrid
                search surface June without an exact-date filter?
  B (control):  a bare "agenda" query + body, NO date   -> is a weak query the
                problem rather than the date filter?
  C (control):  exact target_date on a REAL meeting day  -> confirms eq-match
                works when the date actually exists (2026-06-09).

Seeds _CITATION_STATE exactly as scripts/trace_cc_query.py does (doc_search
reads it). Does NOT modify api/, does NOT write to the index. Read-only.
"""

# Reuse the agent's tool and citation contextvar UNMODIFIED, plus the
# date-extraction helper from the trace harness.
from api.agent import doc_search, _CITATION_STATE
from scripts.trace_cc_query import DATE_RE

VARIANTS = [
    ("A (decisive): focused query, body, no target_date",
     {"query": "June City Council agenda", "body": "City Council"}),
    ("B (control): bare query, body, no target_date",
     {"query": "agenda", "body": "City Council"}),
    ("C (control): exact target_date on a real meeting day",
     {"query": "City Council agenda", "body": "City Council", "target_date": "2026-06-09"}),
]


def _dates_in(content: str) -> list[str]:
    """De-duplicated, sorted date strings found in the tool's content block."""
    return sorted(set(DATE_RE.findall(content)))


def main() -> None:
    for label, args in VARIANTS:
        # Fresh per-call citation state, same seed shape ask()/trace use.
        _CITATION_STATE.set({"pairs": [], "next_n": 1})

        # Call the raw underlying function (bypass the agent's arg selection).
        # response_format is content_and_artifact, so .func returns
        # (numbered_content_str, chunks_list).
        content, chunks = doc_search.func(**args)

        print("=" * 80)
        print(label)
        print(f"  args:    {args}")
        print(f"  results: {len(chunks)}")
        print(f"  dates:   {_dates_in(content) or '(none)'}")
        if not chunks:
            # Surface the deterministic empty/decline signal verbatim.
            print(f"  signal:  {content!r}")
        print()


if __name__ == "__main__":
    main()
