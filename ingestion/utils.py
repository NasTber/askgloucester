"""Shared helpers for the ingestion stages."""

from __future__ import annotations


def classify_meeting_category(title: str) -> str:
    """Classify a meeting into a category from its listing/file title.

    Every School Committee document (AMID 113) is tagged ``School Committee``
    regardless of whether it is a full committee meeting, a subcommittee, or a
    negotiations session. The distinction lives only in the title, so derive it
    here:

        - "negotiat" anywhere in the title -> "negotiations"
        - "subcommittee" anywhere in the title -> "subcommittee"
        - otherwise -> "full_committee"

    Matching is case-insensitive. An empty or missing title falls back to
    "full_committee" (the default, most common meeting type).
    """
    text = (title or "").lower()
    if "negotiat" in text:
        return "negotiations"
    if "subcommittee" in text:
        return "subcommittee"
    return "full_committee"
