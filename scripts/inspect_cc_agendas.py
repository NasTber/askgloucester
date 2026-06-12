"""Read-only diagnostic: do City Council agenda chunks for June 2026 exist?

Inspects the search index to decide whether a declined June City Council
agenda query is an ingest, scope, or retrieval problem. Does NOT reindex,
call ensure_index, or write to the index — read-only, prints to stdout.
"""

import os
from collections import Counter, defaultdict

import dotenv
from azure.identity import DefaultAzureCredential
from azure.search.documents import SearchClient

dotenv.load_dotenv()

ENDPOINT = os.environ["AZURE_SEARCH_ENDPOINT"]
INDEX = os.environ.get("AZURE_SEARCH_INDEX_NAME", "gloucester-documents")

client = SearchClient(ENDPOINT, INDEX, credential=DefaultAzureCredential())


# --- a. Sanity pass: confirm exact casing of field values --------------------
# Don't assume 'City Council' / 'agenda' — sample the index and print the
# distinct values actually present so the filter below uses real casing.
print("=" * 80)
print("a. Distinct field values in a sample (confirm exact casing)")
print("=" * 80)

sample = client.search(
    search_text="*",
    select=["meeting_body", "document_type"],
    top=1000,
)

meeting_bodies = Counter()
document_types = Counter()
for doc in sample:
    meeting_bodies[doc.get("meeting_body")] += 1
    document_types[doc.get("document_type")] += 1

print(f"\ndistinct meeting_body values (from sample of {sum(meeting_bodies.values())} chunks):")
for value, count in sorted(meeting_bodies.items(), key=lambda kv: (kv[0] or "")):
    print(f"  {repr(value):<40} {count:>6}")

print(f"\ndistinct document_type values:")
for value, count in sorted(document_types.items(), key=lambda kv: (kv[0] or "")):
    print(f"  {repr(value):<40} {count:>6}")


# --- b. City Council agendas: total count + per-date chunk counts ------------
# Uses the casing the task confirmed: 'City Council' / 'agenda'. If part (a)
# shows different casing, this filter is what to adjust.
CC_FILTER = "meeting_body eq 'City Council' and document_type eq 'agenda'"

print("\n" + "=" * 80)
print(f"b. City Council agendas — filter: {CC_FILTER}")
print("=" * 80)

# Page through all matches (top caps a single page; loop until drained).
date_counts = Counter()
date_to_doc = {}  # document_date -> a representative doc (for part d)
total = 0
PAGE = 1000
skip = 0
while True:
    page = client.search(
        search_text="*",
        filter=CC_FILTER,
        select=["document_date", "content"],
        top=PAGE,
        skip=skip,
        order_by=["document_date asc"],
    )
    rows = list(page)
    if not rows:
        break
    for doc in rows:
        d = doc.get("document_date")
        date_counts[d] += 1
        date_to_doc.setdefault(d, doc)
        total += 1
    skip += len(rows)
    if len(rows) < PAGE:
        break

print(f"\nTotal City Council agenda chunks: {total}")
print(f"\n{'document_date':<16} {'chunks':>6}")
print("-" * 24)
for date in sorted(date_counts, key=lambda d: (d or "")):
    print(f"{date or '?':<16} {date_counts[date]:>6}")


# --- c. June 2026 presence + freshness ---------------------------------------
print("\n" + "=" * 80)
print("c. June 2026 presence and data freshness")
print("=" * 80)

# document_date is an ISO string ('YYYY-MM-DD' or full timestamp); a string
# prefix match on '2026-06' is sufficient to flag June 2026.
june_dates = sorted(d for d in date_counts if d and str(d).startswith("2026-06"))
if june_dates:
    print("\nJune 2026 City Council agenda dates PRESENT:")
    for d in june_dates:
        print(f"  {d}  ({date_counts[d]} chunks)")
else:
    print("\nNO City Council agenda chunks dated June 2026 (no document_date starts with '2026-06').")

real_dates = [d for d in date_counts if d]
if real_dates:
    max_date = max(real_dates)
    print(f"\nMax (most recent) City Council agenda document_date: {max_date}")
else:
    print("\nNo non-null document_date values among City Council agendas.")


# --- d. Representative content prefix (verify metadata prefix on CC chunks) ---
print("\n" + "=" * 80)
print("d. Representative content prefix (first ~250 chars)")
print("=" * 80)

rep_doc = None
if june_dates:
    rep_doc = date_to_doc.get(june_dates[0])
    label = f"June chunk ({june_dates[0]})"
elif real_dates:
    rep_doc = date_to_doc.get(max(real_dates))
    label = f"most-recent chunk ({max(real_dates)})"
else:
    label = "n/a"

if rep_doc is not None:
    content = rep_doc.get("content") or ""
    print(f"\nSource: {label}")
    print(f"content[:250]:\n{content[:250]!r}")
else:
    print("\nNo City Council agenda chunk available to sample.")
