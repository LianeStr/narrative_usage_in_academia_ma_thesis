"""
Enriches papers with a DOI using the OpenAlex API.
Uses batch DOI lookup (up to 100 DOIs per request, URL-length aware).

Usage:
    uv run enrich_doi.py
"""

import config
import keys

import json
import time
from pathlib import Path
from datetime import datetime, timezone

import requests

# ── Config ────────────────────────────────────────────────────────────────────

PROCESSED_DIR  = Path(config.PROCESSED_DATA_PATH)
ENRICHED_DIR   = Path(config.ENRICHED_DATA_PATH) / "doi"
BATCH_SIZE     = 100_000   # papers per output file
DOI_BATCH      = 100       # DOIs per OpenAlex API request (their max)
SLEEP          = 0.1       # seconds between API requests (polite pool: 10 req/s)
OPENALEX_EMAIL = None                  # not needed — using API key instead
OPENALEX_API_KEY = keys.API_KEY_OPENALEX

MAX_URL_BYTES  = 3500      # conservative limit below OpenAlex's 4094-byte hard cap

# Keys written as None when OpenAlex returns no match for a paper
EMPTY_OA_KEYS = [
    "oa_id", "oa_title", "oa_publication_year", "oa_language",
    "oa_type", "oa_is_retracted", "oa_cited_by_count",
    "oa_citation_normalized_percentile", "oa_authors",
    "oa_source_display_name", "oa_source_issn", "oa_source_issn_l",
    "oa_source_type", "oa_source_host_organization_name", "oa_pdf_url",
    "oa_topics", "oa_keywords", "oa_concepts",
    "oa_has_pdf", "oa_has_grobid_xml", "oa_content_urls",
    "abstract", "abstract_source",
]


# ── Metadata extraction ───────────────────────────────────────────────────────

def reconstruct_abstract(abstract_inverted_index: dict | None) -> str | None:
    """Reconstruct plain-text abstract from OpenAlex inverted index format."""
    if not abstract_inverted_index:
        return None
    try:
        positions: list[tuple[int, str]] = []
        for word, indices in abstract_inverted_index.items():
            for idx in indices:
                positions.append((idx, word))
        return " ".join(word for _, word in sorted(positions))
    except Exception:
        return None


def extract_oa_fields(work: dict, semscho_abstract: str | None) -> dict:
    """
    Extract all requested fields from an OpenAlex work object.
    All keys are prefixed with oa_ except abstract/abstract_source which
    merge SemanticScholar and OpenAlex sources into a single field.
    """
    # Authors: ordered list of display names only
    authors = [
        (a.get("author") or {}).get("display_name")
        for a in sorted(
            work.get("authorships") or [],
            key=lambda a: {"first": 0, "middle": 1, "last": 2}.get(
                a.get("author_position", "last"), 1
            ),
        )
        if (a.get("author") or {}).get("display_name")
    ]

    # Primary location → source fields + pdf_url
    primary  = work.get("primary_location") or {}
    source   = primary.get("source") or {}
    pdf_url  = primary.get("pdf_url")

    # Topics: id + display_name + score + subfield/field/domain display names
    topics = [
        {
            "id":           t.get("id"),
            "display_name": t.get("display_name"),
            "score":        t.get("score"),
            "subfield":     (t.get("subfield") or {}).get("display_name"),
            "field":        (t.get("field")    or {}).get("display_name"),
            "domain":       (t.get("domain")   or {}).get("display_name"),
        }
        for t in (work.get("topics") or [])
    ]

    # Keywords: display_name + score
    keywords = [
        {
            "display_name": k.get("display_name"),
            "score":        k.get("score"),
        }
        for k in (work.get("keywords") or [])
    ]

    # Concepts: display_name + score, only where score != 0
    concepts = [
        {
            "display_name": c.get("display_name"),
            "score":        c.get("score"),
        }
        for c in (work.get("concepts") or [])
        if (c.get("score") or 0) != 0
    ]

    # has_content fields
    has_content = work.get("has_content") or {}

    # Abstract: prefer SemanticScholar, fall back to reconstructed OpenAlex index
    oa_abstract = reconstruct_abstract(work.get("abstract_inverted_index"))
    if semscho_abstract:
        abstract_out    = semscho_abstract
        abstract_source = "semantic_scholar"
    elif oa_abstract:
        abstract_out    = oa_abstract
        abstract_source = "openalex"
    else:
        abstract_out    = None
        abstract_source = None

    # citation_normalized_percentile: extract value only
    cnp = work.get("citation_normalized_percentile") or {}

    return {
        "oa_id":                              work.get("id"),
        "oa_title":                           work.get("title"),
        "oa_publication_year":                work.get("publication_year"),
        "oa_language":                        work.get("language"),
        "oa_type":                            work.get("type"),
        "oa_is_retracted":                    work.get("is_retracted"),
        "oa_cited_by_count":                  work.get("cited_by_count"),
        "oa_citation_normalized_percentile":  cnp.get("value"),
        "oa_authors":                         authors,
        "oa_source_display_name":             source.get("display_name"),
        "oa_source_issn":                     source.get("issn"),
        "oa_source_issn_l":                   source.get("issn_l"),
        "oa_source_type":                     source.get("type"),
        "oa_source_host_organization_name":   source.get("host_organization_name"),
        "oa_pdf_url":                         pdf_url,
        "oa_topics":                          topics,
        "oa_keywords":                        keywords,
        "oa_concepts":                        concepts,
        "oa_has_pdf":                         has_content.get("pdf"),
        "oa_has_grobid_xml":                  has_content.get("grobid_xml"),
        "oa_content_urls":                    work.get("content_urls"),
        "abstract":                           abstract_out,
        "abstract_source":                    abstract_source,
    }


# ── OpenAlex batch lookup ─────────────────────────────────────────────────────

def _clean_doi(doi: str) -> str:
    """Normalise a DOI to the bare form OpenAlex expects (no URL prefix, stripped)."""
    return doi.strip().lower().removeprefix("https://doi.org/").removeprefix("http://doi.org/").strip()


def _fetch_batch(dois: list[str]) -> dict[str, dict]:
    """
    Execute a single OpenAlex batch request for a list of DOIs.
    Returns a dict mapping normalised DOI → raw work object for all hits.
    """
    cleaned = [_clean_doi(d) for d in dois]
    # Drop anything that looks malformed (must start with 10.)
    valid = [d for d in cleaned if d.startswith("10.")]
    dropped = len(cleaned) - len(valid)
    if dropped:
        print(f"  Skipped {dropped} malformed DOI(s) in this batch")

    if not valid:
        return {}

    try:
        response = requests.get(
            "https://api.openalex.org/works",
            params={
                "filter":   f"doi:{'|'.join(valid)}",
                "per_page": len(valid),
                "api_key":  OPENALEX_API_KEY,
            },
            timeout=15,
        )
        if response.status_code != 200:
            print(f"  HTTP {response.status_code} for batch of {len(valid)} DOIs — {response.text[:200]}")
            return {}

        return {
            _clean_doi(r.get("doi") or ""): r
            for r in response.json().get("results", [])
            if r.get("doi")
        }

    except requests.RequestException as e:
        print(f"  Request error: {e}")
        return {}


def fetch_openalex_works(dois: list[str]) -> dict[str, dict]:
    """
    Look up a list of DOIs on OpenAlex, splitting into URL-safe sub-batches.
    Returns a dict mapping normalised DOI → raw work object for all hits.
    """
    # Base URL length determines how many DOI bytes we can fit per request
    base_len = len(
        f"https://api.openalex.org/works?filter=doi:"
        f"&per_page=100&api_key={OPENALEX_API_KEY}"
    )

    results   = {}
    sub_batch: list[str] = []

    for doi in [_clean_doi(d) for d in dois]:
        added_len          = len(doi) + (1 if sub_batch else 0)  # +1 for "|"
        current_filter_len = sum(len(d) + 1 for d in sub_batch)

        if sub_batch and base_len + current_filter_len + added_len > MAX_URL_BYTES:
            results.update(_fetch_batch(sub_batch))
            time.sleep(SLEEP)
            sub_batch = []

        sub_batch.append(doi)

    if sub_batch:
        results.update(_fetch_batch(sub_batch))

    return results


# ── I/O helpers ───────────────────────────────────────────────────────────────

def iter_processed_papers(processed_dir: Path):
    """Yield parsed paper dicts from all processed batch files, in order."""
    files = sorted(processed_dir.glob("search_results_batch_*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No batch files found in {processed_dir}")
    for path in files:
        print(f"Reading {path.name}...")
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def load_done_ids(enriched_dir: Path) -> set[str]:
    """Return set of paperIds already written to the output directory."""
    done = set()
    for path in sorted(enriched_dir.glob("enriched_batch_*.jsonl")):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    pid = json.loads(line).get("paperId")
                    if pid:
                        done.add(pid)
                except json.JSONDecodeError:
                    continue
    print(f"Resuming — {len(done):,} papers already enriched, skipping them.")
    return done


def load_failed_papers(enriched_dir: Path):
    """Return papers that previously failed to match OpenAlex."""
    failed = []

    for path in sorted(enriched_dir.glob("enriched_batch_*.jsonl")):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    paper = json.loads(line)
                    if paper.get("doi") and paper.get("oa_id") is None:
                        failed.append(paper)
                except json.JSONDecodeError:
                    continue

    return failed


def write_log(log_path: Path, stats: dict):
    record = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), **stats}
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    print(f"Run logged to {log_path}")


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_enrichment(
    processed_dir:     Path = PROCESSED_DIR,
    enriched_dir:      Path = ENRICHED_DIR,
    output_batch_size: int  = BATCH_SIZE,
    doi_batch_size:    int  = DOI_BATCH,
):
    enriched_dir.mkdir(parents=True, exist_ok=True)
    done_ids = load_done_ids(enriched_dir)

    print("Loading papers with DOI...")
    pending = [
        paper for paper in iter_processed_papers(processed_dir)
        if paper.get("doi") and paper.get("paperId") not in done_ids
    ]
    print(f"  {len(pending):,} papers to enrich.")

    if not pending:
        failed = load_failed_papers(enriched_dir)

        if not failed:
            print("Nothing to do.")
            return

        print(
            f"No new papers found. Retrying {len(failed):,} "
            "previously unmatched papers..."
        )

        pending = failed
        doi_batch_size = 1

    # Open or resume output file
    existing = sorted(enriched_dir.glob("enriched_batch_*.jsonl"))
    if existing:
        batch_index    = len(existing)
        last_path      = existing[-1]
        count_in_batch = sum(1 for l in open(last_path) if l.strip())
        current_file   = open(last_path, "a", encoding="utf-8")
        print(f"Appending to {last_path.name} ({count_in_batch:,} lines)")
    else:
        batch_index    = 1
        count_in_batch = 0
        path           = enriched_dir / f"enriched_batch_{batch_index}.jsonl"
        current_file   = open(path, "w", encoding="utf-8")
        print(f"Opening {path.name}")

    def open_next_batch():
        nonlocal batch_index, current_file, count_in_batch
        current_file.close()
        batch_index   += 1
        count_in_batch = 0
        path = enriched_dir / f"enriched_batch_{batch_index}.jsonl"
        print(f"Opening {path.name}")
        current_file = open(path, "w", encoding="utf-8")

    total_hit       = 0
    total_not_found = 0

    try:
        for i in range(0, len(pending), doi_batch_size):
            batch_papers = pending[i : i + doi_batch_size]
            dois         = [p["doi"] for p in batch_papers]
            oa_map       = fetch_openalex_works(dois)

            for paper in batch_papers:
                doi_key = _clean_doi(paper["doi"])
                work    = oa_map.get(doi_key)

                if work:
                    paper.update(extract_oa_fields(work, paper.get("abstract")))
                    total_hit += 1
                else:
                    paper.update({k: None for k in EMPTY_OA_KEYS})
                    total_not_found += 1

                current_file.write(json.dumps(paper) + "\n")
                count_in_batch += 1

                if count_in_batch >= output_batch_size:
                    print("Output batch full — rotating...")
                    open_next_batch()

            n_done = min(i + doi_batch_size, len(pending))
            if (i // doi_batch_size) % 50 == 0:
                print(
                    f"  Progress: {n_done:,}/{len(pending):,} | "
                    f"hit={total_hit:,} not_found={total_not_found:,}"
                )

    finally:
        current_file.close()

    stats = {
        "processed_dir":   str(processed_dir),
        "enriched_dir":    str(enriched_dir),
        "total_pending":   len(pending),
        "total_hit":       total_hit,
        "total_not_found": total_not_found,
        "batches_written": batch_index,
    }

    print(
        f"\nDone. {len(pending):,} papers processed.\n"
        f"  Matched in OpenAlex: {total_hit:,}\n"
        f"  Not found:           {total_not_found:,}"
    )
    write_log(enriched_dir / "enrich_doi_log.jsonl", stats)


if __name__ == "__main__":
    run_enrichment()