"""
Enrich processed Semantic Scholar paper records with metadata from OpenAlex.

This module reads processed paper records from JSONL batch files, looks up papers
with a DOI using the OpenAlex API, and enriches matching records with metadata
provided by OpenAlex. Abstracts are sourced preferentially from the existing
Semantic Scholar record and fall back to the abstract reconstructed from the
OpenAlex inverted-index representation.

The pipeline is designed for large datasets and processes records incrementally
in DOI batches rather than issuing one API request per paper. OpenAlex requests
are further split into URL-safe sub-batches to remain below the API's URL-length
limit. Enriched records are written to sequentially numbered JSONL batch files.

For each input record, the following operations are performed in order:

1. Read processed paper records from JSONL batch files.
2. Select records that contain a DOI and have not already been enriched.
3. Normalize DOI strings by removing URL prefixes, surrounding whitespace,
   and converting them to lowercase.
4. Group DOIs into API batches and split batches further when necessary to
   remain below the configured URL-length limit.
5. Query the OpenAlex `works` endpoint for each DOI batch.
6. Match returned OpenAlex work objects to input papers using their normalized DOI.
7. Extract the configured OpenAlex metadata fields from matching work objects.
8. Prefer the existing Semantic Scholar abstract when available and otherwise
   reconstruct the abstract from OpenAlex's inverted-index representation.
9. Add `None` values for all OpenAlex-specific fields when no OpenAlex match
   is found.
10. Write the enriched paper record to the current JSONL output batch.
11. Rotate the output file when the configured maximum number of records is reached.
12. Record summary statistics and the processing configuration in
    `enrich_doi_log.jsonl`.

Previously unmatched papers are handled specially. If a run has no new papers
to process, records from previous output batches that contain a DOI but have
no `oa_id` are retried individually against OpenAlex. This allows temporary
API failures or previously unavailable records to be retried without reprocessing
the complete dataset.

Input
--------
Processed input files are read from `processed_dir`.
By default, this is `config.PROCESSED_DATA_PATH`.

Input files must follow the naming convention:
search_results_batch_*.jsonl

Each non-empty line is expected to contain one processed paper record in JSON
format. Records are expected to contain at least `paperId` and `doi` for
OpenAlex enrichment to take place.

Output
--------
Enriched records are written to `enriched_dir` using sequentially numbered
JSONL files:

```
enriched_batch_1.jsonl
enriched_batch_2.jsonl
...
```

The maximum number of records per output file is controlled by
`output_batch_size`.

Each successfully matched record receives the configured OpenAlex metadata,
including (not the complete list):

`oa_id` (OpenAlex work identifier)

`oa_language` (Language recorded by OpenAlex)

`oa_pdf_url`
(PDF URL from the OpenAlex primary location, when available)

`oa_topics`
(OpenAlex topics including topic identifiers, names, scores, and
subfield/field/domain names)

`oa_keywords` (OpenAlex keywords and their scores)

`oa_concepts` (OpenAlex concepts and their scores, excluding concepts with a zero score)

`oa_has_pdf` (Whether OpenAlex reports PDF content for the work)

`abstract`
(Paper abstract. The existing Semantic Scholar abstract is preferred;
otherwise the abstract is reconstructed from OpenAlex's inverted index)

`abstract_source`
(Indicates the source of the selected abstract: `semantic_scholar`,
`openalex`, or `None`.)

Records for which OpenAlex returns no matching work are still written to the
output. All OpenAlex-specific fields listed in `EMPTY_OA_KEYS` are set to
`None` so that matched and unmatched records have a consistent schema.

Duplicate `paperId` values are not reprocessed when they are already present
in the enriched output directory.

Configuration
----------
The following module-level constants define the default enrichment configuration:

`PROCESSED_DIR`
Default input directory obtained from `config.PROCESSED_DATA_PATH`.

`ENRICHED_DIR`
Default output directory obtained from `config.ENRICHED_DATA_PATH` with
the `doi` subdirectory appended.

`BATCH_SIZE`
Default maximum number of papers written to each output JSONL file.

`DOI_BATCH`
Default maximum number of DOIs submitted to an OpenAlex lookup batch.

`SLEEP`
Delay between OpenAlex API requests, used to limit request frequency.

`OPENALEX_API_KEY`
API key used to authenticate requests to OpenAlex. 
Retrieved from the imported `key.py`

`MAX_URL_BYTES`
Conservative maximum URL length used when constructing OpenAlex DOI
filter requests.

The `run_enrichment` function accepts the input directory, output directory,
output batch size, and DOI batch size as arguments. This allows the enrichment
pipeline to be configured without modifying the implementation.

OpenAlex API Requests
--------
OpenAlex DOI lookups use the `/works` endpoint with a pipe-separated DOI
filter. Although OpenAlex supports up to 100 DOIs per request, the actual
request size is also constrained by URL length. `fetch_openalex_works` therefore
constructs sub-batches dynamically and ensures that requests remain below
`MAX_URL_BYTES`.

Malformed DOI values that do not begin with `10.` are skipped before an API
request is made.

API requests use a timeout and handle HTTP and network errors without aborting
the complete enrichment run. A failed request produces no OpenAlex matches for
that batch, allowing the affected records to be written as unmatched and
potentially retried in a later run.

A short delay is inserted between API requests to limit the request rate.

Resuming and Retry Behavior
--------
The enrichment process is restartable.

Before processing new records, `load_done_ids` scans existing enriched output
files and collects their `paperId` values. Papers whose IDs are already
present are skipped, preventing duplicate enrichment after an interrupted or
repeated run.

If no new papers remain, `load_failed_papers` identifies previously written
records that contain a DOI but have no `oa_id`. These records are retried
individually rather than reprocessing the entire dataset.

This retry behavior is particularly useful for recovering from temporary
OpenAlex API errors or records that were not available during an earlier run.

Log
--------
A processing summary is appended to:

```
enriched_dir / "enrich_doi_log.jsonl"
```

Each log entry contains a UTC timestamp and the main statistics for the run.

The log is written as JSONL so that multiple enrichment runs can be recorded
in the same file and analyzed independently.

Data Processing Details
--------
DOIs are normalized before comparison and API lookup by:

1. stripping surrounding whitespace;
2. converting the DOI to lowercase;
3. removing `https://doi.org/` or `http://doi.org/` prefixes.

OpenAlex authors are reduced to display names and ordered according to their
`author_position` values.

The primary OpenAlex location is used to obtain source metadata and the PDF URL.

OpenAlex topics are reduced to their identifier, display name, score, and
hierarchical subfield, field, and domain names.

OpenAlex keywords and concepts are reduced to their display names and scores.
Concepts with a zero score are omitted.

OpenAlex's `abstract_inverted_index` is converted back into plain text by
sorting words according to their recorded token positions.

Reproducibility
--------
The enrichment configuration is defined by module-level constants and the
arguments passed to `run_enrichment`. Run statistics are persisted in
`enrich_doi_log.jsonl` so that individual enrichment runs can be audited.

The module can be executed with:

```
uv run enrich_doi.py
```

which runs `run_enrichment()` using the configured default paths and batch
sizes.
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
    """Append a enrichment run record to a JSONL log file."""
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