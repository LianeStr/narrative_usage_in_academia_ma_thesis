"""
Preprocess Semantic Scholar paper records for downstream analysis.

This module reads raw Semantic Scholar paper records from JSONL batch files,
cleans and restructures the records, removes duplicate papers, and applies a
series of quality and relevance filters.

The pipeline is designed for large datasets and processes records 
incrementally rather than loading all input records into memory. 
Accepted records are written in batches, while excluded records are 
written to separate files according to the reason for exclusion.

For each input record, the following operations are performed in order: 
1. Read the record from a raw JSONL batch file. 
2. Parse the record either as a Python literal or as JSON. 
3. Restructure selected metadata fields: 
    - extract the DOI from ``externalIds`` 
    or, as a fallback, from the ``openAccessPdf.disclaimer`` field; 
    - remove the redundant top-level ``url`` field; 
    - extract ``openAccessPdf.url`` into ``pdf_url``. 
4. Remove duplicate records based on ``paperId``. 
5. Detect the language of the combined title and abstract using ``langdetect``. 
6. Exclude records that are not detected as English. 
7. Exclude records for which no DOI can be extracted. 
8. Exclude records whose titles match one of the configured regular expression patterns. 
9. Exclude records whose abstracts contain one of the configured exclusion phrases. 
10. Write records that pass all filters to sequentially numbered JSONL batch files. 
11. Write excluded records to dedicated JSONL files according to the exclusion category. 
12. Record summary statistics and the preprocessing configuration in ``preprocess_log.jsonl``.


Input
----- 
Raw input files are read from ``raw_dir``. 
By default, this is ``config.RAW_DATA_PATH``. 

Input files must follow the naming convention:
    search_results_batch_*.jsonl 
    
Each non-empty line is expected to contain one paper record. 
Records are first parsed with ``ast.literal_eval`` to support Python-dictionary-style records 
and, if that fails, are parsed as JSON.

Output 
------ 
Accepted records are written to ``processed_dir`` using sequentially numbered
JSONL files:
    search_results_batch_1.jsonl 
    search_results_batch_2.jsonl 
    ... 

The maximum number of records per output file is controlled by ``batch_size``. 
Records excluded during preprocessing are written to ``filtered_out_dir``: 

``language_langdetect.jsonl`` 
    Records whose detected language is not ``en``. 
    The detected language is stored in ``_detected_lang``. 

``no_doi.jsonl`` 
    Records for which no DOI could be extracted. 

``ngram_exclusion.jsonl`` 
    Records excluded because their title or abstract matched a configured exclusion rule. 
    The matching rule is stored in ``_drop_reason``. 

Duplicate records are not written to any output file.

A processing summary is appended to:
    processed_dir / "preprocess_log.jsonl" 
Each log entry contains the UTC timestamp of the run, input and output directories, 
active exclusion rules, record counts, per-rule exclusion counts, and the number of output batches created.


Configuration 
------------- 
The following module-level constants define the default pipeline configuration: 
``RAW_DIR`` 
    Default raw-data directory obtained from ``config.RAW_DATA_PATH``.

``PROCESSED_DIR`` 
    Default output directory obtained from ``config.PROCESSED_DATA_PATH``. 

``FILTERED_OUT_DIR`` 
    Default directory for excluded records. 
    
``BATCH_SIZE`` 
    Default maximum number of records written to each processed batch. 

The ``run_preprocessing`` function accepts these values as arguments, 
making the pipeline configurable without modifying the implementation.


Reproducibility 
--------------- 
``langdetect`` is initialized with a fixed detector seed so that 
language detection is deterministic across runs, 
subject to the behavior of the installed ``langdetect`` version. 

The preprocessing log records the active title-pattern labels and abstract exclusion phrases, 
allowing the filtering configuration used for a run to be reconstructed from the log. 
The module can be executed directly with:
    uv run python src/ma_project/filter.py 
which runs ``run_preprocessing()`` using the configured default paths and batch size.
"""

import config

import ast
import json
import re
from pathlib import Path
from datetime import datetime, timezone

from langdetect import detect, DetectorFactory

# Seed for reproducibility
DetectorFactory.seed = 0


# ── Config ────────────────────────────────────────────────────────────────────

RAW_DIR         = Path(config.RAW_DATA_PATH)
PROCESSED_DIR   = Path(config.PROCESSED_DATA_PATH)
FILTERED_OUT_DIR = Path("data/filtered_out")
BATCH_SIZE      = 100_000

# ── Title exclusion: regex patterns ──────────────────────────────────────────
#
# Each entry is a (label, compiled_pattern) tuple.
# A title matching ANY pattern is excluded.
# The label is used in drop_counts and _drop_reason for diagnostics.

TITLE_EXCLUSION_PATTERNS: list[tuple[str, re.Pattern]] = [
    # "narrative <up-to-2-words> review(s)"  e.g. "narrative systematic review",
    # "narrative and systematic reviews", "narrative literature review"
    (
        "NarrRev",
        re.compile(r'\bnarrative(?:[-\s]+\w+){0,2}[-\s]+reviews?', re.IGNORECASE),
    ),
    # "narrative <up-to-2-words> summary/summaries"
    (
        "NarrSum",
        re.compile(r'\bnarrative(?:[-\s]+\w+){0,2}[-\s]+summar(?:y|ies)', re.IGNORECASE),
    ),
    # "narrative <up-to-2-words> synthesis/syntheses"
    (
        "NarrSyn",
        re.compile(r'\bnarrative(?:[-\s]+\w+){0,2}[-\s]+synthes(?:is|es)', re.IGNORECASE),
    ),
    (
        "NarrMeta",
        re.compile(r'\bnarrative meta[- ]analysis\b', re.IGNORECASE),
    ),
    # "narrative overview"
    (
        "NarrOverview",
        re.compile(r'\bnarrative overview\b', re.IGNORECASE),
    ),
]

# ── Abstract exclusion: single exact phrase ───────────────────────────────────
#
# (case-insensitive substring match).
ABSTRACT_EXCLUSION_PHRASES: list[str] = [
    "narrative review",
    "narrative synthesis",
    "narrative literature review",
    "narrative summary",
    "synthesized narratively",
    "narratively synthesized",
    "synthesised narratively",
    "narratively synthesised",
    "narrative overview",
    # check narrative n-grams up to down to a freq of 200

]

# Regex to extract a DOI from the externalIds field.
_DOI_RE = re.compile(
    r'https?://doi\.org/([^\s,\'">\]?#]+)'
    r'|(?<!\w)(10\.\d{4,}/[^\s,\'">\]?#]+)',
)


# ── DOI extraction ────────────────────────────────────────────────────────────

def extract_doi_from_external_ids(paper: dict) -> str | None:
    """
    Extract a DOI from the externalIds field (preferred),
    falling back to openAccessPdf.disclaimer if not found there.
    Returns the bare DOI string, or None.
    """
    external_ids = paper.get("externalIds")
    if isinstance(external_ids, dict):
        doi = external_ids.get("DOI") or external_ids.get("doi")
        if doi:
            return doi

    # Fallback: parse from openAccessPdf.disclaimer
    open_access = paper.get("openAccessPdf")
    if isinstance(open_access, dict):
        disclaimer = open_access.get("disclaimer")
        if disclaimer:
            match = _DOI_RE.search(disclaimer)
            if match:
                return match.group(1) or match.group(2)

    return None


def extract_pdf_url(paper: dict) -> str | None:
    """Unnest the url from openAccessPdf."""
    open_access = paper.get("openAccessPdf")
    if isinstance(open_access, dict):
        return open_access.get("url")
    return None


def restructure_paper(paper: dict) -> dict:
    """
    (a) Column restructuring:
      1. Replace externalIds with just the DOI string (field: 'doi').
      2. Remove top-level 'url' (redundant paperId prefix).
      3. Unnest openAccessPdf.url → 'pdf_url'; drop the openAccessPdf object.
    """
    out = dict(paper)

    # (a1) Extract DOI from externalIds, replace the whole field
    doi = extract_doi_from_external_ids(paper)
    out["doi"] = doi
    out.pop("externalIds", None)

    # (a2) Remove top-level url
    out.pop("url", None)

    # (a3) Unnest pdf_url, drop openAccessPdf
    out["pdf_url"] = extract_pdf_url(paper)
    out.pop("openAccessPdf", None)

    return out


# ── Filtering helpers ─────────────────────────────────────────────────────────

def title_exclusion_match(title: str) -> str | None:
    """
    Returns the label of the first matching title exclusion pattern,
    or None if the title should be kept.
    """
    for label, pattern in TITLE_EXCLUSION_PATTERNS:
        if pattern.search(title):
            return label
    return None


def abstract_exclusion_match(abstract: str) -> str | None:
    """
    Returns the first matching exclusion phrase (case-insensitive substring),
    or None if the abstract should be kept.
    """
    abstract_lower = abstract.lower()
    for phrase in ABSTRACT_EXCLUSION_PHRASES:
        if phrase.lower() in abstract_lower:
            return phrase
    return None


def detect_language(text: str) -> str | None:
    """Return ISO 639-1 language code, or None on failure."""
    try:
        return detect(text)
    except Exception:
        return None


# ── I/O helpers ───────────────────────────────────────────────────────────────

def iter_raw_papers(raw_dir: Path):
    """Yield parsed paper dicts from all raw batch files, in order."""
    files = sorted(raw_dir.glob("search_results_batch_*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No batch files found in {raw_dir}")

    for path in files:
        print(f"Reading {path.name}...")
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    # try to read the line as a python literal (e.g. python dict)
                    yield ast.literal_eval(line)
                except (ValueError, SyntaxError):
                    try:
                        # if it fails, read it as a json
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue


class BatchWriter:
    """
    Writes records to sequentially numbered JSONL batch files. 
    
    Records are written incrementally to avoid keeping the entire processed dataset in memory. 
    A new batch file is automatically opened whenever the configured batch size is reached. 
    Each output file contains one JSON record per line and is named using the configured prefix and a sequential batch number, 
    for example: 
        search_results_batch_1.jsonl 
        search_results_batch_2.jsonl 
        search_results_batch_3.jsonl 
        
    The class keeps track of the number of records written to the current batch as well as the total number of records written across all batches. 
    Args: 
        output_dir: Directory in which the batch files are created. 
        prefix: Prefix used when naming the batch files. 
        batch_size: Maximum number of records written to each batch file. 
        
    The writer should be closed with ``close()`` after writing is finished.
    """

    def __init__(self, output_dir: Path, prefix: str = "search_results_batch", batch_size: int = BATCH_SIZE):
        """Initialize the batch writer and create the output directory."""
        self.output_dir  = output_dir
        self.prefix      = prefix
        self.batch_size  = batch_size
        self._file       = None
        self._batch_idx  = 1
        self._count      = 0
        self.total       = 0
        output_dir.mkdir(parents=True, exist_ok=True)

    def _open_next(self):
        """Close the current batch file and open the next numbered file."""
        if self._file:
            self._file.close()
        path = self.output_dir / f"{self.prefix}_{self._batch_idx}.jsonl"
        print(f"  Opening {path}")
        self._file = open(path, "w", encoding="utf-8")
        self._batch_idx += 1
        self._count = 0

    def write(self, record: dict):
        """ Write one record to the current JSONL batch file. 
        
        A new batch file is opened automatically 
        when the current batch reaches the configured batch size. 
        """
        if self._file is None:
            self._open_next()
        self._file.write(json.dumps(record) + "\n")
        self._count += 1
        self.total  += 1
        if self._count >= self.batch_size:
            self._open_next()

    def close(self):
        """Close the currently open batch file, if one exists."""
        if self._file:
            self._file.close()
            self._file = None

    @property
    def batches_written(self):
        """Return the number of batch files created by this writer."""
        return self._batch_idx - 1


class SingleFileWriter:
    """
    Writes filtered-out records incrementally to a single JSONL file.
    
    ``SingleFileWriter`` is a helper for storing records that have been 
    excluded from the preprocessing pipeline. 
    Unlike ``BatchWriter``, which creates multiple output files after 
    reaching a configurable batch size, this class writes all records to 
    one specified file. 
    
    Each record is serialized as a single JSON object and written on its own line. 
    Records are written incrementally as they are received, 
    so the complete collection of records does not need to be held in memory. 
    
    The output file and its parent directory are created when the writer is initialized. 
    If a file already exists at the specified path, it is opened in write mode 
    and therefore overwritten. 
    
    Attributes: 
        total (int): Number of records successfully passed to ``write()`` by this writer instance. 
    
    Args: 
        path (Path): Path of the JSONL file to create. 
        Parent directories are created automatically if they do not already exist. 
    
    Notes:
        The supplied object is expected to be JSON-serializable. 

        The file should be closed explicitly with ``close()`` after writing is complete. 
        In the preprocessing pipeline, this is handled in the ``finally`` block of 
        ``run_preprocessing()`` so that output files are closed even if processing 
        terminates with an exception. 
        
        The ``total`` attribute counts calls to ``write()`` and is intended for reporting 
        preprocessing statistics. It does not independently verify that the resulting file 
        contains the expected number of records.
    """

    def __init__(self, path: Path):
        """Initialize the writer and open the target JSONL file for writing."""
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(path, "w", encoding="utf-8")
        self.total = 0

    def write(self, record: dict):
        """Serialize ``record`` as JSON and append it as one line to the file."""
        self._file.write(json.dumps(record) + "\n")
        self.total += 1

    def close(self):
        """Close the underlying output file."""
        self._file.close()


def write_log(log_path: Path, stats: dict):
    """Append a processing run record to a JSONL log file."""
    record = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), **stats}
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    print(f"Run logged to {log_path}")


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_preprocessing(
    raw_dir: Path              = RAW_DIR,
    processed_dir: Path        = PROCESSED_DIR,
    filtered_out_dir: Path     = FILTERED_OUT_DIR,
    batch_size: int            = BATCH_SIZE,
    title_patterns             = TITLE_EXCLUSION_PATTERNS,
    abstract_phrases: list[str] = ABSTRACT_EXCLUSION_PHRASES,
):
    processed_dir.mkdir(parents=True, exist_ok=True)
    filtered_out_dir.mkdir(parents=True, exist_ok=True)

    # Writers for kept records
    kept_writer = BatchWriter(processed_dir, batch_size=batch_size)

    # Writers for each filter-out bucket
    lang_writer    = SingleFileWriter(filtered_out_dir / "language_langdetect.jsonl")
    doi_writer     = SingleFileWriter(filtered_out_dir / "no_doi.jsonl")
    ngram_writer   = SingleFileWriter(filtered_out_dir / "ngram_exclusion.jsonl")
    # Duplicates are simply not written anywhere (they're just skipped)

    # Stats
    total_read          = 0
    total_kept          = 0
    total_dup           = 0
    total_non_english   = 0
    total_no_doi        = 0
    total_ngram_dropped = 0
    seen_ids: set[str]  = set()

    drop_counts: dict[str, int] = {
        f"title:{label}": 0 for label, _ in title_patterns
    }
    drop_counts.update({f"abstract:{p}": 0 for p in abstract_phrases})

    try:
        for raw_paper in iter_raw_papers(raw_dir):
            total_read += 1

            # ── (a) Restructure columns ───────────────────────────────────
            paper = restructure_paper(raw_paper)

            title    = paper.get("title")    or ""
            abstract = paper.get("abstract") or ""
            paper_id = paper.get("paperId")  or ""

            # ── (b1) Deduplicate ──────────────────────────────────────────
            if paper_id and paper_id in seen_ids:
                total_dup += 1
                continue
            if paper_id:
                seen_ids.add(paper_id)

            # ── (b2) Language filter (English only) ───────────────────────
            detection_text = (title + " " + abstract).strip()
            lang = detect_language(detection_text) if detection_text else None
            if lang != "en":
                paper["_detected_lang"] = lang
                lang_writer.write(paper)
                total_non_english += 1
                continue

            # ── (b3) DOI filter ───────────────────────────────────────────
            if not paper.get("doi"):
                doi_writer.write(paper)
                total_no_doi += 1
                continue

            # ── (b4) Title regex filter ───────────────────────────────────
            title_label = title_exclusion_match(title)
            if title_label:
                drop_counts[f"title:{title_label}"] += 1
                paper["_drop_reason"] = f"title:{title_label}"
                ngram_writer.write(paper)
                total_ngram_dropped += 1
                continue

            # ── (b5) Abstract phrase filter ───────────────────────────────
            matched_phrase = abstract_exclusion_match(abstract)
            if matched_phrase:
                drop_counts[f"abstract:{matched_phrase}"] += 1
                paper["_drop_reason"] = f"abstract:{matched_phrase}"
                ngram_writer.write(paper)
                total_ngram_dropped += 1
                continue

            # ── Write kept record ─────────────────────────────────────────
            kept_writer.write(paper)
            total_kept += 1

    finally:
        kept_writer.close()
        lang_writer.close()
        doi_writer.close()
        ngram_writer.close()

    stats = {
        "raw_dir":               str(raw_dir),
        "processed_dir":         str(processed_dir),
        "filtered_out_dir":      str(filtered_out_dir),
        "title_patterns":        [label for label, _ in title_patterns],
        "abstract_phrases":      abstract_phrases,
        "total_read":            total_read,
        "total_kept":            total_kept,
        "dropped_duplicates":    total_dup,
        "dropped_non_english":   total_non_english,
        "dropped_no_doi":        total_no_doi,
        "dropped_ngram":         total_ngram_dropped,
        "dropped_by_rule":       drop_counts,
        "batches_written":       kept_writer.batches_written,
    }

    print(
        f"\nDone. Read {total_read:,} → kept {total_kept:,}\n"
        f"  Duplicates removed:  {total_dup:,}\n"
        f"  Non-English dropped: {total_non_english:,}  → {filtered_out_dir}/language_langdetect.jsonl\n"
        f"  No DOI dropped:      {total_no_doi:,}  → {filtered_out_dir}/no_doi.jsonl\n"
        f"  N-gram dropped:      {total_ngram_dropped:,}  → {filtered_out_dir}/ngram_exclusion.jsonl"
    )
    for rule, count in drop_counts.items():
        if count:
            print(f"    [{rule}]: {count:,}")

    write_log(processed_dir / "preprocess_log.jsonl", stats)


if __name__ == "__main__":
    run_preprocessing()