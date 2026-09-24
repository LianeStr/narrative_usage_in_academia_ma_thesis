"""
Preprocess enriched Semantic Scholar paper records for downstream analysis.

This module reads enriched paper records from JSONL batch files, removes
duplicate records, filters records based on OpenAlex metadata, and writes
the resulting records to sequentially numbered JSONL batch files.

Records are deduplicated first by ``paperId`` and then by DOI. When duplicate
DOIs are encountered, metadata from later records is used to fill missing
values in the first occurrence. Records without an OpenAlex match or with a
non-English OpenAlex language are excluded and written to separate JSONL
files according to the reason for exclusion.

The pipeline performs the following operations in order:
1. Read enriched paper records from JSONL batch files.
2. Parse each record either as a Python literal or as JSON.
3. Remove duplicate records based on ``paperId``, preferring records with a
   non-null ``oa_id`` when duplicates are encountered.
4. Remove duplicate records based on DOI, merging metadata from duplicate
   records to fill missing fields.
5. Exclude records for which no OpenAlex match (``oa_id``) is available.
6. Exclude records whose OpenAlex language is not English (``"en"``).
7. Write records that pass all filters to sequentially numbered JSONL batch
   files.
8. Write excluded records to dedicated JSONL files according to the
   exclusion category.
9. Append summary statistics and run metadata to ``preprocess_log.jsonl``.


Input
-----
Enriched input files are read from ``enriched_dir``.
By default, this is:

    config.ENRICHED_DATA_PATH / "doi"

Input files must follow the naming convention:

    enriched_batch_*.jsonl

Files are processed in sorted order. Each non-empty line is expected to
contain one paper record. Records are first parsed with
``ast.literal_eval`` to support Python-dictionary-style records and, if
that fails, are parsed as JSON.

All successfully parsed records are currently loaded into memory before
deduplication and filtering.


Output
------
Accepted records are written to ``processed_dir`` using sequentially
numbered JSONL files:

    enriched_batch_1.jsonl
    enriched_batch_2.jsonl
    ...

The maximum number of records per output file is controlled by
``batch_size``.

Records excluded during preprocessing are written to ``filtered_out_dir``:

``no_match_oa.jsonl``
    Records for which no OpenAlex match is available, identified by a
    null ``oa_id``.

``language_oa.jsonl``
    Records whose OpenAlex language is not ``"en"``.

Duplicate records removed during ``paperId`` or DOI deduplication are not
written to the filtered-out files.

A processing summary is appended to:

    processed_dir / "preprocess_log.jsonl"

Each log entry contains the UTC timestamp of the run, input and output
directories, record counts, duplicate counts, filtering counts, and the
number of output batches created.


Deduplication
------------
Two stages of deduplication are applied.

First, records are deduplicated by ``paperId``. If multiple records share
the same ``paperId``, the first record is retained unless a later duplicate
contains a non-null ``oa_id`` while the retained record does not. In that
case, the later record replaces the retained record.

Second, records are deduplicated by DOI. DOI values are normalized by
stripping surrounding whitespace and converting them to lowercase. The
first record for each DOI is retained, while metadata from later duplicate
records is merged into it to fill missing values.

Configuration
-------------
The following module-level constants define the default pipeline
configuration:

``ENRICHED_DIR``
    Default directory containing enriched DOI records, derived from
    ``config.ENRICHED_DATA_PATH / "doi"``.

``PROCESSED_DIR``
    Default directory for records that pass all preprocessing filters,
    derived from ``config.ENRICHED_DATA_PATH / "filtered"``.

``FILTERED_OUT_DIR``
    Default directory for records excluded during preprocessing.

``BATCH_SIZE``
    Default maximum number of records written to each processed output
    batch.

The ``run_preprocessing`` function accepts these values as arguments,
allowing the pipeline to be configured without modifying the implementation.


Reproducibility
---------------
Processing runs are logged with a UTC timestamp in
``preprocess_log.jsonl`` inside ``processed_dir``. 
The log records the input and output directories, record counts, 
duplicate counts, filtering counts, and number of output batches, 
allowing the results of individual preprocessing runs to be tracked and compared.

The module can be executed with:

    uv run python src/ma_project/filter_oa.py

which runs ``run_preprocessing()`` using the configured default paths and
batch size.
"""

import config

import ast
import json
from pathlib import Path
from datetime import datetime, timezone
from copy import deepcopy


# ── Config ────────────────────────────────────────────────────────────────────

ENRICHED_DIR     = Path(config.ENRICHED_DATA_PATH) / "doi"
PROCESSED_DIR    = Path(config.ENRICHED_DATA_PATH) / "filtered"
FILTERED_OUT_DIR = Path("data/filtered_out")
BATCH_SIZE       = 100_000


# ── I/O helpers ───────────────────────────────────────────────────────────────

def iter_enriched_papers(enriched_dir: Path):
    """Yield parsed paper dicts from all enriched_doi batch files, in order."""
    files = sorted(enriched_dir.glob("enriched_batch_*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No batch files found in {enriched_dir}")

    for path in files:
        print(f"Reading {path.name}...")
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield ast.literal_eval(line)
                except (ValueError, SyntaxError):
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue


def deduplicate(papers: list[dict]) -> tuple[list[dict], int]:
    """
    Deduplicate records by ``paperId``.

    The first record for each ``paperId`` is retained by default. If a
    duplicate record has a non-null ``oa_id`` while the retained record
    does not, the duplicate replaces the retained record.

    Records without a ``paperId`` are skipped.

    Returns:
        A tuple containing the deduplicated records and the number of
        duplicate records removed.
    """
    seen: dict[str, dict] = {}
    for paper in papers:
        paper_id = paper.get("paperId")
        if not paper_id:
            continue
        if paper_id not in seen:
            seen[paper_id] = paper
        elif seen[paper_id].get("oa_id") is None and paper.get("oa_id") is not None:
            # Replace: current stored record has no oa_id but this one does
            seen[paper_id] = paper

    n_duplicates = len(papers) - len(seen)
    return list(seen.values()), n_duplicates


def merge_records(primary: dict, secondary: dict) -> dict:
    """
    Fill missing values in primary using secondary.
    """
    merged = deepcopy(primary)

    for key, value in secondary.items():
        if merged.get(key) in (None, "", [], {}):
            merged[key] = value

    return merged


def deduplicate_doi(papers: list[dict]) -> tuple[list[dict], int]:
    """
    Deduplicate by DOI while merging metadata.

    Keeps the first occurrence of each DOI and fills any missing
    fields from later duplicates.
    """
    seen: dict[str, dict] = {}
    result: list[dict] = []

    for paper in papers:
        doi = paper.get("doi").strip().lower()

        # Keep papers without DOI
        if not doi:
            result.append(paper)
            continue

        if doi not in seen:
            seen[doi] = deepcopy(paper)
            result.append(seen[doi])
        else:
            merged = merge_records(seen[doi], paper)
            seen[doi].clear()
            seen[doi].update(merged)

    n_duplicates = len(papers) - len(result)
    return result, n_duplicates


class BatchWriter:
    """
    Writes records to sequentially numbered JSONL batch files. 
    
    Records are written incrementally to avoid keeping the entire processed dataset in memory. 
    A new batch file is automatically opened whenever the configured batch size is reached. 
    Each output file contains one JSON record per line and is named using the configured prefix and a sequential batch number, 
    for example: 
        enriched_batch_1.jsonl 
        enriched_batch_2.jsonl 
        ...
        
    The class keeps track of the number of records written to the current batch 
    as well as the total number of records written across all batches. 

    Args: 
        output_dir: Directory in which the batch files are created. 
        prefix: Prefix used when naming the batch files. 
        batch_size: Maximum number of records written to each batch file. 
        
    The writer should be closed with ``close()`` after writing is finished.
    """

    def __init__(self, output_dir: Path, prefix: str = "enriched_batch", batch_size: int = BATCH_SIZE):
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
    enriched_dir: Path      = ENRICHED_DIR,
    processed_dir: Path     = PROCESSED_DIR,
    filtered_out_dir: Path  = FILTERED_OUT_DIR,
    batch_size: int         = BATCH_SIZE,
):
    processed_dir.mkdir(parents=True, exist_ok=True)
    filtered_out_dir.mkdir(parents=True, exist_ok=True)

    # ── (1) Read all records into memory ─────────────────────────────────────
    print("Loading all records into memory...")
    all_papers = list(iter_enriched_papers(enriched_dir))
    total_read = len(all_papers)

    # ── (2a) Deduplicate ───────────────────────────────────────────────────────
    all_papers, total_dup = deduplicate(all_papers)
    print(f"PaperId Duplicates removed: {total_dup:,}  ({len(all_papers):,} remaining)")

    # ── (2b) Deduplicate ───────────────────────────────────────────────────────
    all_papers, n_doi_dups = deduplicate_doi(all_papers)
    print(f"DOI Duplicates removed: {n_doi_dups:,}  ({len(all_papers):,} remaining)")

    # ── (3) Filter and write ──────────────────────────────────────────────────
    kept_writer  = BatchWriter(processed_dir, batch_size=batch_size)
    oa_writer    = SingleFileWriter(filtered_out_dir / "no_match_oa.jsonl")
    lang_writer  = SingleFileWriter(filtered_out_dir / "language_oa.jsonl")

    total_kept        = 0
    total_no_oa       = 0
    total_non_english = 0

    try:
        for paper in all_papers:
            if paper.get("oa_id") is None:
                oa_writer.write(paper)
                total_no_oa += 1
                continue

            if paper.get("oa_language") != "en":
                lang_writer.write(paper)
                total_non_english += 1
                continue

            kept_writer.write(paper)
            total_kept += 1

    finally:
        kept_writer.close()
        oa_writer.close()
        lang_writer.close()

    stats = {
        "enriched_dir":          str(enriched_dir),
        "processed_dir":         str(processed_dir),
        "filtered_out_dir":      str(filtered_out_dir),
        "total_read":            total_read,
        "total_kept":            total_kept,
        "dropped_paperID_duplicates":    total_dup,
        "dropped_DOI_duplicates": n_doi_dups,
        "dropped_no_oa_match":   total_no_oa,
        "dropped_non_english":   total_non_english,
        "batches_written":       kept_writer.batches_written,
    }

    print(
        f"\nDone. Read {total_read:,} → kept {total_kept:,}\n"
        f"  Duplicates paperID removed:  {total_dup:,}\n"
        f"  Duplicates DOI removed:  {n_doi_dups:,}\n"
        f"  No OpenAlex match:   {total_no_oa:,}  → {filtered_out_dir}/no_match_oa.jsonl\n"
        f"  Non-English dropped: {total_non_english:,}  → {filtered_out_dir}/language_oa.jsonl"
    )

    write_log(processed_dir / "preprocess_log.jsonl", stats)


if __name__ == "__main__":
    run_preprocessing()