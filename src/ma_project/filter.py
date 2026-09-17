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
                    yield ast.literal_eval(line)
                except (ValueError, SyntaxError):
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue


class BatchWriter:
    """Writes records to sequentially numbered JSONL batch files."""

    def __init__(self, output_dir: Path, prefix: str = "search_results_batch", batch_size: int = BATCH_SIZE):
        self.output_dir  = output_dir
        self.prefix      = prefix
        self.batch_size  = batch_size
        self._file       = None
        self._batch_idx  = 1
        self._count      = 0
        self.total       = 0
        output_dir.mkdir(parents=True, exist_ok=True)

    def _open_next(self):
        if self._file:
            self._file.close()
        path = self.output_dir / f"{self.prefix}_{self._batch_idx}.jsonl"
        print(f"  Opening {path}")
        self._file = open(path, "w", encoding="utf-8")
        self._batch_idx += 1
        self._count = 0

    def write(self, record: dict):
        if self._file is None:
            self._open_next()
        self._file.write(json.dumps(record) + "\n")
        self._count += 1
        self.total  += 1
        if self._count >= self.batch_size:
            self._open_next()

    def close(self):
        if self._file:
            self._file.close()
            self._file = None

    @property
    def batches_written(self):
        return self._batch_idx - 1


class SingleFileWriter:
    """Writes filtered-out records to a single JSONL file."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(path, "w", encoding="utf-8")
        self.total = 0

    def write(self, record: dict):
        self._file.write(json.dumps(record) + "\n")
        self.total += 1

    def close(self):
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