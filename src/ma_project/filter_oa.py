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
    Deduplicate by paperId, preferring records with a non-null oa_id.
    Mirrors the notebook logic:
      sort so non-null oa_id comes first, then drop_duplicates keeping first.
    Returns (deduplicated list, number of duplicates removed).
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

"""
def normalize_text(value: str | None) -> str | None:
    #Normalize text for matching.

    if not value:
        return None
    
    value = value.lower().strip()
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"[^\w\s]", "", value)

    return value


def deduplicate_title_venue(papers: list[dict]) -> tuple[list[dict], int]:
   # Deduplicate by title + venue overlap.

    #Only applies to titles with more than one word.
    #Merges metadata from duplicate records.

    seen: dict[tuple[str, str], dict] = {}
    result: list[dict] = []

    for paper in papers:
        title = normalize_text(paper.get("title"))
        venue = normalize_text(paper.get("venue"))

        # Cannot safely match without title or venue
        if not title or not venue:
            result.append(paper)
            continue

        # Avoid short generic titles ("Introduction", "Editorial", etc.)
        if len(title.split()) < 2:
            result.append(paper)
            continue

        key = (title, venue)

        if key not in seen:
            seen[key] = deepcopy(paper)
            result.append(seen[key])
        else:
            merged = merge_records(seen[key], paper)
            seen[key].clear()
            seen[key].update(merged)

    n_duplicates = len(papers) - len(result)
    return result, n_duplicates
    """

class BatchWriter:
    """Writes records to sequentially numbered JSONL batch files."""

    def __init__(self, output_dir: Path, prefix: str = "enriched_batch", batch_size: int = BATCH_SIZE):
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

    # ── (2c) Deduplicate ───────────────────────────────────────────────────────
    #all_papers, n_title_dups = deduplicate_title_venue(all_papers)
    #print(f"Title Duplicates removed: {n_title_dups:,}  ({len(all_papers):,} remaining)")

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
        #"dropped_title_duplicates": n_title_dups,
        "dropped_no_oa_match":   total_no_oa,
        "dropped_non_english":   total_non_english,
        "batches_written":       kept_writer.batches_written,
    }

    print(
        f"\nDone. Read {total_read:,} → kept {total_kept:,}\n"
        f"  Duplicates paperID removed:  {total_dup:,}\n"
        f"  Duplicates DOI removed:  {n_doi_dups:,}\n"
        #f"  Duplicates Title removed:  {n_title_dups:,}\n"
        f"  No OpenAlex match:   {total_no_oa:,}  → {filtered_out_dir}/no_match_oa.jsonl\n"
        f"  Non-English dropped: {total_non_english:,}  → {filtered_out_dir}/language_oa.jsonl"
    )

    write_log(processed_dir / "preprocess_log.jsonl", stats)


if __name__ == "__main__":
    run_preprocessing()