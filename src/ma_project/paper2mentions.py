"""
Extract and characterize mentions of narrative-related terms from paper titles
and abstracts.

This module reads enriched paper records from JSONL batch files, identifies
mentions of `narrative`-related terms in titles and abstracts using regular
expression matching, and stores detailed metadata for each detected mention.

The pipeline distinguishes between exact/full matches of `narrative` or
`narratives` and broader stem-based matches containing `narrat`. For each
mention, the character span, sentence position, text type, and associated
paper metadata are retained. The matched context is additionally tokenized
using the `bert-base-uncased` HuggingFace tokenizer, allowing the token IDs,
token count, and token indices corresponding to the matched sequence to be
stored with each mention.

The pipeline performs the following operations in order:

1. Read enriched paper records from sequentially named JSONL batch files.
2. Normalize title and abstract text using Unicode NFKC normalization,
   removal of soft hyphens, and optional lowercasing.
3. Search paper titles for terms matching the broad `narrat` pattern.
4. Classify title matches as `full_match` when the matched sequence contains
   a complete `narrative`-style term, or as `stem_match` otherwise.
5. Split non-null abstracts into sentences using NLTK sentence tokenization.
6. Search each abstract sentence for the same broad narrative-related pattern.
7. Record detailed metadata for every detected mention, including the paper
   identifiers, text type, cleaned context, match type, matched sequence,
   character offsets, sentence position, and within-context match position.
8. Assign an internal sequential identifier to each extracted mention (´mention_id´).
9. Tokenize each mention context with the `bert-base-uncased` tokenizer and
    record token IDs and the token indices overlapping the matched character
    span.
10. Write all extracted mentions to a JSONL file.
11. Create a frequency summary of matched sequences and their match types.
12. Tokenize the unique matched sequences and record their BERT tokenization
    lengths and write the match summary to CSV.
13. Calculate paper-level and match-level processing statistics.
14. Append the processing statistics and runtime information to a JSONL log.
15. Generate visualizations summarizing the most frequent matched sequences
    and the distribution of title versus abstract mentions per paper.

Input
-----
Enriched paper records are read from `input_dir`. By default, this is
`config.ENRICHED_DATA_PATH / "filtered"`.

Input files must follow the naming convention:

```
enriched_batch_*.jsonl
```

Each non-empty line is expected to contain one paper record with, at minimum,
the following fields:

`paperId`
Unique identifier of the paper.

`doi`
DOI associated with the paper, when available.

`oa_id`
OpenAlex identifier associated with the paper, when available.

`title`
Paper title.

`abstract`
Paper abstract, which may be missing.

Records are parsed first with `ast.literal_eval` to support
Python-dictionary-style records and, if that fails, with `json.loads`.
Malformed records that cannot be parsed by either method are skipped.

Mention Extraction
------------------
Narrative-related terms are detected using two regular expression patterns.

`NARRATIVE_RE`
Matches words containing `narrative` and is used to identify
full/exact narrative-related matches.

`NARRAT_RE`
Matches words containing `narrat` and is used as the broader candidate
pattern.

A mention is classified as `full_match` when the matched sequence contains
a match to `NARRATIVE_RE`. Otherwise, a candidate detected by
`NARRAT_RE` is classified as `stem_match`.

Matching is performed on cleaned, lowercased text. Titles are treated as a
single text unit, whereas abstracts are split into sentences before matching.

For each mention, the following information is recorded:

`paperId`
Semantic Scholar identifier of the source paper.

`doi`
DOI of the source paper.

`oa_id`
OpenAlex identifier of the source paper.

`text_type`
Either `title` or `abstract`.

`text_cleaned`
Cleaned text containing the matched sequence. For abstracts, this is the
individual sentence containing the match.

`match_type`
Either `full_match` or `stem_match`.

`matched_seq`
The exact matched character sequence after text cleaning,
i.e. the word that is/contains the regex-pattern/the substring.

`matched_char_start` / `matched_char_end`
Character offsets of the matched sequence within `text_cleaned`.

`sentence_pos`
Zero-based sentence position within the abstract. This is `-1` for
title mentions.

`pos`
Zero-based position of the match within the corresponding title or
abstract sentence. Enstures unique identifiability of more than one 
narrative mention per title or sentence.

`id`
Sequential internal identifier assigned to each extracted mention.

Tokenization
------------
Mention contexts are tokenized using the HuggingFace
`bert-base-uncased` tokenizer.

For each mention, the pipeline records:

`context_token_ids`
BERT input token IDs for the complete cleaned context, including special
tokens.

`context_n_tokens`
Number of tokens in the complete context, including special tokens.

`matched_token_indices`
Zero-based indices of BERT tokens whose character offsets overlap the
matched sequence.

Special tokens such as `[CLS]` and `[SEP]` are excluded from
`matched_token_indices` because they have zero-length character offsets.

The same tokenizer is used when constructing the match-frequency summary,
where each unique matched sequence is represented by its BERT tokens and
corresponding token count.

Output
------
Extracted mention records are written to:

```
output_dir / "narrative_mentions.jsonl"
```

Each line contains one mention dictionary, including paper metadata, match
metadata, context information, and BERT tokenization information.

A frequency summary of matched sequences is written to:

```
output_dir / "matched_summary.csv"
```

The summary contains the matched sequence, match type, occurrence count,
BERT tokens, and number of BERT tokens.

Processing statistics are appended to:

```
output_dir / "processing_log.jsonl"
```

Each log entry contains a UTC timestamp and summary statistics including:

`total_papers_read`
Number of paper records loaded from the input batches.

`papers_without_mentions`
Number of papers for which no narrative-related mention was detected.

`no_mention_title_abstract_unsure`
Number of papers without a title match where the abstract is missing,
making the absence of an abstract match indeterminate.

`no_mention_title_abstract`
Number of papers without detected mentions after accounting for the
missing-abstract cases.

`match_type_distribution`
Frequency distribution of `full_match` and `stem_match` mentions.

`conservative_match_count`
Number of mentions whose matched sequence is exactly `narrative` or
`narratives`.

`runtime_seconds`
Processing runtime in seconds.

`runtime_human`
Processing runtime formatted as hours, minutes, and seconds.

Visualizations
--------------
Figures are written to `vis_dir`. By default, this is:

```
results/paper2mentions
```

The pipeline generates two visualizations.

`top20_matches.png`
A log-scaled bar chart showing the most frequently detected matched
sequences, with bars distinguished by match type.

`nr_mentions_per_paper_heatmap.png`
A heatmap showing the joint distribution of the number of narrative
mentions in each paper's title and abstract. Marginal distributions show
the corresponding title- and abstract-level paper counts.

Configuration
-------------
The following module-level constants define the default pipeline
configuration:

`INPUT_DIR`
Default input directory obtained from
`config.ENRICHED_DATA_PATH / "filtered"`.

`OUTPUT_DIR`
Default mention-output directory obtained from
`config.MENTIONS_PATH`.

`VIZ_DIR`
Default visualization directory.

`MODEL_NAME`
HuggingFace tokenizer identifier. The current configuration uses
`bert-base-uncased`.

`TOKENIZER`
Pre-loaded HuggingFace tokenizer used for context and matched-sequence
tokenization.

`NARRATIVE_RE`
Regular expression used to identify full `narrative`-style matches.

`NARRAT_RE`
Broader regular expression used to identify candidate
narrative-related matches.

The `transform` function accepts input, output, and visualization
directories as arguments, allowing the pipeline to be executed with
alternative paths without changing the module-level configuration.

Reproducibility
---------------
The extraction procedure is deterministic for a fixed input dataset,
regular-expression configuration, NLTK sentence-tokenization behavior, and
installed tokenizer version.

The tokenizer configuration is explicitly fixed to
`bert-base-uncased`. Character offsets returned by the tokenizer are used
to map matched character spans back to their corresponding BERT token
indices.

Processing statistics are appended to `processing_log.jsonl` so that
individual processing runs and their runtimes can be tracked over time.

The module can be executed with:

```
uv run python src/ma_project/paper2mentions.py
```

which runs `transform()` using the configured default paths.
"""

import pandas as pd
import json
from pathlib import Path
import ast
import nltk
import re
import unicodedata
from datetime import datetime, timezone, timedelta
import time
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm

import config


# ── Config ────────────────────────────────────────────────────────────────────

INPUT_DIR       = Path(config.ENRICHED_DATA_PATH) / "filtered"
OUTPUT_DIR      = Path(config.MENTIONS_PATH)
VIZ_DIR         = Path("results/paper2mentions")

MODEL_NAME = "bert-base-uncased"
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)


# REGEX PATTERNS
NARRATIVE_RE = re.compile(r"\b\w*narrative\w*\b", flags=re.IGNORECASE)
NARRAT_RE = re.compile(r"\b\w*narrat\w*\b", flags=re.IGNORECASE)


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

        
# ── Auxilirary Funcitons ───────────────────────────────────────────────────────

def clean_text(x: str, to_lower=True):
    """
    Normalize and optionally lowercase text before pattern matching.

    Missing values are converted to an empty string. Unicode is normalized
    using NFKC, soft hyphens are removed, and text is lowercased when
    ``to_lower`` is True.

    Parameters
    ----------
    x : str
        Input text or a value that can be converted to a string.
    to_lower : bool, default=True
        Whether to lowercase the normalized text.

    Returns
    -------
    str
        Cleaned text suitable for downstream matching.
    """
    if pd.isna(x):
        return ""
    x = str(x)

    # normalize unicode (important for accented / weird forms)
    x = unicodedata.normalize("NFKC", x)

    # remove soft hyphens
    x = x.replace("\u00ad", "")
    
    if to_lower:
        x = x.lower()

    return x

def calculate_mention_stats(
    total_read: int,
    mentions_list: list[dict],
    unsure_match_abstract: int,

) -> dict:
    """ Calculate summary statistics for narrative-related mentions. 
    The statistics are calculated at both the paper level and the mention level. 
    Papers are considered to have a mention if their ``paperId`` occurs in ``mentions_list``. 
    
    Parameters 
    ---------- 
    total_read : int Total number of papers processed by the pipeline. 
    mentions_list : list[dict] Extracted mention records. 
        Each record must contain at least ``paperId``, ``match_type``, and ``matched_seq``. 
    unsure_match_abstract : int Number of papers without a title match for which 
        the abstract is missing, making it impossible to determine whether the paper 
        has an abstract-level mention. 
    
    Returns 
    ------- 
    dict Dictionary containing: 
    ``total_papers_read`` 
        Total number of papers processed. 
        
    ``papers_without_mentions`` 
        Number of papers for which no narrative-related mention was detected. 
    
    ``no_mention_title_abstract_unsure`` 
        Number of papers without a detected mention where the abstract was unavailable 
        and therefore could not be checked. 
        
    ``no_mention_title_abstract`` 
        Number of papers without a detected mention after excluding cases with a missing abstract. 
        
    ``match_type_distribution`` 
        Frequency of each mention type, such as ``full_match`` and ``stem_match``. 
        
    ``conservative_match_count`` 
        Number of mentions whose matched sequence is exactly ``narrative`` or ``narratives``. 
    """

    mentions_df = pd.DataFrame(mentions_list)

    # Papers with at least one mention
    mentioned_papers = set(mentions_df["paperId"].unique())

    # Papers with no mentions
    no_mention_count = total_read - len(mentioned_papers)

    stats = {
        "total_papers_read": total_read,
        "papers_without_mentions": no_mention_count,
        "no_mention_title_abstract_unsure": unsure_match_abstract,
        "no_mention_title_abstract": no_mention_count - unsure_match_abstract,
    }

    # Match type distribution
    if not mentions_df.empty:
        match_distribution = (
            mentions_df["match_type"]
            .value_counts()
            .to_dict()
        )

        # Conservative count: only exact "narrative" or "narratives"
        conservative_match_count = len(
            mentions_df[
                mentions_df["matched_seq"].isin(
                    ["narrative", "narratives"]
                )
            ]
        )

    else:
        match_distribution = {}
        conservative_match_count = 0

    stats.update(
        {
            "match_type_distribution": match_distribution,
            "conservative_match_count": conservative_match_count,
        }
    )

    return stats


def write_log(log_path: Path, stats: dict):
    """Append a processing run record to a JSONL log file."""
    record = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), **stats}
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    print(f"Run logged to {log_path}")


def write_mentions_jsonl(
    output_path: Path,
    mentions_list: list[dict],
):
    """
    Write mention records as JSON Lines.
    Each line is one mention dictionary.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        for mention in mentions_list:
            f.write(json.dumps(mention, ensure_ascii=False) + "\n")

    print(f"Wrote {len(mentions_list)} mentions to {output_path}")


def write_match_summary(
    matched_summary: pd.DataFrame,
    output_path: Path,
):
    """
    Export match summary table.
    """

    matched_summary.to_csv(
        output_path,
        index=False,
        encoding="utf-8",
    )

    print(f"Wrote match summary to {output_path}")


def format_runtime(seconds: float) -> str:
    """Format a duration in seconds as a human-readable time string."""
    return str(timedelta(seconds=round(seconds)))


def add_context_tokens(
    mentions_list: list[dict],
    tokenizer,
) -> list[dict]:
    """ 
    Add tokenizer information and matched-token indices to mention records. 
    
    Each mention's cleaned text is tokenized with the provided HuggingFace tokenizer. 
    The resulting token IDs and total token count are stored in the mention record. 
    Character offsets for the matched sequence are then mapped to the corresponding 
    tokenizer token indices. 
    
    Special tokens such as ``[CLS]`` and ``[SEP]`` are excluded from ``matched_token_indices`` 
    because they have zero-length character offsets. 
    
    Parameters 
    ---------- 
    mentions_list : list[dict] Extracted mention records. Each record must contain 
        ``text_cleaned``, ``matched_char_start``, and ``matched_char_end``. 
        
    tokenizer HuggingFace tokenizer used to tokenize the mention context. 
    The tokenizer must support ``return_offsets_mapping=True``. 
    
    Returns 
    ------- 
    list[dict] 
        The input mention records with the following fields added: 
        
        ``context_token_ids`` 
            Token IDs for the complete cleaned context, including special tokens 
            added by the tokenizer. 
            
        ``context_n_tokens`` 
            Number of tokens in the complete context, including special tokens. 
            
        ``matched_token_indices`` 
            Zero-based indices of tokens whose character spans overlap the matched sequence. 
    """

    for mention in mentions_list:
        context = mention["text_cleaned"]

        encoding = tokenizer(
            context,
            add_special_tokens=True,
            return_offsets_mapping=True,
        )

        token_ids = encoding["input_ids"]
        offsets = encoding["offset_mapping"]

        mention["context_token_ids"] = token_ids
        mention["context_n_tokens"] = len(token_ids)

        char_start = mention["matched_char_start"]
        char_end = mention["matched_char_end"]

        # tokens whose char span overlaps the matched sequence's char span
        # special tokens ([CLS]/[SEP]) have offset (0, 0) and are excluded
        mention["matched_token_indices"] = [
            idx
            for idx, (start, end) in enumerate(offsets)
            if end > char_start and start < char_end and not (start == 0 and end == 0)
        ]

    return mentions_list


# ── Visualization Funcitons ───────────────────────────────────────────────────────

def plot_top_matches(
    matched_summary: pd.DataFrame,
    output_path: Path,
    top_n: int = 25,
):
    """
    Plot the most frequent narrative-related matched sequences.

    The ``top_n`` matched sequences are selected by descending occurrence
    count. Bars are colored according to match type (``full_match`` or
    ``stem_match``), and the y-axis is displayed on a logarithmic scale to
    accommodate differences in frequency between matched sequences.

    Parameters
    ----------
    matched_summary : pd.DataFrame
        Match summary containing at least the columns ``matched_seq``,
        ``count``, and ``match_type``.
    output_path : Path
        Path at which the generated figure is saved.
    top_n : int, default=25
        Number of highest-frequency matched sequences to include in the
        figure.

    Returns
    -------
    None
        The figure is saved to ``output_path`` and is not returned.
    """

    plot_df = (
        matched_summary
        .sort_values("count", ascending=False)
        .head(top_n)
        .copy()
    )

    colors = {
        "full_match": "tab:blue",
        "stem_match": "tab:orange",
    }

    plt.figure(figsize=(10, 6))

    bars = plt.bar(
        plot_df["matched_seq"],
        plot_df["count"],
        color=plot_df["match_type"].map(colors),
    )

    plt.yscale("log")

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{int(height):,}",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=45,
        )

    plt.xlabel("Matched sequence")
    plt.ylabel("Count (log scale)")
    plt.title(f"Top {top_n} matched sequences")

    plt.xticks(rotation=45, ha="right")

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=colors[k])
        for k in colors
    ]
    plt.legend(handles, colors.keys(), title="Match type")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Wrote figure to {output_path}")


def plot_title_vs_abstract_heatmap(
    mentions_list: list[dict],
    output_path: Path,
    ):
    """ 
    Plot the distribution of title and abstract narrative mentions per paper. 
    
    For each paper, the number of detected narrative-related mentions in the title and abstract is calculated. 
    The resulting two-dimensional distribution is displayed as a heatmap, where each cell represents 
    the number of papers with a given combination of title- and abstract-level mention counts. 
    Marginal bar plots show the corresponding distributions of title and abstract mention counts across papers. 
    
    The heatmap and marginal distributions use logarithmic scaling to accommodate 
    differences in the number of papers across mention-count categories. 
    
    Parameters 
    ---------- 
    mentions_list : list[dict] 
        Extracted mention records. Each record must contain ``paperId`` and 
        ``text_type``, where ``text_type`` identifies whether the mention occurs 
        in a ``title`` or ``abstract``. 
        
    output_path : Path 
        Path at which the generated figure is saved. The parent directory is created
        if it does not already exist. 
    
    Returns 
    ------- 
    None 
        The figure is saved to ``output_path`` and is not returned. 
    """

    mentions_df = pd.DataFrame(mentions_list)

    # Paper-level counts
    paper_counts = (
        mentions_df
        .groupby(["paperId", "text_type"])
        .size()
        .unstack(fill_value=0)
    )

    if "title" not in paper_counts:
        paper_counts["title"] = 0

    if "abstract" not in paper_counts:
        paper_counts["abstract"] = 0

    paper_counts = paper_counts.rename(
        columns={
            "title": "title_mentions",
            "abstract": "abstract_mentions",
        }
    )

    paper_counts = paper_counts[
        ["title_mentions", "abstract_mentions"]
    ]

    # 2D count matrix
    heatmap_data = (
        paper_counts
        .groupby(
            ["title_mentions", "abstract_mentions"]
        )
        .size()
        .unstack(fill_value=0)
    )

    # ensure a ascending counts starting from the bottom, left
    heatmap_data = heatmap_data.sort_index(ascending=False)

    # Marginals
    title_dist = paper_counts["title_mentions"].value_counts().sort_index()
    abstract_dist = paper_counts["abstract_mentions"].value_counts().sort_index()


    # Layout
    fig = plt.figure(figsize=(10, 10))

    gs = fig.add_gridspec(
        4,
        2,
        width_ratios=(4, 1),
        height_ratios=(1, 4, 0.4, 0.2),
        hspace=0.05,
        wspace=0.05,
    )

    ax_top = fig.add_subplot(gs[0, 0])
    ax_heat = fig.add_subplot(gs[1, 0])
    ax_right = fig.add_subplot(gs[1, 1])
    cbar_ax = fig.add_subplot(gs[3, 0]) # colorbar


    # Top marginal
    ax_top.bar(
        abstract_dist.index,
        abstract_dist.values,
    )

    ax_top.set_yscale("log")
    ax_top.set_ylabel("Papers")
    ax_top.tick_params(
        axis="x",
        labelbottom=False,
    )


    # Main heatmap
    heat = sns.heatmap(
        heatmap_data,
        ax=ax_heat,
        norm=LogNorm(),
        cmap="viridis",
        cbar=True,
        cbar_ax=cbar_ax,
        cbar_kws={"orientation": "horizontal"}
    )

    heat.collections[0].colorbar.set_label("Number of papers (log scale)")

    ax_heat.set_xlabel(
        "Number of narrative mentions in abstract per paper"
    )

    ax_heat.set_ylabel(
        "Number of narrative mentions in title per paper"
    )


    # Right marginal
    ax_right.barh(
        title_dist.index,
        title_dist.values,
    )

    ax_right.set_xscale("log")
    ax_right.set_xlabel("Papers")
    ax_right.tick_params(
        axis="y",
        labelleft=False,
    )


    fig.suptitle(
        "Narrative mentions per paper: title vs abstract \n NOTE: axis might not overlap",
        y=0.95,
    )

    output_path.parent.mkdir(parents=True,exist_ok=True,)

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Wrote figure to {output_path}")


# ── Core Funcitons ───────────────────────────────────────────────────────
def paper2mentions(
    df: pd.DataFrame,
    narrow_pattern=NARRATIVE_RE,
    broad_pattern=NARRAT_RE,
) -> tuple[list[dict], int]:
    """ 
    Extract narrative-related mentions from paper titles and abstracts. 
    
    Titles and abstracts are searched using a broad regular expression containing 
    the ``narrat`` stem. Each resulting match is subsequently classified as either 
    a ``full_match`` or ``stem_match`` using the narrower ``narrative`` pattern. 
    
    Titles are processed as complete text fields, while non-missing abstracts are 
    split into sentences using NLTK's sentence tokenizer before matching. 
    For each detected mention, the function records paper metadata, the text type, 
    cleaned context, match type, matched character span, sentence position, 
    and position of the match within the corresponding text unit. 

    Papers without a title match and with a missing abstract are counted separately 
    because the absence of an abstract prevents the function from determining whether 
    the paper contains a narrative-related mention. 
    
    Parameters 
    ---------- 
    df : pd.DataFrame 
        DataFrame containing paper records. The following columns are required: 
        
        ``paperId`` 
            Unique identifier of the paper. 
        
        ``doi`` 
            DOI associated with the paper. 
        
        ``oa_id`` 
            Open-access identifier associated with the paper. 
            
        ``title`` 
            Paper title. 
            
        ``abstract`` 
        Paper abstract, which may be missing. 
        
    narrow_pattern : re.Pattern, default=NARRATIVE_RE 
        Regular expression used to classify broad matches as ``full_match`` 
        when the matched sequence contains a ``narrative``-related term. 
        
    broad_pattern : re.Pattern, default=NARRAT_RE 
        Broader regular expression used to identify candidate narrative-related 
        matches. 
        
    
    Returns 
    ------- 
    tuple[list[dict], int] A tuple containing: 
    
        ``mentions_list`` 
            List of dictionaries, with one dictionary for each detected mention. 
            Each record contains paper metadata, text type, cleaned context, match type, 
            matched sequence, character offsets, sentence position, and within-text match 
            position. 
            
        ``unsure_match_abstract`` 
            Number of papers for which no title mention was detected and the abstract is 
            missing. These papers cannot be classified as having no narrative-related 
            mention because their abstract could not be searched. 
    """

    narrative_mentioned = []
    # stats: how many cases where there there is no narrative-match in the title and the abstract is not known
    unsure_match_abstract = 0 


    for _, row in df.iterrows():

        paperId = row.paperId
        doi = row.doi
        oa_id = row.oa_id

        # --- TITLE ---
        clean_title = clean_text(row.title)

        matches = list(broad_pattern.finditer(clean_title))
        stem_match = bool(matches)

        if stem_match:
            for pos, m in enumerate(matches):
                mention = m.group()
                full_match = bool(narrow_pattern.search(mention))

                narrative_mentioned.append(
                    {
                        "paperId": paperId,
                        "doi": doi,
                        "oa_id": oa_id,
                        "text_type": "title",
                        "text_cleaned": clean_title,
                        "match_type": "full_match" if full_match else "stem_match",
                        "matched_seq": mention,
                        "matched_char_start": m.start(),
                        "matched_char_end": m.end(),
                        "sentence_pos": -1,
                        "pos": pos,
                    }
                )
        else:
            if pd.isna(row.abstract):
                unsure_match_abstract += 1

        # --- ABSTRACT ---
        if pd.notna(row.abstract):

            sentences = nltk.tokenize.sent_tokenize(row.abstract)

            for sent_pos, sent in enumerate(sentences):

                clean_sent = clean_text(sent)
                matches = list(broad_pattern.finditer(clean_sent))

                for pos, m in enumerate(matches):
                    mention = m.group()
                    full_match = bool(narrow_pattern.search(mention))

                    narrative_mentioned.append(
                        {
                            "paperId": paperId,
                            "doi": doi,
                            "oa_id": oa_id,
                            "text_type": "abstract",
                            "text_cleaned": clean_sent,
                            "match_type": "full_match" if full_match else "stem_match",
                            "matched_seq": mention,
                            "matched_char_start": m.start(),
                            "matched_char_end": m.end(),
                            "sentence_pos": sent_pos,
                            "pos": pos,
                        }
                    )


    # create an primary key for internal handeling
    for i, mention in enumerate(narrative_mentioned):
        mention["id"] = i


    return narrative_mentioned, unsure_match_abstract


def create_match_summary(
    mentions_list: list[dict],
    tokenizer,
) -> pd.DataFrame:
    """ 
    Create a frequency summary of detected matched sequences. 
    
    The extracted mention records are grouped by matched sequence and match type. 
    For each group, the number of occurrences is calculated and the matched sequence 
    is tokenized using the provided HuggingFace tokenizer. The resulting token sequence 
    and number of tokens are added to the summary. 
    
    Special tokens are not added during tokenization, so ``n_tokens`` represents 
    the number of tokenizer subword tokens required to represent the matched sequence itself. 
    
    Parameters 
    ---------- 
    mentions_list : list[dict] 
        Extracted mention records. Each record must contain ``matched_seq`` and
        ``match_type``. 
    
    tokenizer 
        HuggingFace tokenizer used to tokenize the matched sequences. 
        
    Returns 
    ------- 
    pd.DataFrame 
        Summary table containing one row per unique combination of matched sequence 
        and match type, with the following columns: 
        
        ``matched_seq`` 
            Matched character sequence. 
            
        ``match_type`` 
            Match classification, either ``full_match`` or ``stem_match``. 
            
        ``count``
            Number of occurrences of the matched sequence with the given match type. 
            
        ``tokens`` 
            BERT tokenizer subword tokens corresponding to the matched sequence. 
            
        ``n_tokens`` 
            Number of tokenizer subword tokens in the matched sequence. 
    
        Rows are sorted by ``count`` in descending order.
    """

    mentions_df = pd.DataFrame(mentions_list)

    matched_summary = (
        mentions_df
        .groupby(["matched_seq", "match_type"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    matched_summary["tokens"] = matched_summary["matched_seq"].apply(
        lambda x: tokenizer.convert_ids_to_tokens(
            tokenizer(
                x,
                add_special_tokens=False
            )["input_ids"]
        )
    )

    matched_summary["n_tokens"] = matched_summary["tokens"].str.len()

    return matched_summary


# ── Main pipeline ─────────────────────────────────────────────────────────────

def transform(
        input_dir: Path      = INPUT_DIR,
        output_dir: Path     = OUTPUT_DIR,
        vis_dir: Path        = VIZ_DIR):

    # record run time
    start_time = time.perf_counter()
    
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    # ── (1) Read all records into memory ─────────────────────────────────────
    print("Loading all records into memory...")
    all_papers = list(iter_enriched_papers(input_dir))
    total_read = len(all_papers) # ADD FOR LOG

    df = pd.DataFrame(all_papers)

    # ── (2) Extract mentions ─────────────────────────────────────────────────
    mentions_list, unsure_match_abstract = paper2mentions(df)

    mentions_list = add_context_tokens(
        mentions_list,
        tokenizer=TOKENIZER,
        )

    # ── (3) Write mentions ───────────────────────────────────────────────────
    write_mentions_jsonl(
        output_path=output_dir / "narrative_mentions.jsonl",
        mentions_list=mentions_list,
    )

    # ── (4) Create match summary ─────────────────────────────────────────────
    matched_summary = create_match_summary(
        mentions_list=mentions_list,
        tokenizer=TOKENIZER,
    )

    write_match_summary(
        matched_summary=matched_summary,
        output_path=output_dir / "matched_summary.csv",
    )

    # ── (5) Logging ──────────────────────────────────────────────────────────
    stats = calculate_mention_stats(
        total_read=total_read,
        mentions_list=mentions_list,
        unsure_match_abstract = unsure_match_abstract,
    )

    elapsed_seconds = time.perf_counter() - start_time

    stats.update(
    {
        "runtime_seconds": round(elapsed_seconds, 2),
        "runtime_human": str(timedelta(seconds=round(elapsed_seconds))),
    }
    )

    print(f"Finished in {format_runtime(elapsed_seconds)} (hh:mm:ss)")

    write_log(
        log_path=output_dir / "processing_log.jsonl",
        stats=stats,
    )

    # add visualisations 
    plot_top_matches(
    matched_summary,
    vis_dir / "top20_matches.png",
    )


    plot_title_vs_abstract_heatmap(
        mentions_list, 
        vis_dir / "nr_mentions_per_paper_heatmap.png")


if __name__ == "__main__":
    transform()