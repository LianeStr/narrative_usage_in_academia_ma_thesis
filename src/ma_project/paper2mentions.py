import pandas as pd
#import glob
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
    if pd.isna(x):
        return ""
    x = str(x)

    # normalize unicode (important for accented / weird forms)
    x = unicodedata.normalize("NFKC", x)

    # remove soft hyphens (your bug)
    x = x.replace("\u00ad", "")
    
    if to_lower:
        x = x.lower()

    return x

def calculate_mention_stats(
    total_read: int,
    mentions_list: list[dict],
    unsure_match_abstract: int,

) -> dict:
    """
    Calculate statistics about narrative mentions.
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
    return str(timedelta(seconds=round(seconds)))


def add_context_tokens(
    mentions_list: list[dict],
    tokenizer,
) -> list[dict]:
    """
    Add tokenizer output, token count, and the token indices
    corresponding to the matched sequence within the context.

    Parameters
    ----------
    mentions_list : list[dict]
        Extracted mention records.

    tokenizer :
        HuggingFace tokenizer.

    Returns
    -------
    list[dict]
        Mentions with added token information.
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
    Plot the top-N matched sequences, colored by match type.
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
    Heatmap of title vs abstract mention counts with marginal distributions.
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

    # ensure a (0,0) in the bottom, left
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
    Extract mentions of narrative-related terms from paper titles and abstracts.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing at least:
        - paperId
        - doi
        - oa_id
        - title
        - abstract

    narrow_pattern : re.Pattern
        More specific regex pattern used to classify full matches.

    broad_pattern : re.Pattern
        Broader regex pattern used to identify candidate mentions.

    Returns
    -------
    list[dict]
        List of dictionaries containing metadata about each mention.
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
    Create a frequency table of matched sequences and classify
    them by BERT tokenization length.

    Parameters
    ----------
    mentions_list : list[dict]
        Extracted mention records.

    tokenizer :
        HuggingFace tokenizer.

    Returns
    -------
    pd.DataFrame
        Summary table with match frequencies and token information.
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