# Note on Script Execution
uv run python src/ma_project/semanticscholar.py

# Dataset Creation
## Search narrative papers
src/ma_project/semanticscholar.py

## Filter Semantic Scholar Results
src/ma_project/filter.py
Input: data/raw
Output: 
data/processed (for the once kept)
data/filtered_out (for the once filtered)
Log: data/processed/preprocess_log.jsonl

## Enrich Metadata with OpenAlex using DOI
enrich_doi.py
Input: data/processed
Output: data/enriched/doi
Log: data/entiched/doi/enrich_doi_log.jsonl
Note: If stopped, the process will resume approximately from where it left off in the dataset. May produce duplicates

## Filter OpenAlex Enriched Data
filter_oa.py
Input: data/enriched/doi
Output: data/enriched/filtered
Log: data/enriched/filtered/preprocess_log.jsonl
Note: removes duplicates (and non-English papers and not found papers)

## Transform the Dataset from Paper to Mentions
paper2mentions.py
Input: data/einriched/filtered
Output: data/mentions
Log: data/mentions/processing_log.jsonl

# Retrieve Contextualized Word Embeddings
Run in kaggle:
notebooks/kaggle/get_embeddings_bert-base.ipynb
notebooks/kaggle/get_embeddings_scibert.ipynb
Input and Output info in local README

# 

