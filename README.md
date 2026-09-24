# Note on Script Execution

Run the scripts with:

```bash
uv run python src/ma_project/semanticscholar.py
```

# Dataset Creation

## Search Narrative Papers

Script:

```text
src/ma_project/semanticscholar.py
```

Flags that override `config.py`:

```bash
uv run python semanticscholar.py --query "narrative"
uv run python semanticscholar.py --year-range "2015-2025"
uv run python semanticscholar.py --output-dir "data/raw"
uv run python semanticscholar.py --verbose
```

**Expected Scripts:**

* `src/ma_project/config.py` — to set `YEAR_RANGE`, `FIELDS_SEMSCHO`, `RAW_DATA_PATH`
* `src/ma_project/key.py` — OPTIONAL to store the API keys (`API_KEY_OPENALEX`, `API_KEY_SEMSCHO`)

**Log:** `data/raw/run_log.jsonl`

## Filter Semantic Scholar Results

Script:

```text
src/ma_project/filter.py
```

**Input:** `data/raw`

**Output:**

* `data/processed` — papers that were kept in batches of 100_000
* `data/filtered_out` — papers that were filtered out (langdetect, narrative review ngram, no doi)

**Log:** `data/processed/preprocess_log.jsonl`

## Enrich Metadata with OpenAlex Using DOI

Script:

```text
src/ma_project/enrich_doi.py
```

**Input:** `data/processed`

**Output:** `data/enriched/doi`

**Log:** `data/enriched/doi/enrich_doi_log.jsonl`

> **Note:** If the process is stopped, it will approximately resume from where it left off in the dataset. This may produce duplicates.

## Filter OpenAlex Enriched Data

Script:

```text
src/ma_project/filter_oa.py
```

**Input:** `data/enriched/doi`

**Output:** `data/enriched/filtered`

**Log:** `data/enriched/filtered/preprocess_log.jsonl`

> **Note:** Removes duplicates, non-English papers, and papers that were not found.

## Transform the Dataset from Papers to Mentions

Script:

```text
src/ma_project/paper2mentions.py
```

**Input:** `data/enriched/filtered`

**Output:** `data/mentions`

```text
matched_summary.csv
narrative_mentions.jsonl
```

**Visualisations:** `results/paper2mentions`
```text
nr_mentions_per_paper_heatmap.png
top20_matches.png
```

**Log:** `data/mentions/processing_log.jsonl`

# Retrieve Contextualized Word Embeddings

Run the following notebooks in Kaggle:

```text
notebooks/kaggle/get_embeddings_bert-base.ipynb
notebooks/kaggle/get_embeddings_scibert.ipynb
```

Input and output information is provided in the local README.

# Create Sample for Tuning of Clustering Parameters

```text
notebooks/sample.ipynb
```

**Input:** 
```text
data/mentions/narrative_mentions.jsonl
data/embeddings/bert-base/progress.json
data/enriched/filtered/enriched_batch_*.jsonl
```

**Output:** data/sample/sample_tuning_10per.json

## Get Sampled Embeddings in Kaggle

```text
notebooks/kaggle/create-embeddings-samples-bertbase.ipynb
- sample_bert_base_mean_last4.parquet
- sample_bert_base_last.parquet
- sample_bert_base_2nd_last.parquet

notebooks/kaggle/create-embeddings-samples-scibert.ipynb
- sample_scibert_base_mean_last4.parquet
- sample_scibert_base_last.parquet
- sample_scibert_base_2nd_last.parquet
```

> **Note:** More infos in the local README.txt 
('notebooks/kaggle/README.txt`)

# Create K-Distance Graph

## Full Dataset (Kaggle)

```text
notebooks/kaggle/sorted-k-dist-graph.ipynb
```

## Sample (Kaggle)

```text
notebooks/kaggle/bert-base-sample-tuning-kdist.ipynb
notebooks/kaggle/bert-base-sample-tuning-kdist.ipynb
```

# DBSCAN Parameter Tuning

## Runtime Estimation (Kaggle)

```text
notebooks/paper_vis/Testing_Runtime_dbscan.ipynb
```

**Data:**

```text
notebooks/cluster_speed_kaggle.jsonl
notebooks/cluster_speed_kaggle_larger.jsonl
notebooks/cluster_speed_myPC.jsonl
```

## Sample (Kaggle)

```text
notebooks/kaggle/sample-tuning-dbscan-bb-11.ipynb
notebooks/kaggle/sample-tuning-dbscan-bb-12.ipynb
notebooks/kaggle/sample-tuning-dbscan-bb-mean.ipynb
notebooks/kaggle/sample-tuning-dbscan-sb-11.ipynb
notebooks/kaggle/sample-tuning-dbscan-sb-12.ipynb
notebooks/kaggle/sample-tuning-dbscan-sb-mean.ipynb
```

## Full Dataset (Kaggle)

```text
notebooks/kaggle/full-run-dbscan-bb-12-adjusted.ipynb
notebooks/kaggle/full-run-dbscan-bb-12.ipynb
notebooks/kaggle/full-run-dbscan-sb-11-adjusted.ipynb
notebooks/kaggle/full-run-dbscan-sb-11.ipynb
```

## Clustering Results

### Full Dataset

```text
results/dbscan_full/full_clustering_results_bert-base_layer_12.json
results/dbscan_full/full_clustering_results_bert-base_layer_12_adjusted.json
results/dbscan_full/full_clustering_results_scibert_layer_11.json
results/dbscan_full/full_clustering_results_scibert_layer_11_adjusted.json
```

### Sample

```text
results/dbscan_sample/clustering_results_bert-base_layer_11.json
results/dbscan_sample/clustering_results_bert-base_layer_12.json
results/dbscan_sample/clustering_results_bert-base_mean.json
results/dbscan_sample/clustering_results_scibert_layer_11.json
results/dbscan_sample/clustering_results_scibert_layer_12.json
results/dbscan_sample/clustering_results_scibert_layer_mean.json
```

# Visualisation of the Results

## k-Distance: BERT vs. SciBERT

```text
notebooks/visualisations/vis_kDist_sample.ipynb
```

## DBCV, Noise, Largest Cluster, and Number of Clusters

```text
notebooks/06d_vis_cluster.ipynb
```

## Sample vs. Full Dataset

```text
notebooks/06e_sample_vs_full_KDist.ipynb
```

# Visualisations for the Thesis

```text
notebooks/paper_vis
```



