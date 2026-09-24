# semanticscholar.py
QUERY = '"narrative"'
YEAR_RANGE = "2015-2025" # to 2018 for over 100.000 results
FIELDS_SEMSCHO = "title,year,authors,citationCount,abstract,fieldsOfStudy,url,publicationTypes,venue,externalIds"
RAW_DATA_PATH = "data/raw/"

# filter.py
PROCESSED_DATA_PATH = "data/processed"

# enrich_doi.py
ENRICHED_DATA_PATH = "data/enriched"

# paper2mentions.py
MENTIONS_PATH = "data/mentions"


RESULTS_PATH = "results/"
RESULTS_PATH_FILTERED = "results/filtered"

#FINAL_PATH = "data/final.parquet"
#OUTPUT_PATH = "data/processed/narrative_papers_meta.csv"