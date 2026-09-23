"""
Semantic Scholar bulk paper retrieval pipeline.

This script queries the Semantic Scholar bulk search API and streams the returned papers to disk as JSON Lines (JSONL) files. 
Results are written incrementally in batches so that large searches can be processed without holding the complete result set in memory.

The pipeline performs the following steps:
1. Parse command-line arguments, with configuration-file values as fallbacks.
2. Query the Semantic Scholar bulk search endpoint.
3. Handle pagination using the API's continuation token.
4. Retry failed requests and temporarily missing continuation tokens.
5. Stream papers one at a time to batched JSONL files.
6. Record metadata about the run, including the number of papers found, the number fetched, and the total runtime.

The raw search results are intentionally kept unfiltered. 
Language filtering, quality checks, deduplication, and other preprocessing are deferred to later stages of the data-processing pipeline.

Output:
search_results_batch_1.jsonl
search_results_batch_2.jsonl
...
run_log.jsonl

The batch files contain one paper per line as a JSON object. 
The run log also uses JSONL format, with one record per execution.

Configuration:
Search parameters and API-specific settings can be supplied through
the imported config module. An optional Semantic Scholar API key is
read from keys.API_KEY_SEMSCHO.

Usage:
uv run python semanticscholar.py
uv run python semanticscholar.py --query "machine learning"
uv run python semanticscholar.py --year-range "2015-2025"
uv run python semanticscholar.py --output-dir "data/raw"

Notes:
- Semantic Scholar may apply stemming to search queries by default.
- API pagination depends on continuation tokens returned by the server.
- The script includes retry logic to handle transient API failures.
- Results are written incrementally to avoid excessive memory usage.
"""

import config
import keys
import argparse
import requests
import time
import json
from pathlib import Path
from datetime import datetime, timezone

def parse_args():
    """
    Parse command-line arguments used to configure the data retrieval run.

    Command-line arguments override corresponding values from the config
    module when the script is executed.
    
    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Fetch papers from Semantic Scholar.")
    parser.add_argument("--query", type=str, default=None, help="Search query - attention: Semantic Scholar by default uses a stemmer (overrides config)")
    parser.add_argument("--output-dir",type=str, default=None, help=f"Output directory (overrides config) [current directory: {Path.cwd()}]")
    parser.add_argument("--year-range", type=str, default=None, help="Year range e.g. 2015-2025 (overrides config)")
    #parser.add_argument("--verbose", type=str, default=True, help="Monitor the process with print statements")
    parser.add_argument("--verbose", action="store_true", help="Print progress information.")
    return parser.parse_args()

def stream_papers_with_meta(query, year_range=None, fields=None, sleep=1, max_results=None, verbose=True):
    """
    Stream papers from the Semantic Scholar bulk search API.

    Results are yielded one paper at a time, allowing callers to process
    large result sets without loading all papers into memory. The function
    handles API pagination, request retries, and missing continuation tokens.

    Args:
        query (str): Search query passed to Semantic Scholar.
        year_range (str | None): Optional publication year range, e.g. "2015-2025".
        fields (str | None): Comma-separated list of paper fields to request.
            Uses a default set of fields when omitted.
        sleep (float): Delay in seconds between successful API requests.
        max_results (int | None): Optional maximum number of papers to yield.
        verbose (bool): Whether to print progress and retry information.

    Yields:
        dict: Metadata for one paper returned by Semantic Scholar.

    Returns:
        dict: Run metadata containing:
            - total_found: Number of papers reported by the API.
            - total_fetched: Number of papers successfully fetched.
            - duration_seconds: Total execution time in seconds.

    Raises:
        RuntimeError: If an API request fails after the maximum number
            of retries.
    """
    total_found = None
    total_fetched = 0
    start_time = time.monotonic()
    retries = 0
    MAX_RETRIES = 5

    url = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"
    params = {
        "query": query,
        "fields": fields or "title,year,authors,citationCount,abstract,fieldsOfStudy,url,publicationTypes,venue",
    }
    headers = {"x-api-key": key} if (key := getattr(keys, "API_KEY_SEMSCHO", None)) else {}
    if year_range:
        params["year"] = year_range

    token = None
    last_valid_token = None


    while True:
        # Check for the current pagination token from Semantic Scholar
        if token:
            params["token"] = token
        elif last_valid_token:
            params["token"] = last_valid_token

        response = requests.get(url, params=params, headers=headers)
        
        if response.status_code != 200:
            if retries >= MAX_RETRIES:
                raise RuntimeError(f"Failed after {MAX_RETRIES} retries (last status: {response.status_code})")
            wait = 2 ** retries
            print(f"Error {response.status_code}, retrying in {wait}s...")
            time.sleep(wait)
            retries += 1
            continue
        retries = 0

        data = response.json()

        # Store the total results information returned from the API here
        # typically this information is contained in the first return (no pagination token used yet)
        if total_found is None:
            total_found = data.get("total")
            if verbose:
                print(f"Total papers: {total_found}")

        papers = data.get("data", [])

        # Stop While-Loop if there is no data in the API return
        if not papers:
            break

        for paper in papers:
            yield paper
            total_fetched += 1

            # Stop While-Loop if the number of papers processed exceeds the number of papers expected as per the API return information
            if max_results and total_fetched >= max_results:
                if verbose:
                    print("Reached max_results limit.")
                duration = time.monotonic() - start_time
                return {
                    "total_found": total_found,
                    "total_fetched": total_fetched,
                    "duration_seconds": round(duration, 2),
                }

        # Print Progress after each API call
        if verbose:
            print(f"Fetched {total_fetched} papers so far...")

        next_token = data.get("token")
        if next_token:
            last_valid_token = next_token  # remember the last good one
            token = next_token
            retries = 0
        else:
            token = None  # triggers retry logic, but last_valid_token is preserved
        
        # What to do if there is no next_token
        if not next_token:
            # Breaks if enough papers have been fetched (at least the amount estimated by the API return)
            if total_found is not None and total_fetched < total_found:
                # retries to fetch a valid next_token up to MAX_RETRIES
                if retries < MAX_RETRIES:
                    wait = 2 ** retries
                    print(f"No token but only fetched {total_fetched}/{total_found}. Retrying in {wait}s...")
                    time.sleep(wait)
                    retries += 1
                    # Don't update token — re-send the last token to get the next page
                    continue
                else:
                    print(f"Warning: gave up after {MAX_RETRIES} retries. Got {total_fetched}/{total_found} papers.")
                    break
            else:
                break

        time.sleep(sleep)

    duration = time.monotonic() - start_time
    return {
        "total_found": total_found,
        "total_fetched": total_fetched,
        "duration_seconds": round(duration, 2),
    }


def write_run_log(log_path, params, meta):
    """
    Append metadata for one pipeline run to a JSONL log file.

    Args:
        log_path (str | Path): Path to the JSONL log file.
        params (dict): Parameters used for the Semantic Scholar search.
        meta (dict): Metadata produced during the run.
    """
    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "parameters": params,
        **meta,
    }
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    print(f"Run logged to {log_path}")


def write_papers_in_batches(
    query,
    output_dir,
    year_range=None,
    fields=None,
    sleep=1,
    max_results=None,
    batch_size=100_000,
    verbose=True,
):
    """
    Fetch papers from Semantic Scholar and write them to batched JSONL files.

    Papers are streamed from the API and written incrementally, so the full
    result set does not need to be held in memory. A new output file is
    created after every `batch_size` papers.

    Language filtering, deduplication, and other quality filtering are
    intentionally deferred to later pipeline stages.

    Args:
        query (str): Search query passed to Semantic Scholar.
        output_dir (str | Path): Directory for batch files and the run log.
        year_range (str | None): Optional publication year range.
        fields (str | None): Fields requested from the Semantic Scholar API.
        sleep (float): Delay between API requests.
        max_results (int | None): Optional maximum number of papers to fetch.
        batch_size (int): Maximum number of papers written to each batch file.
        verbose (bool): Whether to print progress information.

    Outputs:
        Creates one or more JSONL batch files and a `run_log.jsonl` file
        inside `output_dir`.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_params = {
        "query": query,
        "year_range": year_range,
        "fields": fields,
        "sleep": sleep,
        "max_results": max_results,
        "batch_size": batch_size,
    }

    batch_index = 1
    count_in_batch = 0
    current_file = None

    def open_next_batch():
        # nonlocal means that they refer to variables OUTSIDE the function
        nonlocal batch_index, current_file
        if current_file:
            current_file.close()
        path = output_dir / f"search_results_batch_{batch_index}.jsonl"
        if verbose:
            print(f"Opening new batch file: {path}")
        current_file = open(path, "w", encoding="utf-8")
        batch_index += 1
        return current_file

    current_file = open_next_batch()
    gen = stream_papers_with_meta(
        query=query,
        year_range=year_range,
        fields=fields,
        sleep=sleep,
        max_results=max_results,
        verbose=verbose,
    )

    meta = {}
    try:
        while True:
            # runs as long as there is a next paper
            try:
                paper = next(gen)
                current_file.write(json.dumps(paper) + "\n")
                count_in_batch += 1

                # ensures the size of the batches
                if count_in_batch >= batch_size:
                    if verbose:
                        print(f"Batch full. Rotating file...")
                    current_file = open_next_batch()
                    count_in_batch = 0

            # Collects meta data if there is no next paper
            except StopIteration as e:
                meta = e.value or {}
                break
    finally:
        if current_file:
            current_file.close()

    if verbose:
        print(
            f"Done. {meta.get('total_fetched', '?')} papers written across {batch_index - 1} batch file(s) "
            f"in {meta.get('duration_seconds', '?')}s."
        )

    write_run_log(
        log_path=output_dir / "run_log.jsonl",
        params=run_params,
        meta=meta,
    )


if __name__ == "__main__":
    args = parse_args()

    query      = args.query      or getattr(config, "QUERY",      None)
    output_dir = args.output_dir or getattr(config, "RAW_DATA_PATH", "output")
    year_range = args.year_range or getattr(config, "YEAR_RANGE",  None)
    verbose    = args.verbose
    
    write_papers_in_batches(
        query=query, #config.QUERY,
        output_dir=output_dir, #config.RAW_DATA_PATH,
        year_range=year_range, #config.YEAR_RANGE,
        fields=config.FIELDS_SEMSCHO,
        sleep=1,
        max_results=None,
        batch_size=100_000,
        verbose=verbose
    )