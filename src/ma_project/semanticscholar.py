import config
import keys
import argparse
import requests
import time
import json
from pathlib import Path
from datetime import datetime, timezone

def parse_args():
    parser = argparse.ArgumentParser(description="Fetch papers from Semantic Scholar.")
    parser.add_argument("--query", type=str, default=None, help="Search query - attention: Semantic Scholar by default uses a stemmer (overrides config)")
    parser.add_argument("--output-dir",type=str, default=None, help=f"Output directory (overrides config) [current directory: {Path.cwd()}]")
    parser.add_argument("--year-range", type=str, default=None, help="Year range e.g. 2015-2025 (overrides config)")
    parser.add_argument("--verbose", type=str, default=True, help="Monitor the process with print statements")
    return parser.parse_args()

def stream_papers_with_meta(query, year_range=None, fields=None, sleep=1, max_results=None, verbose=True):
    """
    Streams papers from Semantic Scholar bulk search (all languages).
    Captures run metadata (total found, fetched) via StopIteration.value.

    Yields:
        dict: paper metadata

    Returns via StopIteration.value:
        dict: run metadata with keys total_found, total_fetched, duration_seconds
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

        if total_found is None:
            total_found = data.get("total")
            if verbose:
                print(f"Total papers: {total_found}")

        papers = data.get("data", [])
        if not papers:
            break

        for paper in papers:
            yield paper
            total_fetched += 1

            if max_results and total_fetched >= max_results:
                if verbose:
                    print("Reached max_results limit.")
                duration = time.monotonic() - start_time
                return {
                    "total_found": total_found,
                    "total_fetched": total_fetched,
                    "duration_seconds": round(duration, 2),
                }

        if verbose:
            print(f"Fetched {total_fetched} papers so far...")

        next_token = data.get("token")
        if next_token:
            last_valid_token = next_token  # remember the last good one
            token = next_token
            retries = 0
        else:
            token = None  # triggers retry logic, but last_valid_token is preserved
        print(f"Next token: {next_token}")  # temporary debug
        
        if not next_token:
            # Only break if we've actually fetched everything
            if total_found is not None and total_fetched < total_found:
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
    Appends a single run record to a JSONL log file.

    Args:
        log_path (str | Path): path to the log file
        params (dict): the API call parameters used for this run
        meta (dict): runtime metadata
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
    Streams all papers (unfiltered) and writes them to batched .jsonl files.
    Every `batch_size` papers, a new file is started.
    Logs run metadata (parameters, timing, paper counts) to a JSONL log file.

    Language filtering and other quality filtering is deferred to a later pipeline step.

    Args:
        query (str): search query
        output_dir (str): directory to write batch files and log into
        year_range (str): e.g. "2016-2019"
        fields (str): fields to request
        sleep (float): delay between requests
        max_results (int): optional cap on total papers fetched
        batch_size (int): number of papers per file (default: 100,000)
        verbose (bool): print progress
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
            try:
                paper = next(gen)
                current_file.write(json.dumps(paper) + "\n")
                count_in_batch += 1

                if count_in_batch >= batch_size:
                    if verbose:
                        print(f"Batch full. Rotating file...")
                    current_file = open_next_batch()
                    count_in_batch = 0

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
        query=config.QUERY,
        output_dir=config.RAW_DATA_PATH,
        year_range=config.YEAR_RANGE,
        fields=config.FIELDS_SEMSCHO,
        sleep=1,
        max_results=None,
        batch_size=100_000,
        verbose=verbose
    )