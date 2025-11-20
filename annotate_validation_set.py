"""
Script to annotate all commits in the validation set using LLMCommitAnnotator.
Supports parallel processing and handles rate limit errors with retries.

Usage:
    python annotate_validation_set.py [model_name]
    
    model_name: Optional. LLM model to use (default: meta-llama/llama-4-maverick:free)
                Examples:
                - meta-llama/llama-4-maverick:free
                - deepseek/deepseek-chat-v3.1:free
                - google/gemini-2.0-flash-exp:free
"""

import os
import sys
import json
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from dotenv import load_dotenv
from LLMCommitAnnotator import LLMCommitAnnotator

# ANSI color codes
COLOR_ORANGE = "\033[38;5;214m"
COLOR_GREEN = "\033[32m"
COLOR_RED = "\033[31m"
COLOR_GRAY = "\033[90m"
COLOR_RESET = "\033[0m"

# Global lock for thread-safe printing
print_lock = Lock()

# Configuration
INPUT_FILE = "data/50-random-commits-validation.jsonl"
MAX_WORKERS = 10  # Maximum parallel annotations
RETRY_DELAY = 90  # Seconds to wait on rate limit error
MAX_RETRIES = 3   # Maximum number of retries per commit
DEFAULT_MODEL = "ollama/gpt-oss:20b"

load_dotenv()


def save_annotation(result: dict, model: str) -> str:
    """
    Save annotation result to JSON file.
    
    Args:
        result: Annotation result dictionary
        model: Model name used for annotation
        
    Returns:
        Path to the saved file
    """
    # Create output directory structure based on model name
    model_folder = model.replace("/", "_").replace(":", "_")
    output_dir = Path("output") / model_folder
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save result as JSON file named {commit_hash}.json
    output_file = output_dir / f"{result['commit_hash']}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    return str(output_file)


def check_annotation_exists(commit_hash: str, model: str) -> bool:
    """
    Check if annotation already exists for a commit and model.
    
    Args:
        commit_hash: The commit SHA hash
        model: Model name used for annotation
        
    Returns:
        True if annotation file exists, False otherwise
    """
    model_folder = model.replace("/", "_").replace(":", "_")
    output_file = Path("output") / model_folder / f"{commit_hash}.json"
    return output_file.exists()


def annotate_commit_with_retry(commit_data: dict, annotator: LLMCommitAnnotator, 
                                commit_index: int, total_commits: int) -> dict:
    """
    Annotate a single commit with retry logic for rate limits.
    
    Args:
        commit_data: Commit data dictionary
        annotator: LLMCommitAnnotator instance
        commit_index: Index of current commit (for logging)
        total_commits: Total number of commits (for logging)
        
    Returns:
        Dictionary with status and result/error information
    """
    commit_hash = commit_data["data"]["commit"]
    
    # Check if annotation already exists
    if check_annotation_exists(commit_hash, annotator.model):
        model_folder = annotator.model.replace("/", "_").replace(":", "_")
        output_file = str(Path("output") / model_folder / f"{commit_hash}.json")
        with print_lock:
            print(f"{COLOR_GRAY}[{commit_index}/{total_commits}] ⊙ Skipped {commit_hash[:8]} (already exists){COLOR_RESET}")
        
        return {
            "status": "skipped",
            "commit_hash": commit_hash,
            "output_file": output_file,
            "reason": "Annotation already exists"
        }
    
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            # Annotate the commit (silently)
            result = annotator.annotate_commit(commit_data)
            
            # Save the result
            output_file = save_annotation(result, annotator.model)
            
            # Print success in green
            with print_lock:
                print(f"{COLOR_GREEN}[{commit_index}/{total_commits}] ✓ Saved to {output_file}{COLOR_RESET}")
            
            return {
                "status": "success",
                "commit_hash": commit_hash,
                "output_file": output_file,
                "usage_metadata": result.get("usage_metadata")
            }
            
        except Exception as e:
            error_msg = str(e)
            
            # Check if it's a rate limit error
            is_rate_limit = any(keyword in error_msg.lower() 
                               for keyword in ["rate limit", "too many requests", "429"])
            
            if is_rate_limit and attempt < MAX_RETRIES:
                # Print warning in orange
                with print_lock:
                    print(f"{COLOR_ORANGE}[{commit_index}/{total_commits}] ⚠ Rate limit hit for {commit_hash[:8]}, "
                          f"waiting {RETRY_DELAY}s before retry {attempt + 1}/{MAX_RETRIES}...{COLOR_RESET}")
                time.sleep(RETRY_DELAY)
                continue
            
            # If not rate limit or max retries reached, print error in red
            with print_lock:
                print(f"{COLOR_RED}[{commit_index}/{total_commits}] ✗ Failed {commit_hash[:8]}: {error_msg}{COLOR_RESET}")
            
            return {
                "status": "error",
                "commit_hash": commit_hash,
                "error": error_msg,
                "attempts": attempt
            }
    
    # Should not reach here, but just in case
    return {
        "status": "error",
        "commit_hash": commit_hash,
        "error": "Max retries exceeded",
        "attempts": MAX_RETRIES
    }


def main():
    """Main execution function."""
    
    # Parse command line arguments
    if len(sys.argv) > 2:
        print("Usage: python annotate_validation_set.py [model_name]", file=sys.stderr)
        print(f"  model_name: Optional. Default is '{DEFAULT_MODEL}'", file=sys.stderr)
        sys.exit(1)
    
    model = sys.argv[1] if len(sys.argv) == 2 else DEFAULT_MODEL
    
    # Check if input file exists
    if not os.path.exists(INPUT_FILE):
        print(f"ERROR: Input file not found: {INPUT_FILE}", file=sys.stderr)
        sys.exit(1)
    
    # Read all commits from the validation set
    print(f"Reading commits from {INPUT_FILE}...")
    commits = []
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            commits.append(json.loads(line.strip()))
    
    print(f"Loaded {len(commits)} commits")
    
    # Initialize the annotator
    try:
        print(f"Initializing annotator with model: {model}")
        annotator = LLMCommitAnnotator(model=model)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Process commits in parallel
    print(f"\nStarting annotation with {MAX_WORKERS} parallel workers...")
    print(f"Rate limit retry delay: {RETRY_DELAY}s")
    print(f"Max retries per commit: {MAX_RETRIES}\n")
    
    results = []
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks
        futures = {
            executor.submit(
                annotate_commit_with_retry, 
                commit, 
                annotator, 
                idx + 1, 
                len(commits)
            ): commit 
            for idx, commit in enumerate(commits)
        }
        
        # Collect results as they complete
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
    
    # Calculate statistics
    elapsed_time = time.time() - start_time
    successful = sum(1 for r in results if r["status"] == "success")
    failed = sum(1 for r in results if r["status"] == "error")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    
    # Print summary
    print("\n" + "="*80)
    print("ANNOTATION SUMMARY")
    print("="*80)
    print(f"Total commits:    {len(commits)}")
    print(f"Successful:       {successful} ({successful/len(commits)*100:.1f}%)")
    print(f"Skipped:          {skipped} ({skipped/len(commits)*100:.1f}%)")
    print(f"Failed:           {failed} ({failed/len(commits)*100:.1f}%)")
    print(f"Total time:       {elapsed_time:.1f}s ({elapsed_time/60:.1f}m)")
    
    # Calculate average time only for processed commits (not skipped)
    processed = successful + failed
    if processed > 0:
        print(f"Avg time/commit:  {elapsed_time/processed:.1f}s (excluding skipped)")
    
    # Print failed commits if any
    if failed > 0:
        print("\nFailed commits:")
        for r in results:
            if r["status"] == "error":
                print(f"  - {r['commit_hash'][:12]}: {r['error']}")
    
    # Calculate total token usage if available
    total_input_tokens = 0
    total_output_tokens = 0
    for r in results:
        if r["status"] == "success" and r.get("usage_metadata"):
            metadata = r["usage_metadata"]
            total_input_tokens += metadata.get("input_tokens", 0)
            total_output_tokens += metadata.get("output_tokens", 0)
    
    if total_input_tokens > 0 or total_output_tokens > 0:
        print(f"\nToken usage:")
        print(f"  Input tokens:  {total_input_tokens:,}")
        print(f"  Output tokens: {total_output_tokens:,}")
        print(f"  Total tokens:  {total_input_tokens + total_output_tokens:,}")
    
    print("="*80)
    
    # Exit with error code if any failed
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
