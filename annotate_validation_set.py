"""Script to annotate all commits in the validation set using LLMCommitAnnotator.
Supports parallel processing and handles rate limit errors with retries.

Usage:
    python annotate_validation_set.py [options]
    
Options:
    --input FILE          Input JSONL file with commits (default: data/50-random-commits-validation.jsonl)
    --output DIR          Output directory for annotation results (default: output/)
    --model MODEL         LLM model to use (default: ollama/gpt-oss:20b)
    --temperature FLOAT   Sampling temperature (default: 0.0)
    --context-mode MODE   Context to include: message, message+diff, full (default: message)
    --max-tokens INT      Maximum response tokens (default: 3072)
    --workers INT         Number of parallel workers (default: 10)
    --retry-delay INT     Seconds to wait on rate limit (default: 90)
    --max-retries INT     Maximum retries per commit (default: 3)
"""

import os
import sys
import json
import time
import argparse
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

# Default configuration
DEFAULT_INPUT_FILE = "data/50-random-commits-validation.jsonl"
DEFAULT_OUTPUT_DIR = "output/"
DEFAULT_MODEL = "ollama/gpt-oss:20b"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_CONTEXT_MODE = "message"
DEFAULT_MAX_TOKENS = 3072
DEFAULT_MAX_WORKERS = 10
DEFAULT_RETRY_DELAY = 90
DEFAULT_MAX_RETRIES = 3

load_dotenv()


def save_annotation(result: dict, model: str, output_base_dir: str) -> str:
    """
    Save annotation result to JSON file.
    
    Args:
        result: Annotation result dictionary
        model: Model name used for annotation
        output_base_dir: Base output directory
        
    Returns:
        Path to the saved file
    """
    # Create output directory structure based on model name
    model_folder = model.replace("/", "_").replace(":", "_")
    output_dir = Path(output_base_dir) / model_folder
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save result as JSON file named {commit_hash}.json
    output_file = output_dir / f"{result['commit_hash']}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    return str(output_file)


def check_annotation_exists(commit_hash: str, model: str, output_base_dir: str) -> bool:
    """
    Check if annotation already exists for a commit and model.
    
    Args:
        commit_hash: The commit SHA hash
        model: Model name used for annotation
        output_base_dir: Base output directory
        
    Returns:
        True if annotation file exists, False otherwise
    """
    model_folder = model.replace("/", "_").replace(":", "_")
    output_file = Path(output_base_dir) / model_folder / f"{commit_hash}.json"
    return output_file.exists()


def annotate_commit_with_retry(commit_data: dict, annotator: LLMCommitAnnotator, 
                                commit_index: int, total_commits: int,
                                output_base_dir: str, max_retries: int, 
                                retry_delay: int) -> dict:
    """
    Annotate a single commit with retry logic for rate limits.
    
    Args:
        commit_data: Commit data dictionary
        annotator: LLMCommitAnnotator instance
        commit_index: Index of current commit (for logging)
        total_commits: Total number of commits (for logging)
        output_base_dir: Base output directory
        max_retries: Maximum number of retries per commit
        retry_delay: Seconds to wait on rate limit error
        
    Returns:
        Dictionary with status and result/error information
    """
    commit_hash = commit_data["data"]["commit"]
    
    # Check if annotation already exists
    if check_annotation_exists(commit_hash, annotator.model, output_base_dir):
        model_folder = annotator.model.replace("/", "_").replace(":", "_")
        output_file = str(Path(output_base_dir) / model_folder / f"{commit_hash}.json")
        with print_lock:
            print(f"{COLOR_GRAY}[{commit_index}/{total_commits}] ⊙ Skipped {commit_hash[:8]} (already exists){COLOR_RESET}")
        
        return {
            "status": "skipped",
            "commit_hash": commit_hash,
            "output_file": output_file,
            "reason": "Annotation already exists"
        }
    
    for attempt in range(1, max_retries + 1):
        try:
            # Annotate the commit (silently)
            result = annotator.annotate_commit(commit_data)
            
            # Save the result
            output_file = save_annotation(result, annotator.model, output_base_dir)
            
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
            
            if is_rate_limit and attempt < max_retries:
                # Print warning in orange
                with print_lock:
                    print(f"{COLOR_ORANGE}[{commit_index}/{total_commits}] ⚠ Rate limit hit for {commit_hash[:8]}, "
                          f"waiting {retry_delay}s before retry {attempt + 1}/{max_retries}...{COLOR_RESET}")
                time.sleep(retry_delay)
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
        "attempts": max_retries
    }


def main():
    """Main execution function."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Annotate commits using LLMs with parallel processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with defaults
  python annotate_validation_set.py
  
  # Custom model and input file
  python annotate_validation_set.py --model "google/gemini-2.0-flash-exp" --input data/my-commits.jsonl
  
  # With diff context
  python annotate_validation_set.py --context-mode message+diff --input data/commits-with-diff.jsonl
  
  # Custom output directory
  python annotate_validation_set.py --output results/my-experiment/
"""
    )
    
    parser.add_argument(
        "--input",
        type=str,
        default=DEFAULT_INPUT_FILE,
        help=f"Input JSONL file with commits (default: {DEFAULT_INPUT_FILE})"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for annotation results (default: {DEFAULT_OUTPUT_DIR})"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"LLM model to use (default: {DEFAULT_MODEL})"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help=f"Sampling temperature (default: {DEFAULT_TEMPERATURE})"
    )
    
    parser.add_argument(
        "--context-mode",
        type=str,
        choices=["message", "message+diff", "full"],
        default=DEFAULT_CONTEXT_MODE,
        help=f"Context to include in prompts (default: {DEFAULT_CONTEXT_MODE})"
    )
    
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=f"Maximum response tokens (default: {DEFAULT_MAX_TOKENS})"
    )
    
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help=f"Number of parallel workers (default: {DEFAULT_MAX_WORKERS})"
    )
    
    parser.add_argument(
        "--retry-delay",
        type=int,
        default=DEFAULT_RETRY_DELAY,
        help=f"Seconds to wait on rate limit (default: {DEFAULT_RETRY_DELAY})"
    )
    
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help=f"Maximum retries per commit (default: {DEFAULT_MAX_RETRIES})"
    )
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"ERROR: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    
    # Read all commits from the validation set
    print(f"Reading commits from {args.input}...")
    commits = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            commits.append(json.loads(line.strip()))
    
    print(f"Loaded {len(commits)} commits")
    
    # Initialize the annotator
    try:
        print(f"Initializing annotator with model: {args.model}")
        annotator = LLMCommitAnnotator(
            model=args.model,
            temperature=args.temperature,
            context_mode=args.context_mode,
            max_tokens=args.max_tokens
        )
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Create output directory
    Path(args.output).mkdir(parents=True, exist_ok=True)
    
    # Process commits in parallel
    print(f"\nConfiguration:")
    print(f"  Model: {args.model}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Context mode: {args.context_mode}")
    print(f"  Max tokens: {args.max_tokens}")
    print(f"  Output directory: {args.output}")
    print(f"  Parallel workers: {args.workers}")
    print(f"  Rate limit retry delay: {args.retry_delay}s")
    print(f"  Max retries per commit: {args.max_retries}\n")
    
    results = []
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(
                annotate_commit_with_retry, 
                commit, 
                annotator, 
                idx + 1, 
                len(commits),
                args.output,
                args.max_retries,
                args.retry_delay
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
