"""
Pre-process JSONL files to add diff information.

This approach enriches existing JSONL files with diffs, so you only need
to fetch them once. Best for: Batch processing, repeated annotations.
"""

import json
import subprocess
from pathlib import Path
from typing import Optional, Callable
import sys


def get_diff_via_git(commit_hash: str, repo_path: str) -> Optional[str]:
    """
    Get diff using git subprocess.
    
    Args:
        commit_hash: The SHA hash of the commit
        repo_path: Path to the git repository
        
    Returns:
        The commit diff or None if error occurs
    """
    try:
        result = subprocess.run(
            ["git", "show", "--no-color", "--format=", commit_hash],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True,
            timeout=30
        )
        return result.stdout
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        print(f"Error getting diff for {commit_hash}: {e}", file=sys.stderr)
        return None


def get_stats_via_git(commit_hash: str, repo_path: str) -> Optional[str]:
    """
    Get commit stats using git subprocess.
    
    Args:
        commit_hash: The SHA hash of the commit
        repo_path: Path to the git repository
        
    Returns:
        The commit stats or None if error occurs
    """
    try:
        result = subprocess.run(
            ["git", "show", "--no-color", "--stat", "--format=", commit_hash],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True,
            timeout=30
        )
        return result.stdout
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        print(f"Error getting stats for {commit_hash}: {e}", file=sys.stderr)
        return None


def enrich_jsonl_with_diffs(
    input_file: str,
    output_file: str,
    repo_path: str,
    include_diff: bool = True,
    include_stats: bool = True,
    diff_retriever: Optional[Callable] = None,
    verbose: bool = True
) -> dict:
    """
    Enrich a JSONL file with git diff information.
    
    Args:
        input_file: Path to input JSONL file
        output_file: Path to output JSONL file
        repo_path: Path to git repository
        include_diff: Whether to include full diff (default: True)
        include_stats: Whether to include commit stats (default: True)
        diff_retriever: Custom function to retrieve diffs (optional)
        verbose: Print progress (default: True)
        
    Returns:
        Dictionary with processing statistics
    """
    input_path = Path(input_file)
    output_path = Path(output_file)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    stats = {
        "total": 0,
        "enriched": 0,
        "failed": 0,
        "skipped": 0
    }
    
    with open(input_path, 'r') as fin, open(output_path, 'w') as fout:
        for line_num, line in enumerate(fin, 1):
            stats["total"] += 1
            
            try:
                data = json.loads(line.strip())
                
                # Check if already enriched
                if 'diff' in data.get('data', {}):
                    if verbose:
                        print(f"Line {line_num}: Already has diff, skipping")
                    stats["skipped"] += 1
                    fout.write(line)
                    continue
                
                commit_hash = data['data']['commit']
                
                if verbose:
                    print(f"Line {line_num}: Processing {commit_hash[:8]}...")
                
                # Get diff
                if include_diff:
                    if diff_retriever:
                        diff = diff_retriever(commit_hash, repo_path)
                    else:
                        diff = get_diff_via_git(commit_hash, repo_path)
                    
                    if diff is not None:
                        data['data']['diff'] = diff
                    else:
                        stats["failed"] += 1
                        if verbose:
                            print(f"Line {line_num}: Failed to get diff")
                
                # Get stats
                if include_stats:
                    stats_text = get_stats_via_git(commit_hash, repo_path)
                    if stats_text is not None:
                        data['data']['stats'] = stats_text
                
                # Write enriched data
                fout.write(json.dumps(data) + '\n')
                stats["enriched"] += 1
                
            except json.JSONDecodeError as e:
                print(f"Line {line_num}: JSON decode error: {e}", file=sys.stderr)
                stats["failed"] += 1
                fout.write(line)  # Write original line
                
            except KeyError as e:
                print(f"Line {line_num}: Missing key: {e}", file=sys.stderr)
                stats["failed"] += 1
                fout.write(line)  # Write original line
    
    if verbose:
        print("\n=== Processing Summary ===")
        print(f"Total lines: {stats['total']}")
        print(f"Successfully enriched: {stats['enriched']}")
        print(f"Already had diffs: {stats['skipped']}")
        print(f"Failed: {stats['failed']}")
    
    return stats

# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Enrich JSONL file with git diffs")
    parser.add_argument("input_file", help="Input JSONL file")
    parser.add_argument("output_file", help="Output JSONL file")
    parser.add_argument("--repo", required=True, help="Path to git repository")
    parser.add_argument("--no-diff", action="store_true", help="Don't include diff")
    parser.add_argument("--no-stats", action="store_true", help="Don't include stats")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")
    
    args = parser.parse_args()
    
    stats = enrich_jsonl_with_diffs(
        input_file=args.input_file,
        output_file=args.output_file,
        repo_path=args.repo,
        include_diff=not args.no_diff,
        include_stats=not args.no_stats,
        verbose=not args.quiet
    )
    
    sys.exit(0 if stats["failed"] == 0 else 1)
