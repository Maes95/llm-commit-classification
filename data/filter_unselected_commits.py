#!/usr/bin/env python3
"""
Script to obtain commits that were NOT randomly selected.

This script reads the file with 1000 commits and filters those that are not
in the file with 50 random commits, nor in the list of additional commits.
"""

import json
import argparse


def read_commits(jsonl_file, exclude_merge=True):
    """Read a JSONL file and return a list of commits.
    
    Args:
        jsonl_file: Path to the JSONL file
        exclude_merge: If True, filter out commits with 'Merge' field (default: True)
    """
    commits = []
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                commit = json.loads(line)
                # Filter out merge commits if requested
                if exclude_merge and 'Merge' in commit.get('data', {}):
                    continue
                commits.append(commit)
    return commits


def extract_commit_ids(commits):
    """Extract commit IDs from a list of commits."""
    return set(commit['data']['commit'] for commit in commits)


def filter_unselected_commits(all_commits, selected_ids):
    """Filter commits that are NOT in the set of selected IDs."""
    return [commit for commit in all_commits 
            if commit['data']['commit'] not in selected_ids]


def save_commits(commits, output_file):
    """Save commits to a JSONL file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        for commit in commits:
            f.write(json.dumps(commit, ensure_ascii=False) + '\n')


def main():
    parser = argparse.ArgumentParser(
        description='Filter unselected commits from the set of 1000 commits'
    )
    parser.add_argument(
        '--all-commits',
        default='1000-linux-commits.jsonl',
        help='File with all commits (default: 1000-linux-commits.jsonl)'
    )
    parser.add_argument(
        '--selected-commits',
        default='50-random-commits-validation.jsonl',
        help='File with selected commits (default: 50-random-commits-validation.jsonl)'
    )
    parser.add_argument(
        '--additional-ids',
        nargs='*',
        default=[],
        help='List of additional commit IDs to exclude (space-separated)'
    )
    parser.add_argument(
        '--output',
        default='unselected-commits.jsonl',
        help='Output file (default: unselected-commits.jsonl)'
    )
    
    args = parser.parse_args()
    
    print(f"Reading all commits from: {args.all_commits}")
    all_commits = read_commits(args.all_commits)
    print(f"  Total commits (excluding merge commits): {len(all_commits)}")
    
    print(f"\nReading selected commits from: {args.selected_commits}")
    selected_commits = read_commits(args.selected_commits)
    print(f"  Total selected commits (excluding merge commits): {len(selected_commits)}")
    
    # Get selected commit IDs
    selected_ids = extract_commit_ids(selected_commits)
    
    # Add additional IDs if provided
    if args.additional_ids:
        print(f"\nAdding {len(args.additional_ids)} additional IDs:")
        for commit_id in args.additional_ids:
            print(f"  - {commit_id}")
            selected_ids.add(commit_id)
    
    print(f"\nTotal IDs to exclude: {len(selected_ids)}")
    
    # Filter unselected commits
    unselected_commits = filter_unselected_commits(all_commits, selected_ids)
    print(f"Unselected commits: {len(unselected_commits)}")
    
    # Save result
    save_commits(unselected_commits, args.output)
    print(f"\n✓ Unselected commits saved to: {args.output}")
    print(f"  Total: {len(unselected_commits)} commits")


if __name__ == '__main__':
    main()

# python filter_unselected_commits.py --additional-ids d11a327ed95dbec756b99cbfef2a7fd85c9eeb09 1eba86c096e35e3cc83de1ad2c26f2d70470211b 9a10064f5625d5572c3626c1516e0bebc6c9fe9b