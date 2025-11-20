"""
Script to randomly sample 50 commits from the full dataset for validation purposes.
Uses a fixed random seed for reproducibility.
"""

import json
import random

# Configuration
INPUT_FILE = "1000-linux-commits.jsonl"
OUTPUT_FILE = "50-random-commits-validation.jsonl"
SAMPLE_SIZE = 50
RANDOM_SEED = 42  # Fixed seed for reproducibility

def main():
    # Set random seed for reproducibility
    random.seed(RANDOM_SEED)
    
    # Read all commits from the input file
    # Filter out commits that have the 'Merge' field
    commits = []
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            commit = json.loads(line.strip())
            if 'Merge' not in commit.get('data', {}):
                commits.append(commit)
    
    print(f"Total commits in dataset (excluding merge commits): {len(commits)}")
    
    # Randomly sample 50 commits
    sampled_commits = random.sample(commits, SAMPLE_SIZE)
    
    print(f"Sampled {len(sampled_commits)} commits with seed {RANDOM_SEED}")
    
    # Write sampled commits to output file
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for commit in sampled_commits:
            f.write(json.dumps(commit, ensure_ascii=False) + "\n")
    
    print(f"Sampled commits saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
