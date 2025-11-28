"""
Script to convert LLM model annotation results from JSON files to CSV format.
Transforms the output folder structure (output/model_name/) into a CSV file
compatible with the annotator results format.

Usage:
    python convert_model_to_csv.py <model_folder>
    
    model_folder: Path to the model output folder (e.g., output/ollama_deepseek-r1_14b)
"""

import os
import sys
import json
import csv
from pathlib import Path

# Hash length for commit identifiers
HASH_LENGTH = 10


def json_to_csv_row(json_file: Path, model_name: str) -> dict:
    """
    Convert a single JSON annotation file to a CSV row dictionary.
    
    Args:
        json_file: Path to JSON file
        model_name: Name of the model (used as annotator)
        
    Returns:
        Dictionary with CSV row data
    """
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Extract scores from the nested structure
    understanding_score = data.get("understanding", {}).get("score", "")
    bfc_score = data.get("bfc", {}).get("score", "")
    bpc_score = data.get("bpc", {}).get("score", "")
    prc_score = data.get("prc", {}).get("score", "")
    nfc_score = data.get("nfc", {}).get("score", "")
    
    # Truncate hash to match reference format
    full_hash = data.get("commit_hash", "")
    truncated_hash = full_hash[:HASH_LENGTH] if full_hash else ""
    
    # Get elapsed time
    elapsed_time = data.get("elapsed_time_seconds", "")
    
    # Get purpose (understanding description)
    purpose = data.get("understanding", {}).get("description", "")
    
    # Create row with similar structure to annotations_A.csv
    row = {
        "hash": truncated_hash,
        "annotator": model_name,
        "understand": understanding_score,
        "purpose": purpose,
        "bfc": bfc_score,
        "bpc": bpc_score,
        "prc": prc_score,
        "nfc": nfc_score,
        "time": elapsed_time,
        # Fields that don't have equivalents in model output - leave empty
        "specification": "",
        "asc": "",
        "obvious": "",
        "safety": "",
        "timing": "",
        "memory": "",
        "info": "",
        "safety_exp": "",
        "see_commit_clicked": "",
        "lore_clicked": "",
        "lore_founded": "",
        "is_merge_commit": "",
        "is_part_patchset": "",
        "lore_found": ""
    }
    
    return row


def convert_model_folder_to_csv(model_folder: str, output_csv: str = None) -> None:
    """
    Convert all JSON files in a model folder to a CSV file.
    
    Args:
        model_folder: Path to folder containing JSON annotation files
        output_csv: Optional path for output CSV. If not provided, will be generated
                   as annotations_{model_name}.csv in data/annotator-results/
    """
    model_folder_path = Path(model_folder)
    
    if not model_folder_path.exists():
        print(f"ERROR: Model folder not found: {model_folder}", file=sys.stderr)
        sys.exit(1)
    
    # Extract model name from folder path
    model_name = model_folder_path.name
    
    # Determine output CSV path
    if output_csv is None:
        output_dir = Path("data/annotator-results")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_csv = output_dir / f"annotations_{model_name}.csv"
    
    # Collect all JSON files
    json_files = sorted(model_folder_path.glob("*.json"))
    
    if not json_files:
        print(f"WARNING: No JSON files found in {model_folder}", file=sys.stderr)
        return
    
    print(f"Found {len(json_files)} annotation files")
    print(f"Model name: {model_name}")
    print(f"Hash length: {HASH_LENGTH}")
    print(f"Output file: {output_csv}")
    
    # Convert all JSON files to rows
    rows = []
    for json_file in json_files:
        try:
            row = json_to_csv_row(json_file, model_name)
            rows.append(row)
        except Exception as e:
            print(f"WARNING: Failed to process {json_file.name}: {e}", file=sys.stderr)
    
    # Write CSV file
    if rows:
        fieldnames = [
            "hash", "annotator", "understand", "purpose", "bfc", "bpc", "prc", "nfc",
            "specification", "asc", "obvious", "safety", "timing", "memory", "info",
            "safety_exp", "time", "see_commit_clicked", "lore_clicked", "lore_founded",
            "is_merge_commit", "is_part_patchset", "lore_found"
        ]
        
        with open(output_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        
        print(f"\n✓ Successfully converted {len(rows)} annotations to {output_csv}")
    else:
        print("ERROR: No rows to write", file=sys.stderr)
        sys.exit(1)


def main():
    """Main execution function."""
    if len(sys.argv) != 2:
        print("Usage: python convert_model_to_csv.py <model_folder>", file=sys.stderr)
        print("  Example: python convert_model_to_csv.py output/ollama_deepseek-r1_14b", file=sys.stderr)
        sys.exit(1)
    
    model_folder = sys.argv[1]
    convert_model_folder_to_csv(model_folder)


if __name__ == "__main__":
    main()
