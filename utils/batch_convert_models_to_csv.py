#!/usr/bin/env python3
"""Script to batch convert all LLM model annotations from JSON to CSV format.

This script processes all model folders in the output directory structure,
maintaining the round organization (r1, r2, etc.) and converting each model's
JSON annotations to CSV format in the corresponding llm-annotator-results folder.

Directory structure:
  output/
    r1/
      ollama_model1/
      ollama_model2/
    r2/
      ollama_model3/
      
Output structure:
  data/llm-annotator-results/
    r1/
      annotations_ollama_model1.csv
      annotations_ollama_model2.csv
    r2/
      annotations_ollama_model3.csv

Usage:
    python batch_convert_models_to_csv.py [--output-base <dir>] [--force]
    
Options:
    --output-base DIR   Base output directory (default: data/llm-annotator-results)
    --force            Force re-conversion of existing CSV files
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Tuple

# Import the conversion function from the existing script
# Add parent directory to path to import convert_model_to_csv
sys.path.insert(0, str(Path(__file__).parent))
from convert_model_to_csv import convert_model_folder_to_csv


def find_model_folders(base_dir: str = "output") -> List[Tuple[str, str, str]]:
    """
    Find all model folders in the output directory structure.
    
    Args:
        base_dir: Base directory to search (default: "output")
        
    Returns:
        List of tuples: (round_name, model_name, full_path)
    """
    base_path = Path(base_dir)
    
    if not base_path.exists():
        print(f"ERROR: Output directory not found: {base_dir}", file=sys.stderr)
        sys.exit(1)
    
    model_folders = []
    
    # Iterate through round folders (r1, r2, etc.)
    for round_folder in sorted(base_path.iterdir()):
        if not round_folder.is_dir():
            continue
        
        round_name = round_folder.name
        
        # Skip non-round folders
        if not round_name.startswith('r'):
            continue
        
        # Iterate through model folders within each round
        for model_folder in sorted(round_folder.iterdir()):
            if not model_folder.is_dir():
                continue
            
            model_name = model_folder.name
            full_path = str(model_folder)
            
            model_folders.append((round_name, model_name, full_path))
    
    return model_folders


def batch_convert_models(output_base: str = "data/llm-annotator-results", 
                         force: bool = False) -> None:
    """
    Convert all model folders to CSV format.
    
    Args:
        output_base: Base output directory for CSV files
        force: If True, overwrite existing CSV files
    """
    # Find all model folders
    model_folders = find_model_folders()
    
    if not model_folders:
        print("No model folders found in output/", file=sys.stderr)
        sys.exit(1)
    
    print(f"Found {len(model_folders)} model folders to process\n")
    print("="*80)
    
    # Statistics
    converted = 0
    skipped = 0
    failed = 0
    
    # Process each model folder
    for round_name, model_name, model_path in model_folders:
        # Determine output path
        output_dir = Path(output_base) / round_name
        output_dir.mkdir(parents=True, exist_ok=True)
        output_csv = output_dir / f"annotations_{model_name}.csv"
        
        # Check if output already exists
        if output_csv.exists() and not force:
            print(f"⊙ SKIPPED: {round_name}/{model_name}")
            print(f"  Output already exists: {output_csv}")
            print(f"  (use --force to overwrite)\n")
            skipped += 1
            continue
        
        # Convert model folder to CSV
        print(f"→ CONVERTING: {round_name}/{model_name}")
        print(f"  Input:  {model_path}")
        print(f"  Output: {output_csv}")
        
        try:
            convert_model_folder_to_csv(model_path, str(output_csv))
            print(f"✓ SUCCESS\n")
            converted += 1
        except Exception as e:
            print(f"✗ FAILED: {e}\n", file=sys.stderr)
            failed += 1
    
    # Print summary
    print("="*80)
    print("BATCH CONVERSION SUMMARY")
    print("="*80)
    print(f"Total models:     {len(model_folders)}")
    print(f"Converted:        {converted}")
    print(f"Skipped:          {skipped}")
    print(f"Failed:           {failed}")
    print("="*80)
    
    if converted > 0:
        print(f"\n✓ CSV files saved to: {output_base}/")
    
    # Exit with error code if any failed
    if failed > 0:
        sys.exit(1)


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Batch convert all LLM model annotations from JSON to CSV format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (skip existing files)
  python batch_convert_models_to_csv.py
  
  # Force re-conversion of all files
  python batch_convert_models_to_csv.py --force
  
  # Custom output directory
  python batch_convert_models_to_csv.py --output-base results/csv-exports
"""
    )
    
    parser.add_argument(
        "--output-base",
        default="data/llm-annotator-results",
        help="Base output directory for CSV files (default: data/llm-annotator-results)"
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-conversion of existing CSV files"
    )
    
    args = parser.parse_args()
    
    batch_convert_models(
        output_base=args.output_base,
        force=args.force
    )


if __name__ == "__main__":
    main()
