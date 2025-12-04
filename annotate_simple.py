import os
import sys
import json
import argparse
from dotenv import load_dotenv
from LLMCommitAnnotator import LLMCommitAnnotator

load_dotenv()

# Parse command-line arguments
parser = argparse.ArgumentParser(
    description="Annotate a single git commit using an LLM",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Examples:
  python annotate_simple.py data/sample-commits/commit.json
  python annotate_simple.py data/sample-commits/commit.json --model "openai/gpt-4"
  python annotate_simple.py data/sample-commits/commit.json --model "gemini-2.0-flash-exp" --temperature 0.2
  python annotate_simple.py data/sample-commits/commit.json --context-mode message+diff
"""
)

parser.add_argument(
    "commit_file",
    help="Path to the JSON file containing commit data"
)
parser.add_argument(
    "--model",
    default="openai/gpt-oss-20b:free",
    help="LLM model to use (default: openai/gpt-oss-20b:free)"
)
parser.add_argument(
    "--temperature",
    type=float,
    default=0.0,
    help="Sampling temperature (default: 0.0)"
)
parser.add_argument(
    "--max-tokens",
    type=int,
    default=3072,
    help="Maximum tokens for LLM response (default: 3072)"
)
parser.add_argument(
    "--context-mode",
    choices=["message", "message+diff"],
    default="message",
    help="Context to include in annotation (default: message). message+diff includes diff, stats, and modified files."
)

args = parser.parse_args()

# Read commit data from file
try:
  with open(args.commit_file, "r") as f:
    commit_data = json.load(f)
except FileNotFoundError:
  print(f"ERROR: Commit file not found: {args.commit_file}", file=sys.stderr)
  sys.exit(1)
except json.JSONDecodeError as e:
  print(f"ERROR: Invalid JSON in commit file: {e}", file=sys.stderr)
  sys.exit(1)

# Initialize the annotator
try:
  annotator = LLMCommitAnnotator(
      model=args.model,
      temperature=args.temperature,
      max_tokens=args.max_tokens,
      context_mode=args.context_mode
  )
except ValueError as e:
  print(f"ERROR: {e}", file=sys.stderr)
  sys.exit(1)
except FileNotFoundError as e:
  print(f"ERROR: {e}", file=sys.stderr)
  sys.exit(1)

# Annotate the commit
try:
  result = annotator.annotate_commit(commit_data)
  
  # Create output directory structure based on model name
  # Convert "meta-llama/llama-4-maverick:free" to "meta-llama_llama-4-maverick_free"
  model_folder = result["model"].replace("/", "_").replace(":", "_")
  output_dir = os.path.join("output", model_folder)
  os.makedirs(output_dir, exist_ok=True)
  
  # Save result as JSON file named {commit_hash}.json
  output_file = os.path.join(output_dir, f"{result['commit_hash']}.json")
  with open(output_file, "w") as f:
    json.dump(result, f, indent=2, ensure_ascii=False)
  
  print(f"Annotation saved to: {output_file}")
  print(f"Commit: {result['commit_hash']}")
  print(f"Timestamp: {result['timestamp']}")
  print(f"Model: {result['model']}")
  print(f"Usage: {result['usage_metadata']}", file=sys.stderr)
  
except Exception as e:
  print(f"ERROR: Failed to annotate commit: {e}", file=sys.stderr)
  sys.exit(1)
