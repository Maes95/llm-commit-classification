import os
import sys
from dotenv import load_dotenv
from LLMCommitAnnotator import LLMCommitAnnotator

load_dotenv()

if len(sys.argv) < 2:
  print("ERROR: Missing required argument.", file=sys.stderr)
  print("Usage: python script.py <commit_file>", file=sys.stderr)
  sys.exit(1)

commit_file = sys.argv[1]

# Initialize the annotator
try:
  annotator = LLMCommitAnnotator(model="meta-llama/llama-4-maverick:free")
except ValueError as e:
  print(f"ERROR: {e}", file=sys.stderr)
  sys.exit(1)
except FileNotFoundError as e:
  print(f"ERROR: {e}", file=sys.stderr)
  sys.exit(1)

# Annotate the commit
try:
  result = annotator.annotate_commit_from_file(commit_file)
  
  # Print the classification result
  print(result["classification"])
  print(f"\nUsage: {result['usage_metadata']}", file=sys.stderr)
  print(f"Model: {result['model']}", file=sys.stderr)
  
except FileNotFoundError:
  print(f"ERROR: Commit file not found: {commit_file}", file=sys.stderr)
  sys.exit(1)
except Exception as e:
  print(f"ERROR: Failed to annotate commit: {e}", file=sys.stderr)
  sys.exit(1)
