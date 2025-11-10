import os
import sys
import json
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
  
except FileNotFoundError:
  print(f"ERROR: Commit file not found: {commit_file}", file=sys.stderr)
  sys.exit(1)
except Exception as e:
  print(f"ERROR: Failed to annotate commit: {e}", file=sys.stderr)
  sys.exit(1)
