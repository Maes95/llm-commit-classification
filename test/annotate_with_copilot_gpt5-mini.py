import os
import sys
import json
from dotenv import load_dotenv

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from LLMCommitAnnotator import LLMCommitAnnotator

load_dotenv()

COMMIT_FILE = "data/sample-commits/734-0c7833b9e86d61cdfe44c2af17dcf8a08ba0ee61.json"
MODEL = "copilot/gpt-5-mini"
TEMPERATURE = 0.0
MAX_TOKENS = 10000
CONTEXT_MODE = "message"

if not MODEL.startswith("copilot/"):
    print(
        f"ERROR: Copilot test expects models with 'copilot/' prefix. Got: {MODEL}",
        file=sys.stderr,
    )
    sys.exit(1)

# Read commit data from file
try:
    with open(COMMIT_FILE, "r") as f:
        commit_data = json.load(f)
except FileNotFoundError:
    print(f"ERROR: Commit file not found: {COMMIT_FILE}", file=sys.stderr)
    sys.exit(1)
except json.JSONDecodeError as e:
    print(f"ERROR: Invalid JSON in commit file: {e}", file=sys.stderr)
    sys.exit(1)

# Initialize the annotator
try:
    annotator = LLMCommitAnnotator(
        model=MODEL,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        context_mode=CONTEXT_MODE,
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

    # Save output under output/<model_name>/<commit_hash>.json
    model_folder = result["model"].replace("/", "_").replace(":", "_")
    output_dir = os.path.join("output", model_folder)
    os.makedirs(output_dir, exist_ok=True)

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
