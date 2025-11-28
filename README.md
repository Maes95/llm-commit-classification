# LLM Commit Classifier

A research tool for annotating git commits using Large Language Models (LLMs). The system classifies commits across multiple dimensions to evaluate their purpose and characteristics using a 5-point Likert scale (0-4).

## Commit Categories

The tool evaluates commits across four categories simultaneously:

- **Bug-Fixing Commit (BFC)**: Fixes a bug present in the source code that manifests as a failure
- **Bug-Preventing Commit (BPC)**: Prevents potential bugs that could cause failures in the future
- **Perfective Commit (PRC)**: Improves code quality (refactoring, optimization, style improvements, comments)
- **New Feature Commit (NFC)**: Adds new functionality or capabilities to the codebase

Each category is scored from 0 (not applicable) to 4 (primary characteristic).

See `documentation/definitions.md` for detailed definitions of each category.

## Project Structure

```
llm-commit-classification/
├── LLMCommitAnnotator.py       # Main annotation class
├── annotate_simple.py          # Annotate a single commit
├── annotate_validation_set.py  # Annotate multiple commits (batch)
├── llms/                       # LLM provider wrappers
│   ├── google_llm.py          # Google Gemini models
│   ├── openai_llm.py          # OpenAI GPT models
│   ├── openrouter_llm.py      # OpenRouter API
│   └── ollama_llm.py          # Local Ollama models
├── diffs/                      # Diff retrieval utilities
│   ├── git_subprocess.py      # Git via subprocess
│   ├── gitpython_diff.py      # Git via GitPython
│   ├── github_api.py          # GitHub API retrieval
│   └── enrich_jsonl.py        # Pre-process JSONL with diffs
├── utils/                      # Utility scripts
│   ├── convert_model_to_csv.py # Convert annotations to CSV
│   └── add_diff_to_jsonl.py   # Add diff context to data
├── data/                       # Commit data and annotations
└── documentation/              # Category definitions and methodology
```

## Installation

### Prerequisites

- Python 3.8 or higher
- Git (for diff retrieval)
- Optional: Local git repository for commit context

### Setup

1. Clone the repository:
```bash
git clone https://github.com/Maes95/llm-commit-classification.git
cd llm-commit-classification
```

2. Create and activate a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

4. Configure API keys:
```bash
cp dotenv-example .env
# Edit .env and add your API keys
```

### LLM Provider Configuration

The tool supports multiple LLM providers. Configure at least one:

#### OpenRouter (Recommended for variety)
1. Visit https://openrouter.ai/ and create an account
2. Generate an API key from the dashboard
3. Add to `.env`: `OPENROUTER_API_KEY=sk-or-v1-...`

#### Google Gemini
1. Get API key from https://makersuite.google.com/app/apikey
2. Add to `.env`: `GOOGLE_API_KEY=AIza...`

#### OpenAI
1. Get API key from https://platform.openai.com/api-keys
2. Add to `.env`: `OPENAI_API_KEY=sk-...`

#### Ollama (Local models)
1. Install Ollama from https://ollama.ai/
2. Pull models: `ollama pull llama3.1:8b`
3. No API key needed - runs locally

**Security**: Never commit `.env` to version control.

## Usage

### Annotate a Single Commit

Use `annotate_simple.py` to annotate one commit:

```bash
python annotate_simple.py data/sample-commits/724-7392d87a9febb5f46f28d4704eb5636c5e22cdeb.json
```

**With custom model:**
```bash
python annotate_simple.py \
    data/sample-commits/724-7392d87a9febb5f46f28d4704eb5636c5e22cdeb.json \
    --model "google/gemini-2.0-flash-exp"
```

**Output:** JSON with scores (0-4) for each category, reasoning, and metadata.

### Annotate Multiple Commits (Batch)

Use `annotate_validation_set.py` for batch annotation:

```bash
python annotate_validation_set.py \
    --input data/50-random-commits-validation.jsonl \
    --output output/my-experiment/ \
    --model "openai/gpt-4"
```

**Note:** `--output` specifies a directory where results will be saved. Each commit will be saved as a separate JSON file in a subdirectory named after the model.

**Options:**
- `--input`: Input JSONL file with commits (default: `data/50-random-commits-validation.jsonl`)
- `--output`: Output directory for results (default: `output/`)
- `--model`: LLM model identifier (default: `ollama/gpt-oss:20b`)
- `--temperature`: Sampling temperature (default: 0.0)
- `--context-mode`: Context to include: `message`, `message+diff`, `full` (default: `message`)
- `--max-tokens`: Maximum response tokens (default: 3072)
- `--workers`: Number of parallel workers (default: 10)
- `--retry-delay`: Seconds to wait on rate limit (default: 90)
- `--max-retries`: Maximum retries per commit (default: 3)

### Data Format

**Input JSONL format:**
```json
{
  "data": {
    "commit": "abc123...",
    "message": "Fix memory leak in driver",
    "files": [...],
    "diff": "..." // Optional, for context_mode="message+diff"
  }
}
```

**Output annotation format:**
```json
{
  "commit_hash": "abc123...",
  "timestamp": "2025-11-28T10:30:00Z",
  "model": "openai/gpt-4",
  "context_mode": "message",
  "understanding": {
    "score": 3,
    "description": "The commit fixes a memory leak..."
  },
  "bfc": {"score": 4, "reasoning": "..."},
  "bpc": {"score": 1, "reasoning": "..."},
  "prc": {"score": 2, "reasoning": "..."},
  "nfc": {"score": 0, "reasoning": "..."},
  "summary": "Primary bug fix with minor perfective improvements",
  "usage_metadata": {...}
}
```

## Advanced Features

### Adding Diff Context

To include commit diffs in annotations, first enrich your JSONL file:

```bash
python diffs/enrich_jsonl.py \
    data/50-random-commits-validation.jsonl \
    data/50-random-commits-validation-with-diff.jsonl \
    --repo /path/to/linux
```

Then annotate with diff context:
```bash
python annotate_validation_set.py \
    --input data/50-random-commits-validation-with-diff.jsonl \
    --output output/with-diff-experiment/ \
    --context-mode message+diff
```

See `diffs/README.md` for more diff retrieval options (GitHub API, GitPython, etc.).

### Convert Annotations to CSV

To convert all JSON annotations from a model's output directory to a single CSV:

```bash
python utils/convert_model_to_csv.py \
    output/my-experiment/openai_gpt-4/ \
    output/annotations_gpt4.csv
```

Or use the model folder name pattern:
```bash
python utils/convert_model_to_csv.py \
    output/openai_gpt-4/ \
    output/annotations_gpt4.csv
```

### Using the Annotation Class Programmatically

```python
from LLMCommitAnnotator import LLMCommitAnnotator
import json

# Initialize annotator
annotator = LLMCommitAnnotator(
    model="openai/gpt-4",
    temperature=0.0,
    context_mode="message"
)

# Load commit data
with open("data/sample-commits/commit.json") as f:
    commit_data = json.load(f)

# Annotate
result = annotator.annotate_commit(commit_data)

print(f"BFC Score: {result['bfc']['score']}")
print(f"Summary: {result['summary']}")
```

## Supported Models

The tool automatically detects and routes to the appropriate provider:

**OpenRouter** (most models):
- `meta-llama/llama-3.1-70b-instruct`
- `anthropic/claude-3.5-sonnet`
- `google/gemini-2.0-flash-exp`

**Google**:
- `gemini-2.0-flash-exp`
- `gemini-1.5-pro`

**OpenAI**:
- `gpt-4`
- `gpt-3.5-turbo`

**Ollama** (local):
- `llama3.1:8b`
- `codellama:13b`
- `mistral:7b`

## Research Workflow

1. **Prepare data**: Collect commits in JSONL format
2. **Optional**: Enrich with diffs using `diffs/enrich_jsonl.py`
3. **Annotate**: Run batch annotations with different models
4. **Convert**: Transform to CSV for analysis
5. **Analyze**: Use `analysis/disagreement_analysis.ipynb` for inter-annotator agreement

## Documentation

- `documentation/definitions.md` - Detailed category definitions
- `documentation/METHOD.md` - Research methodology
- `diffs/README.md` - Diff retrieval methods
- `analysis/disagreement_analysis.ipynb` - Statistical analysis

## Contributing

This is a research project. Feel free to open issues or submit pull requests.