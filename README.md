# LLM Commit Classifier

This is a simple script that uses OpenRouter LLMs to classify git commits into specific categories based on their purpose and nature.

## Commit Categories

The script classifies commits into one of the following categories:

- **Bug-Fixing Commit (BFC)**: Fixes a bug present in the source code that manifests as a failure
- **Bug-Preventing Commit (BPC)**: Prevents potential bugs that could cause failures in the future
- **Perfective Commit (PRC)**: Improves code quality (refactoring, optimization, style improvements, comments)
- **New Feature Commit (NFC)**: Adds new functionality or capabilities to the codebase

See `documentation/definitions.md` for detailed definitions of each category.

## How to install

Clone repository, and install dependencies (likely in a Python virtual environment):

```commandline
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Copy the `dotenv-example` file to `.env` and adjust it to your needs,
by setting your OpenRouter API key.

### How to get an OpenRouter API key

1. Visit https://openrouter.ai/ and create an account (you can use GitHub, Google, or email)
2. Navigate to your dashboard and go to API Keys section
3. Create a new API key
4. Copy the key (starts with `sk-...`) and paste it in your `.env` file as `OPENROUTER_API_KEY`

**Important**: Keep your API key secure and never commit the `.env` file to version control.

## How to use

The script requires a JSON file containing the commit data in the following format:

```json
{
  "data": {
    "message": "Your commit message here"
  }
}
```

Run the script with the commit file as an argument:

```bash
source .venv/bin/activate
python script.py <commit-file.json>
```

Example with demo data:

```bash
python script.py data/sample-commits/20-1eba86c096e35e3cc83de1ad2c26f2d70470211b.json
```

The script will output:
- The classification category (BFC, BPC, PRC, or NFC)
- A detailed explanation of why that category was chosen
- Token usage statistics (printed to stderr)