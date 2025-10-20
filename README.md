# LLM Commit Classifier

This is a simple script that uses OpenRouter LLMs to classify commits

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

## How to use with the demo data

Just run the script:

```commandline
python script.py <commit-in-percival-format>
```

```commandline
python script.py data/sample-commits/20-1eba86c096e35e3cc83de1ad2c26f2d70470211b.json
```