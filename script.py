import os
import sys
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
import json

load_dotenv()

if len(sys.argv) < 2:
  print("ERROR: Missing required argument.", file=sys.stderr)
  print("Usage: python script.py <commit_file>", file=sys.stderr)
  sys.exit(1)

commit_file = sys.argv[1]

# Require API key to be present (friendly message if not set)
if not os.getenv("OPENROUTER_API_KEY"):
  print(
    "ERROR: OPENROUTER_API_KEY is not set.\n"
    "Copy `dotenv-example` to `.env` and set your OpenRouter API key (OPENROUTER_API_KEY=...).\n"
    "You can get a key at https://openrouter.ai/",
    file=sys.stderr,
  )
  sys.exit(1)

llm = init_chat_model(
    model_provider="openai",
    model="meta-llama/llama-4-maverick:free",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url=os.getenv("OPENROUTER_BASE_URL"),
    extra_body={"provider": {"require_parameters": True, "only": ["meta"]}}
)

template_report = """

BEGINNING

{commit_message}

END
"""

with open(commit_file, "r") as f:
  commit_data = json.load(f)
commit_message = commit_data["commit"]["message"]

# Format the prompt directly (avoids dependency on PromptTemplate across langchain versions)
prompt_text = template_report.format(commit_message=commit_message)

response = llm.invoke(prompt_text)

print(response)