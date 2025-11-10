import os
import json
from langchain.chat_models import init_chat_model
from typing import Dict, Any, Optional


class LLMCommitAnnotator:
    """
    Class responsible for annotating git commits using Large Language Models.
    Handles LLM initialization, prompt construction, and annotation execution.
    """

    DEFINITIONS = "documentation/definitions.md"

    def __init__(
        self,
        model: str = "meta-llama/llama-4-maverick:free",
        temperature: float = 0.0,
        max_tokens: int = 3072
    ):
        """
        Initialize the LLM Commit Annotator.
        
        Args:
            model: The model identifier to use (e.g., "meta-llama/llama-4-maverick:free")
            api_key: OpenRouter API key (if None, reads from environment)
            base_url: OpenRouter base URL (if None, reads from environment)
            definitions_file: Path to the file containing category definitions
            temperature: LLM temperature for reproducibility (default: 0.0)
            max_tokens: Maximum tokens for LLM response (default: 3072)
        """
        self.model = model
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.base_url = os.getenv("OPENROUTER_BASE_URL")
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        if not self.api_key:
            raise ValueError(
                "OPENROUTER_API_KEY is not set.\n"
                "Set it in your environment or pass it to the constructor.\n"
                "You can get a key at https://openrouter.ai/"
            )
        
        # Load definitions
        self.definitions_content = self._load_definitions()
        
        # Initialize LLM
        self.llm = self._initialize_llm()
    
    def _load_definitions(self) -> str:
        """Load category definitions from file."""
        if not os.path.exists(self.DEFINITIONS):
            raise FileNotFoundError(f"Definitions file not found: {self.DEFINITIONS}")

        with open(self.DEFINITIONS, "r") as f:
            return f.read()
    
    def _initialize_llm(self):
        """Initialize the LLM client."""
        # Extract provider from model string (e.g., "meta-llama" -> "meta")
        provider = self.model.split("/")[0].split("-")[0] if "/" in self.model else "meta"
        
        return init_chat_model(
            model_provider="openai",
            model=self.model,
            api_key=self.api_key,
            base_url=self.base_url,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            extra_body={"provider": {"require_parameters": True, "only": [provider]}}
        )
    
    def _build_prompt(self, commit_message: str) -> str:
        """
        Build the annotation prompt for a commit.
        
        Args:
            commit_message: The commit message to annotate
            
        Returns:
            The formatted prompt string
        """
        template = """You are an expert software engineering analyst tasked with classifying git commits into specific categories.

Below are the definitions and categories you must use for classification:

{definitions}

---

Now, classify the following commit message into ONE of these categories:
- Bug-Fixing Commit (BFC)
- Bug-Preventing Commit (BPC)
- Perfective Commit (PRC)
- New Feature Commit (NFC)

Commit message to classify:

```
{commit_message}
```

Your response must include:
1. The selected category (one of: BFC, BPC, PRC, or NFC)
2. A paragraph explaining why you chose that category, based on the definitions provided above

Format your response as:
**Classification:** [Category Name]

**Explanation:** [Your detailed explanation here]
"""
        
        return template.format(
            definitions=self.definitions_content,
            commit_message=commit_message
        )
    
    def annotate_commit(self, commit_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Annotate a single commit.
        
        Args:
            commit_data: Dictionary containing commit information with 'data.message' field
            
        Returns:
            Dictionary containing:
                - classification: The LLM's classification text
                - usage_metadata: Token usage information
                - model: Model used for annotation
        """
        commit_message = commit_data["data"]["message"]
        
        # Build prompt
        prompt_text = self._build_prompt(commit_message)
        
        # Invoke LLM
        response = self.llm.invoke(prompt_text)
        
        return {
            "classification": response.content,
            "usage_metadata": response.usage_metadata,
            "model": self.model
        }
    
    def annotate_commit_from_file(self, commit_file: str) -> Dict[str, Any]:
        """
        Annotate a commit from a JSON file.
        
        Args:
            commit_file: Path to JSON file containing commit data
            
        Returns:
            Dictionary containing classification results and metadata
        """
        with open(commit_file, "r") as f:
            commit_data = json.load(f)
        
        return self.annotate_commit(commit_data)
