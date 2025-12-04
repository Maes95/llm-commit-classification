import os
import json
import re
import time
from datetime import datetime
from typing import Dict, Any, Optional
from llms import GoogleLLM, OpenAILLM, OpenRouterLLM, OllamaLLM


class LLMCommitAnnotator:
    """
    Class responsible for annotating git commits using Large Language Models.
    Handles LLM initialization, prompt construction, and annotation execution.
    """

    DEFINITIONS = "documentation/definitions.md"
    CONTEXT_FOR_ANNOTATORS = "documentation/context.md"

    def __init__(
        self,
        model: str = "meta-llama/llama-4-maverick:free",
        temperature: float = 0.0,
        max_tokens: int = 3072,
        context_mode: str = "message"
    ):
        """
        Initialize the LLM Commit Annotator.
        
        Args:
            model: The model identifier to use
                   For OpenRouter: "meta-llama/llama-4-maverick:free"
                   For Google: "gemini-2.0-flash-exp" or "gemini-1.5-pro"
                   For OpenAI: "gpt-4", "gpt-3.5-turbo"
                   For Ollama: "gpt-oss:20b" or "ollama/gpt-oss:20b"
            temperature: LLM temperature for reproducibility (default: 0.0)
            max_tokens: Maximum tokens for LLM response (default: 3072)
            context_mode: Type of context to include in the prompt (default: "message")
                         Options:
                         - "message": Only commit message
                         - "message+diff": Commit message, diff, stats, and modified files
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.context_mode = context_mode
        
        # Validate context_mode
        valid_modes = ["message", "message+diff"]
        if context_mode not in valid_modes:
            raise ValueError(f"Invalid context_mode: {context_mode}. Must be one of {valid_modes}")
        
        # Load definitions and context
        self.definitions_content = self._load_definitions()
        self.context_for_annotators = self._load_context_for_annotators()
        
        # Initialize LLM
        self.llm = self._initialize_llm()
    
    def _load_definitions(self) -> str:
        """Load category definitions from file."""
        if not os.path.exists(self.DEFINITIONS):
            raise FileNotFoundError(f"Definitions file not found: {self.DEFINITIONS}")

        with open(self.DEFINITIONS, "r") as f:
            return f.read()
    
    def _load_context_for_annotators(self) -> str:
        """Load context for annotators from file."""
        if not os.path.exists(self.CONTEXT_FOR_ANNOTATORS):
            raise FileNotFoundError(f"Context file not found: {self.CONTEXT_FOR_ANNOTATORS}")

        with open(self.CONTEXT_FOR_ANNOTATORS, "r") as f:
            return f.read()
    
    def _initialize_llm(self):
        """Initialize the LLM client based on model type."""
        # List of providers to check (order matters - more specific first)
        providers = [OllamaLLM, OpenRouterLLM, GoogleLLM, OpenAILLM]
        
        # Find the appropriate provider
        for provider in providers:
            if provider.is_supported(self.model):
                return provider.initialize(self.model, self.temperature, self.max_tokens)
        
        # Should never reach here due to OpenRouterLLM being a catch-all
        raise ValueError(f"No provider found for model: {self.model}")
    
    def _build_commit_context(self, commit_data: Dict[str, Any]) -> str:
        """
        Build the commit context section based on context_mode.
        
        Args:
            commit_data: Dictionary containing commit information
            
        Returns:
            Formatted context string
        """
        data = commit_data["data"]
        commit_message = data["message"]
        
        # Remove 'Fixes:' lines from commit message
        commit_message = re.sub(r'^Fixes:.*\n', '', commit_message, flags=re.M)
        
        context_parts = []
        
        # Always include commit message
        context_parts.append(f"COMMIT MESSAGE:\n{commit_message}")
        
        # Add diff, stats, and files if message+diff mode
        if self.context_mode == "message+diff":
            if "diff" in data and data["diff"]:
                context_parts.append(f"\nCOMMIT DIFF:\n{data['diff']}")
            if "stats" in data and data["stats"]:
                context_parts.append(f"\nCOMMIT STATS:\n{data['stats']}")
            if "files" in data and data["files"]:
                files_list = "\n".join([f"  - {f}" for f in data["files"]])
                context_parts.append(f"\nMODIFIED FILES:\n{files_list}")
        
        return "\n".join(context_parts)
    
    def _build_prompt(self, commit_data: Dict[str, Any]) -> str:
        """
        Build the annotation prompt for a commit.
        
        Args:
            commit_data: Dictionary containing commit information
            
        Returns:
            The formatted prompt string
        """
        commit_context = self._build_commit_context(commit_data)
        
        template = """[SYSTEM INSTRUCTION]
You are an expert software engineering analyst specializing in commit annotation.
Your task is to evaluate commits across multiple dimensions simultaneously.

[CONTEXT FOR ANNOTATORS]
{context_for_annotators}

[TAXONOMY DEFINITIONS]
Below are the definitions and categories you must use for annotation:
{definitions}

[UNDERSTANDING ASSESSMENT]
Before annotating, assess your comprehension of the commit using this rubric:
0 = No comprehension - Cannot determine what the commit does
1 = Minimal comprehension - Only vague understanding of general area
2 = Partial comprehension - Understand some aspects but missing key details
3 = Good comprehension - Understand the main changes and their purpose
4 = Complete comprehension - Full understanding of all technical changes and context

[TASK INSTRUCTION]
Annotate the following commit by assigning a score from 0 to 4 for EACH of the four categories:
- Bug-Fixing Commit (BFC)
- Bug-Preventing Commit (BPC)
- Perfective Commit (PRC)
- New Feature Commit (NFC)

Scoring rubric:
0 = Not applicable - The commit shows no characteristics of this category
1 = Minimal - The commit shows very slight or tangential characteristics
2 = Moderate - The commit partially exhibits characteristics of this category
3 = Strong - The commit clearly exhibits characteristics of this category
4 = Primary - This is a primary or dominant characteristic of the commit

Use chain-of-thought reasoning for each dimension:
1. Identify evidence in the commit relevant to this dimension
2. Evaluate the strength and significance of this evidence
3. Assign an appropriate score based on the rubric

[COMMIT CONTEXT]
{commit_message}

[OUTPUT FORMAT]
Provide your annotation as a valid JSON object with the following structure:
{{
  "understanding": {{
    "score": 0,
    "description": "Clear description of what the commit does and its technical changes"
  }},
  "bfc": {{
    "score": 0,
    "reasoning": "Detailed explanation of BFC score"
  }},
  "bpc": {{
    "score": 0,
    "reasoning": "Detailed explanation of BPC score"
  }},
  "prc": {{
    "score": 0,
    "reasoning": "Detailed explanation of PRC score"
  }},
  "nfc": {{
    "score": 0,
    "reasoning": "Detailed explanation of NFC score"
  }},
  "summary": "Brief synthesis of the commit's primary purposes"
}}

CRITICAL: Your response must be ONLY the raw JSON object. Do not wrap it in markdown code blocks (```json or ```). Do not include any explanatory text before or after the JSON. Start your response with {{ and end with }}.
"""
        
        return template.format(
            context_for_annotators=self.context_for_annotators,
            definitions=self.definitions_content,
            commit_message=commit_context
        )
    
    def annotate_commit(self, commit_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Annotate a single commit.
        
        Args:
            commit_data: Dictionary containing commit information with 'data.message' field
                        and optionally 'data.diff', 'data.stats', 'data.files'
            
        Returns:
            Dictionary containing:
                - commit_hash: The commit SHA hash
                - timestamp: ISO format timestamp when annotation was generated
                - understanding: Dict with score and description
                - bfc: Dict with score and reasoning
                - bpc: Dict with score and reasoning
                - prc: Dict with score and reasoning
                - nfc: Dict with score and reasoning
                - summary: Brief synthesis of the commit's primary purposes
                - usage_metadata: Token usage information
                - model: Model used for annotation
                - temperature: Temperature parameter used
                - max_tokens: Maximum tokens parameter used
                - context_mode: The context mode used for annotation
                - raw_response: Original LLM response text (for debugging)
                - prompt: The full prompt sent to the LLM
        """
        commit_hash = commit_data["data"]["commit"]
        
        # Build prompt
        prompt_text = self._build_prompt(commit_data)
        
        # Invoke LLM and capture timestamp
        timestamp = datetime.utcnow().isoformat() + "Z"
        start_time = time.time()
        response = self.llm.invoke(prompt_text)
        elapsed_time = time.time() - start_time
        
        # Parse JSON response
        try:
            annotation = json.loads(response.content)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Failed to parse LLM response as JSON: {e}\n"
                f"Raw response: {response.content}"
            )
        # Return structured result with metadata
        return {
            "commit_hash": commit_hash,
            "timestamp": timestamp,
            "elapsed_time_seconds": round(elapsed_time, 3),
            "understanding": annotation.get("understanding"),
            "bfc": annotation.get("bfc"),
            "bpc": annotation.get("bpc"),
            "prc": annotation.get("prc"),
            "nfc": annotation.get("nfc"),
            "summary": annotation.get("summary"),
            "usage_metadata": response.usage_metadata,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "context_mode": self.context_mode,
            "raw_response": response.content,
            "prompt": prompt_text
        }
