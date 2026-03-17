import os
import json
import re
import time
from datetime import datetime
from typing import Dict, Any, Optional, Set
from llms import GoogleLLM, OpenAILLM, OpenRouterLLM, OllamaLLM


class LLMCommitAnnotator:
    """
    Class responsible for annotating git commits using Large Language Models.
    Handles LLM initialization, prompt construction, and annotation execution.
    """

    DEFINITIONS = "documentation/definitions.md"
    CONTEXT_FOR_ANNOTATORS = "documentation/context.md"
    FEW_SHOT_EXAMPLES = "documentation/few-shot-examples.md"

    def __init__(
        self,
        model: str = "meta-llama/llama-4-maverick:free",
        temperature: float = 0.0,
        max_tokens: int = 10000,
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
            max_tokens: Maximum tokens for LLM response (default: 10000)
            context_mode: Context/policy mode string (default: "message")
                                        Flags can be combined with '+':
                                        - "message": Only commit message (base context)
                                        - "diff": Adds diff, stats, and modified files
                                        - "single-label": Adds single-label scoring policy
                                        - "few-shot": Adds human annotation examples from
                                            documentation/few-shot-examples.md
                                        - "diff+single-label": Enables both diff context and
                                            single-label scoring policy
                                        - "diff+single-label+few-shot": Enables all behaviors
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.mode_flags = self._parse_context_mode(context_mode)
        self.context_mode = self._normalize_context_mode(self.mode_flags)

        # Load definitions and context
        self.definitions_content = self._load_definitions()
        self.context_for_annotators = self._load_context_for_annotators()
        self.few_shot_examples = self._load_few_shot_examples()

        # Initialize LLM
        self.llm = self._initialize_llm()

    def _parse_context_mode(self, context_mode: str) -> Set[str]:
        """Parse context_mode into composable flags."""
        valid_flags = {"message", "diff", "single-label", "few-shot"}

        if not isinstance(context_mode, str) or not context_mode.strip():
            raise ValueError("context_mode must be a non-empty string")

        parts = [part.strip() for part in context_mode.split("+")]
        if any(not part for part in parts):
            raise ValueError(
                "Invalid context_mode format. Use 'message', 'diff', 'single-label', "
                "'few-shot', or combinations like 'diff+single-label+few-shot'."
            )

        mode_flags = set(parts)
        unknown = mode_flags - valid_flags
        if unknown:
            raise ValueError(
                "Invalid context_mode flag(s): "
                f"{sorted(unknown)}. Valid flags: {sorted(valid_flags)}. "
                "Supported examples: 'message', 'diff', 'single-label', 'few-shot', "
                "'diff+single-label', 'diff+single-label+few-shot'."
            )

        return mode_flags

    def _normalize_context_mode(self, mode_flags: Set[str]) -> str:
        """Return canonical mode name for output metadata consistency."""
        effective = set(mode_flags)

        # message is implicit base context; remove it in combined canonical names.
        if "message" in effective and len(effective) > 1:
            effective.remove("message")

        if not effective:
            return "message"

        if effective == {"message"}:
            return "message"

        ordered = [name for name in ["diff", "single-label", "few-shot"] if name in effective]
        return "+".join(ordered)

    def _load_few_shot_examples(self) -> str:
        """Load few-shot examples when few-shot mode is active."""
        if "few-shot" not in self.mode_flags:
            return ""

        if not os.path.exists(self.FEW_SHOT_EXAMPLES):
            raise FileNotFoundError(
                f"Few-shot examples file not found: {self.FEW_SHOT_EXAMPLES}. "
                "Create and fill this template when using context_mode with 'few-shot'."
            )

        with open(self.FEW_SHOT_EXAMPLES, "r") as f:
            return f.read()
    
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
        
        # Add diff, stats, and files when diff mode is active.
        if "diff" in self.mode_flags:
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
        policy_section = ""
        few_shot_section = ""
        compact_output_section = """
    [OUTPUT BUDGET]
    Keep output compact to avoid malformed/truncated JSON:
    - understanding.description: max 80 words
    - each reasoning field (bfc/bpc/prc/nfc): max 60 words
    - summary: max 40 words
    Use concise, evidence-based statements. Avoid long quotations from the commit.
    """

        if "single-label" in self.mode_flags:
            policy_section = """
[SINGLE-LABEL POLICY]
Apply a conservative single-label policy across BFC/BPC/PRC/NFC:
1) By default, exactly one category may have score > 0. All others must be 0.
2) Exception allowed ONLY under considerable doubt:
   - understanding.score <= 2, OR
   - explicit, independent, and comparably strong evidence of two different purposes.
3) If understanding.score >= 3 and secondary evidence is weak/tangential,
   collapse to single-label: keep one dominant category > 0 and set all others to 0.
4) Avoid residual positives: do not assign 1 to a secondary category unless rule (2) is met.
"""

        if "few-shot" in self.mode_flags:
            few_shot_section = f"""
[FEW-SHOT HUMAN EXAMPLES]
Use these human-annotated examples as calibration references.
Do not copy text verbatim; use them to calibrate scoring decisions.

{self.few_shot_examples}
"""
        
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

Use concise reasoning for each dimension:
1. Identify evidence in the commit relevant to this dimension
2. Evaluate the strength and significance of this evidence
3. Assign an appropriate score based on the rubric

IMPORTANT: Provide only short final justifications, not long internal deliberations.

{policy_section}

{few_shot_section}

{compact_output_section}

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
            policy_section=policy_section,
            few_shot_section=few_shot_section,
            compact_output_section=compact_output_section,
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