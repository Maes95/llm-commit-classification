import os
import json
from datetime import datetime
from langchain.chat_models import init_chat_model
from typing import Dict, Any, Optional

# Import specific chat models
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:
    ChatGoogleGenerativeAI = None


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
            model: The model identifier to use
                   For OpenRouter: "meta-llama/llama-4-maverick:free"
                   For Google: "gemini-2.0-flash-exp" or "gemini-1.5-pro"
                   For OpenAI: "gpt-4", "gpt-3.5-turbo"
            temperature: LLM temperature for reproducibility (default: 0.0)
            max_tokens: Maximum tokens for LLM response (default: 3072)
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
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
        """Initialize the LLM client based on model type."""
        # Detect provider based on model name
        if "gemini" in self.model.lower() or "gemma" in self.model.lower() or self.model.startswith("google/"):
            # Google Generative AI - use direct API
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError(
                    "GOOGLE_API_KEY is not set.\n"
                    "Set it in your environment for Google models.\n"
                    "Get a key at https://aistudio.google.com/app/apikey"
                )
            
            # Remove "google/" prefix and ":free" suffix if present
            model_name = self.model.replace("google/", "").replace(":free", "")
            
            # Create a custom wrapper using google-generativeai directly
            try:
                from google import generativeai as genai
            except ImportError:
                raise ImportError(
                    "google-generativeai is not installed.\n"
                    "Install it with: pip install google-generativeai"
                )
            
            class GoogleWrapper:
                def __init__(self, model, api_key, temperature, max_tokens):
                    genai.configure(api_key=api_key)
                    self.model = genai.GenerativeModel(model)
                    self.temperature = temperature
                    self.max_tokens = max_tokens
                
                def invoke(self, prompt):
                    generation_config = {
                        "temperature": self.temperature,
                        "max_output_tokens": self.max_tokens,
                    }
                    response = self.model.generate_content(
                        prompt,
                        generation_config=generation_config
                    )
                    
                    # Create a response object similar to LangChain's
                    class Response:
                        def __init__(self, text):
                            self.content = text
                            self.usage_metadata = {
                                "input_tokens": 0,  # Google API doesn't always provide this
                                "output_tokens": 0
                            }
                    
                    return Response(response.text)
            
            return GoogleWrapper(model_name, api_key, self.temperature, self.max_tokens)
        
        elif "gpt" in self.model.lower() or self.model.startswith("openai/"):
            # OpenAI
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "OPENAI_API_KEY is not set.\n"
                    "Set it in your environment for OpenAI models.\n"
                    "Get a key at https://platform.openai.com/api-keys"
                )
            
            model_name = self.model.replace("openai/", "")
            
            return init_chat_model(
                model_provider="openai",
                model=model_name,
                api_key=api_key,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
        
        else:
            # Default to OpenRouter for other models
            api_key = os.getenv("OPENROUTER_API_KEY")
            base_url = os.getenv("OPENROUTER_BASE_URL")
            
            if not api_key:
                raise ValueError(
                    "OPENROUTER_API_KEY is not set.\n"
                    "Set it in your environment or pass it to the constructor.\n"
                    "You can get a key at https://openrouter.ai/"
                )
            
            return init_chat_model(
                model_provider="openai",
                model=self.model,
                api_key=api_key,
                base_url=base_url,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
    
    def _build_prompt(self, commit_message: str) -> str:
        """
        Build the annotation prompt for a commit.
        
        Args:
            commit_message: The commit message to annotate
            
        Returns:
            The formatted prompt string
        """
        template = """[SYSTEM INSTRUCTION]
You are an expert software engineering analyst specializing in commit annotation.
Your task is to evaluate commits across multiple dimensions simultaneously.

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
                - raw_response: Original LLM response text (for debugging)
        """
        commit_message = commit_data["data"]["message"]
        commit_hash = commit_data["data"]["commit"]
        
        # Build prompt
        prompt_text = self._build_prompt(commit_message)
        
        # Invoke LLM and capture timestamp
        timestamp = datetime.utcnow().isoformat() + "Z"
        response = self.llm.invoke(prompt_text)
        
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
            "understanding": annotation.get("understanding"),
            "bfc": annotation.get("bfc"),
            "bpc": annotation.get("bpc"),
            "prc": annotation.get("prc"),
            "nfc": annotation.get("nfc"),
            "summary": annotation.get("summary"),
            "usage_metadata": response.usage_metadata,
            "model": self.model,
            "raw_response": response.content,
            "prompt": prompt_text
        }
