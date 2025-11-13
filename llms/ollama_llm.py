"""
Ollama LLM wrapper.
"""

import os


class OllamaLLM:
    """Wrapper for Ollama models."""
    
    @staticmethod
    def is_supported(model_name: str) -> bool:
        """Check if this provider supports the given model."""
        # Support models with ollama/ prefix or known Ollama models
        return (
            model_name.startswith("ollama/") or
            "gpt-oss" in model_name.lower() or
            "gtp-oss" in model_name.lower()  # Common typo
        )
    
    @staticmethod
    def initialize(model: str, temperature: float, max_tokens: int):
        """Initialize Ollama client."""
        # Import ollama library
        try:
            import ollama
        except ImportError:
            raise ImportError(
                "ollama is not installed.\n"
                "Install it with: pip install ollama"
            )
        
        # Get Ollama host from environment or use default
        host = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        
        # Remove "ollama/" prefix if present
        model_name = model.replace("ollama/", "")
        
        # Create wrapper class
        class OllamaWrapper:
            def __init__(self, model, host, temperature, max_tokens):
                self.model = model
                self.client = ollama.Client(host=host)
                self.temperature = temperature
                self.max_tokens = max_tokens
            
            def invoke(self, prompt):
                """
                Call Ollama API to generate a response.
                """
                try:
                    response = self.client.generate(
                        model=self.model,
                        prompt=prompt,
                        options={
                            "temperature": self.temperature,
                            "num_predict": self.max_tokens
                        }
                    )
                    
                    # Create a response object similar to LangChain's
                    class Response:
                        def __init__(self, text, prompt_tokens, response_tokens):
                            self.content = text
                            self.usage_metadata = {
                                "input_tokens": prompt_tokens,
                                "output_tokens": response_tokens,
                                "total_tokens": prompt_tokens + response_tokens
                            }
                    
                    # Extract response and token counts
                    response_text = response.get("response", "")
                    prompt_tokens = response.get("prompt_eval_count", 0)
                    response_tokens = response.get("eval_count", 0)
                    
                    return Response(response_text, prompt_tokens, response_tokens)
                    
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to connect to Ollama at {host}.\n"
                        f"Make sure Ollama is running and accessible.\n"
                        f"Error: {str(e)}"
                    )
        
        return OllamaWrapper(model_name, host, temperature, max_tokens)
