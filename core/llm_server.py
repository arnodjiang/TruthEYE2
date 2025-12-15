from abc import ABC, abstractmethod
from typing import Any, List

class LLMServerABC(ABC):
    """Abstract base class for LLM servers. Which may be used to serve LLM models or APIs. Called by users to get LLM responses.
    """
    @abstractmethod
    def generate(self, user_input: List[str], system_prompt: str, json_schema: dict = None) -> List[str]:
        """
        Generate data from input.
        user_input: List[str], the input of the generator
        system_prompt: str, the system prompt to be used
        """
        pass
    
    @abstractmethod
    def start_server(self):
        """
        Cleanup the generator and garbage collect all GPU/CPU memory.
        """
        pass
    
    @abstractmethod
    def cleanup(self):
        """
        Cleanup the generator and garbage collect all GPU/CPU memory.
        """
        pass
    
    def load_model(self, model_path: str, **kwargs: Any):
        """
        Load the model from the given path.
        This method is optional and can be overridden by subclasses if needed.
        model_path: str, the path to the model to be loaded
        """
        raise NotImplementedError("This method should be implemented by subclasses.")