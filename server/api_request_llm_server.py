import json
import os
import requests
from ..logger import logger
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from agentriven.core import LLMServerABC
import re
import time

class APIREQUESTLLMServer(LLMServerABC):
    """Use Request API to generate responses based on input messages.
    """
    def start_server(self) -> None:
        logger.info("APIREQUESTLLMServing_request: no local service to start.")
        return
    
    def __init__(self, 
                 api_url: str = "https://api.openai.com/v1/chat/completions",
                 api_key: str = "OPENAI_API_KEY",
                 model_name: str = "gpt-4o",
                 max_workers: int = 10,
                 max_retries: int = 5,
                 ):
        # Get API key from environment variable or config
        self.api_url = api_url
        self.model_name = model_name
        self.max_workers = max_workers
        self.max_retries = max_retries

        # config api_key in os.environ global, since safty issue.
        self.api_key = os.environ.get(api_key)
        if self.api_key is None:
            error_msg = f"Lack of `{api_key}` in environment variables. Please set `{api_key}` as your api-key to {api_url} before using APILLMServing_request."
            logger.error(error_msg)
            raise ValueError(error_msg)
    def cleanup(self):
        # Cleanup resources if needed
        logger.info("Cleaning up resources in APIOPENAILLMServer")
        return
    def generate(
        self, 
        user_input: list[str], 
        system_prompt: str = "You are a helpful assistant",
        json_schema: dict = None,
    ) -> list[str]:
        responses = [None] * len(user_input)
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(
                    self._api_chat_id_retry,
                    payload = [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": question}
                        ],
                    model = self.model_name,
                    json_schema = json_schema,
                    id = idx,
                ) for idx, question in enumerate(user_input)
            ]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Generating......"):
                    response = future.result() # (id, response)
                    responses[response[0]] = response[1]
        return responses
        
    def _api_chat_id_retry(self, id, payload, model, is_embedding : bool = False, json_schema: dict = None):
        for i in range(self.max_retries):
            id, response = self._api_chat_with_id(id, payload, model, is_embedding, json_schema)
            if response is not None:
                return id, response
            time.sleep(2**i)
        return id, None    
    
    def _api_chat_with_id(self, id, messages, model, is_embedding: bool = False, json_schema: dict = None):
        try:
            if is_embedding:
                messages = json.dumps({
                    "model": model,
                    "input": messages
                })
            elif json_schema is None:
                messages = json.dumps({
                    "model": model,
                    "messages": messages
                })
            else:
                messages = json.dumps({
                    "model": model,
                    "messages": messages,
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "custom_response",
                            "strict": True,
                            "schema": json_schema
                        }
                    }
                })
                
            headers = {
                'Authorization': f"Bearer {self.api_key}",
                'Content-Type': 'application/json',
                'User-Agent': 'Apifox/1.0.0 (https://apifox.com)'
            }
            # Make a POST request to the API
            response = requests.post(self.api_url, headers=headers, data=messages, timeout=1800)
            if response.status_code == 200:
                # logging.info(f"API request successful")
                response_data = response.json()
                # logging.info(f"API response: {response_data['choices'][0]['message']['content']}")
                return id,self.format_response(response_data, is_embedding)
            else:
                logger.error(f"API request failed with status {response.status_code}: {response.text}")
                return id, None
        except Exception as e:
            logger.error(f"API request error: {e}")
            return id, None
    