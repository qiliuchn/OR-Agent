# src/orcanvas/tools.py
import os 
import sys
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Union, Any
from openai import OpenAI
import concurrent
import random
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


# ===== LLM client for OR-Canvas=====
class LLMClient:
    def __init__(self, llm_provider=None, model_name=None, **kwargs):
        """
        Initialize the LLM client.
        If `llm_provider` is not in ['openai', 'qwen', 'deepseek'], you should set `llm_provider` to be None and 
        provide the following parameters in `kwargs`: 
        - `api_key`
        - `base_url`
        
        Args:
            llm_provider (str): The LLM provider name
            model_name (str): The model name to use
            **kwargs: Additional arguments passed to the OpenAI client
        """        
        # -----LLM client settings-----
        self.max_tries = 5
        # If LLM is not specified, use default LLM settings from config file
        self.llm_provider = llm_provider or "deepseek"
        self.model_name = model_name or "deepseek-reasoner"
        
        # Create client
        if "openai" in self.llm_provider.lower():
            self.client = OpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url="https://api.openai.com/v1",
                **kwargs
            )
        elif "qwen" in self.llm_provider.lower():
            self.client = OpenAI(
                api_key=os.getenv("DASHSCOPE_API_KEY"),
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                **kwargs
            )
        elif "deepseek" in self.llm_provider.lower():
            self.client = OpenAI(
                api_key=os.getenv("DEEPSEEK_API_KEY"),
                base_url="https://api.deepseek.com",
                **kwargs
            )
        else:
            try:
                self.client = OpenAI(
                    **kwargs  # include `api_key` and `base_url` in `kwargs` in this case
                )
            except:
                raise ValueError(f"Invalid model config: {kwargs}")


    def chat(self, messages: list[dict], **kwargs) -> str:
        """
        Generate one response using OpenAI Chat Completions API.
        
        Args:
            messages (list[dict]): One round of messages
            **kwargs: Additional arguments passed to the chat completions API
            
        Returns:
            str: LLM response string
        """
        choices = None
        for attempt in range(self.max_tries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    **kwargs
                )
                choices = response.choices
                if not choices:
                    print("\n>>>[LLMClient] None response generated")
                    print(f">>>[LLMClient] Response: {response}")
                else:
                    # check to make sure that choices is not none, then you can break! 
                    # otherwise, sometimes it returns None, and you will regard this as a successful generation
                    break
            except Exception as e:
                sleep_time = random.randint(3, 60)
                print(f"\n>>>[LLMClient] Attempt {attempt + 1} failed with error: {e}; sleep for {sleep_time} seconds")

                # Try to extract request ID from different exception structures
                request_id = None
                if hasattr(e, 'response') and hasattr(e.response, 'headers'):
                    request_id = e.response.headers.get('x-request-id')
                elif hasattr(e, 'headers'):
                    request_id = e.headers.get('x-request-id')
                elif hasattr(e, 'request_id'):
                    request_id = e.request_id

                if request_id:
                    print(f">>>[LLMClient] Request ID: {request_id}")
                else:
                    print(f">>>[LLMClient] No request ID found in exception")

                time.sleep(sleep_time)
                # Note: add random sleep time to avoid too frequent LLM call
                # TODO: this is a simple trick; could be made better
                # "devise technique to avoid LLM calls flooding" - This requires further exploration, analysis, and validation, and is marked with a TODO flag
        
        if choices is None:
            print("\n>>>[LLMClient] All retry attempts failed!")
            raise RuntimeError("\n>>>[LLMClient] All retry attempts failed")
        
        return choices[0].message.content
    