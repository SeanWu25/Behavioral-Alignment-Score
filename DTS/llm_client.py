import os
from openai import AzureOpenAI, OpenAI
from anthropic import AnthropicFoundry

# Default Configuration (can be overridden)
DEFAULT_AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "https://sean-bci.cognitiveservices.azure.com/")
DEFAULT_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
DEFAULT_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

# For the 'openai' style custom endpoint used in some scripts
DEFAULT_CUSTOM_BASE_URL = os.getenv("CUSTOM_LLM_BASE_URL", "https://sean-bci.services.ai.azure.com/openai/v1/")

class LLMClient:
    def __init__(self, provider="azure", base_url=None, api_key=None, api_version=None):
        self.provider = provider
        self.api_key = api_key or DEFAULT_API_KEY
        
        if self.provider == "azure":
            self.endpoint = base_url or DEFAULT_AZURE_ENDPOINT
            self.api_version = api_version or DEFAULT_API_VERSION
            self.client = AzureOpenAI(
                azure_endpoint=self.endpoint,
                api_key=self.api_key,
                api_version=self.api_version
            )
        elif self.provider == "custom":
            # Backward compatibility for the OpenAI(base_url=...) approach
            self.base_url = base_url or DEFAULT_CUSTOM_BASE_URL
            self.client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key
            )
        elif self.provider == "anthropic_azure":
             self.client = AnthropicFoundry(
                api_key=self.api_key,
                base_url=base_url
             )
        else:
            # Standard OpenAI
            self.client = OpenAI(api_key=self.api_key)
        
        print(f"[LLMClient] Initialized with provider='{self.provider}'")
        if self.provider == "custom":
            print(f"[LLMClient] Custom Base URL: {self.base_url}")
        elif self.provider == "azure":
            print(f"[LLMClient] Azure Endpoint: {self.endpoint}")

    def chat_completion(self, messages, model, temperature=0, **kwargs):
        """
        Unified call for chat completions.
        """
        try:
            if(model != "o3-mini"):
                return self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    **kwargs
                )
            elif self.provider == "anthropic_azure":
                # Adapt OpenAI-style messages to Anthropic format if needed, but the user snippet suggests
                # they are passing standard list-of-dicts. Anthropic SDK expects that too.
                # However, system prompt in Anthropic is often a top-level parameter, not in messages.
                # We will naively pass messages for now, but robustness might require extracting system prompt.
                
                # Check for system message and extract it if present
                system_prompt = None
                filtered_messages = []
                for msg in messages:
                    if msg['role'] == 'system':
                        system_prompt = msg['content']
                    else:
                        filtered_messages.append(msg)
                
                # Prepare arguments
                create_kwargs = {
                    "model": model,
                    "messages": filtered_messages,
                    "max_tokens": 1024, # Default max tokens as per user snippet
                    **kwargs
                }
                if system_prompt:
                    create_kwargs["system"] = system_prompt
                
                response = self.client.messages.create(**create_kwargs)
                
                # Wrap response to match OpenAI structure expected by simpleqa.py
                # OpenAI uses: completion.choices[0].message.content
                # Anthropic uses: message.content[0].text
                class AnthropicResponseWrapper:
                    def __init__(self, content):
                        self.choices = [type('Choice', (object,), {'message': type('Message', (object,), {'content': content})()})()]
                
                return AnthropicResponseWrapper(response.content[0].text)

            else:
                return self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    **kwargs
                )
        except Exception as e:
            print(f"LLM Call Error ({self.provider}): {e}")
            raise e
