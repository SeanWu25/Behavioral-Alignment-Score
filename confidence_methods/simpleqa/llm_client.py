# llm_client.py (secure version)
import os
from openai import AzureOpenAI, OpenAI
from anthropic import AnthropicFoundry

# Default Configuration (can be overridden)
DEFAULT_AZURE_ENDPOINT = "https://bascolm.cognitiveservices.azure.com/"
DEFAULT_API_VERSION = "2024-12-01-preview"
DEFAULT_CUSTOM_BASE_URL = "https://bascolm.services.ai.azure.com/openai/v1/"

class LLMClient:
    def __init__(self, provider="azure", base_url=None, api_key=None, api_version=None):
        self.provider = provider

        # 🔐 Load API key securely
        self.api_key = api_key or os.getenv("LLM_API_KEY")
        if not self.api_key:
            raise ValueError("API key not provided. Set LLM_API_KEY environment variable.")

        if self.provider == "azure":
            self.endpoint = base_url or DEFAULT_AZURE_ENDPOINT
            self.api_version = api_version or DEFAULT_API_VERSION
            self.client = AzureOpenAI(
                azure_endpoint=self.endpoint,
                api_key=self.api_key,
                api_version=self.api_version
            )

        elif self.provider == "custom":
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

        elif self.provider == "openai":
            self.client = OpenAI(api_key=self.api_key)

        else:
            raise ValueError(f"Unknown provider '{self.provider}'")

        print(f"[LLMClient] Initialized with provider='{self.provider}'")
        if self.provider == "custom":
            print(f"[LLMClient] Custom Base URL: {self.base_url}")
        elif self.provider == "azure":
            print(f"[LLMClient] Azure Endpoint: {self.endpoint}")

    def chat_completion(self, model, messages, temperature=0, **kwargs):
        """
        Unified call for chat completions.
        """
        try:
            if self.provider == "anthropic_azure":
                system_prompt = None
                filtered_messages = []

                for msg in messages:
                    if msg.get('role') == 'system':
                        system_prompt = msg.get('content')
                    else:
                        filtered_messages.append(msg)

                create_kwargs = {
                    "model": model,
                    "messages": filtered_messages,
                    "max_tokens": kwargs.pop("max_tokens", 1024),
                    **kwargs
                }

                if system_prompt:
                    create_kwargs["system"] = system_prompt

                response = self.client.messages.create(**create_kwargs)

                class AnthropicResponseWrapper:
                    def __init__(self, content):
                        self.choices = [
                            type(
                                'Choice', (), {
                                    'message': type(
                                        'Message', (), {'content': content}
                                    )()
                                }
                            )()
                        ]

                return AnthropicResponseWrapper(response.content[0].text)

            elif self.provider in ("azure", "custom", "openai"):
                return self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    **kwargs
                )

            else:
                raise ValueError(f"Unhandled provider in chat_completion: {self.provider}")

        except Exception as e:
            print(f"LLM Call Error ({self.provider}): {e}")
            raise