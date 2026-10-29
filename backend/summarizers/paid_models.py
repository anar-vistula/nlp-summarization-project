"""
Paid Model Summarization (Claude, GPT-4).
Uses API calls to commercial LLMs for high-quality abstractive summarization.
"""

import os
from typing import Optional


class PaidModelSummarizer:
    """
    Summarizer using paid APIs (OpenAI GPT-4, Anthropic Claude).
    Requires API keys set as environment variables:
      - OPENAI_API_KEY
      - ANTHROPIC_API_KEY
    """

    SUPPORTED_MODELS = {
        "gpt-4": {
            "provider": "openai",
            "model_id": "gpt-4",
            "display_name": "GPT-4 (OpenAI)",
        },
        "gpt-4o-mini": {
            "provider": "openai",
            "model_id": "gpt-4o-mini",
            "display_name": "GPT-4o Mini (OpenAI)",
        },
        "claude-sonnet": {
            "provider": "anthropic",
            "model_id": "claude-sonnet-5",
            "display_name": "Claude Sonnet (Anthropic)",
        },
        "claude-haiku": {
            "provider": "anthropic",
            "model_id": "claude-haiku-4-5-20251001",
            "display_name": "Claude Haiku (Anthropic)",
        },
    }

    def __init__(self):
        self._openai_client = None
        self._anthropic_client = None

    def _get_openai_client(self):
        if self._openai_client is None:
            import certifi
            os.environ['SSL_CERT_FILE'] = certifi.where()
            os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
            from openai import OpenAI
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")
            self._openai_client = OpenAI(api_key=api_key)
        return self._openai_client

    def _get_anthropic_client(self):
        if self._anthropic_client is None:
            import certifi
            os.environ['SSL_CERT_FILE'] = certifi.where()
            os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
            from anthropic import Anthropic
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY environment variable not set")
            self._anthropic_client = Anthropic(api_key=api_key)
        return self._anthropic_client

    def get_available_models(self) -> list:
        """Return list of models that have valid API keys configured."""
        available = []
        if os.environ.get("OPENAI_API_KEY"):
            available.extend(["gpt-4", "gpt-4o-mini"])
        anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if anthropic_key and "your-key-here" not in anthropic_key:
            available.extend(["claude-sonnet", "claude-haiku"])
        return available

    def summarize(
        self,
        text: str,
        model_name: str = "gpt-4o-mini",
        lang: Optional[str] = None,
        ratio: float = 0.3,
    ) -> dict:
        """
        Generate a summary using a paid model API.

        Args:
            text: Input text to summarize
            model_name: One of the SUPPORTED_MODELS keys
            lang: Language code (for prompt tuning)
            ratio: Target compression ratio (used in prompt)

        Returns:
            Dictionary with summary and metadata
        """
        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(f"Unknown model: {model_name}. Supported: {list(self.SUPPORTED_MODELS.keys())}")

        config = self.SUPPORTED_MODELS[model_name]
        provider = config["provider"]

        word_count = len(text.split())
        target_words = max(50, int(word_count * ratio))

        # Build prompt
        lang_instruction = ""
        if lang and lang != "en":
            lang_instruction = f" Respond in the same language as the input text ({lang})."

        system_prompt = (
            "You are a professional text summarizer. Create a concise, accurate summary "
            "that captures all key points, main arguments, and important details. "
            "Do NOT invent or hallucinate any facts not present in the source text. "
            "Only include information that is explicitly stated in the text."
            f"{lang_instruction}"
        )

        user_prompt = (
            f"Summarize the following text in approximately {target_words} words. "
            f"Preserve the main ideas, key facts, and logical flow.\n\n"
            f"TEXT:\n{text}"
        )

        if provider == "openai":
            summary, usage = self._call_openai(config["model_id"], system_prompt, user_prompt)
        elif provider == "anthropic":
            summary, usage = self._call_anthropic(config["model_id"], system_prompt, user_prompt)
        else:
            raise ValueError(f"Unknown provider: {provider}")

        return {
            "summary": summary,
            # Real token counts from the provider, so cost is measured rather
            # than estimated (see backend/evaluation/costs.py).
            "usage": usage,
            "method": f"Paid API ({config['display_name']})",
            "model_id": config["model_id"],
            "provider": provider,
            "compression_ratio": len(summary.split()) / word_count if word_count > 0 else 0,
            "language": lang,
            "reliability": "high",
            "warning": None,
        }

    def _call_openai(self, model_id: str, system_prompt: str, user_prompt: str):
        """Call the OpenAI API. Returns (summary, token usage)."""
        client = self._get_openai_client()
        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=1000,
        )
        usage = getattr(response, "usage", None)
        return response.choices[0].message.content.strip(), {
            "input_tokens": getattr(usage, "prompt_tokens", None),
            "output_tokens": getattr(usage, "completion_tokens", None),
        }

    def _call_anthropic(self, model_id: str, system_prompt: str, user_prompt: str):
        """Call the Anthropic API. Returns (summary, token usage)."""
        client = self._get_anthropic_client()
        response = client.messages.create(
            model=model_id,
            max_tokens=1000,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_prompt},
            ],
        )
        usage = getattr(response, "usage", None)
        return response.content[0].text.strip(), {
            "input_tokens": getattr(usage, "input_tokens", None),
            "output_tokens": getattr(usage, "output_tokens", None),
        }


class ColabGPUClient:
    """
    Client that calls the Colab GPU server for heavy model summarization.
    Set COLAB_GPU_URL in .env to the ngrok URL from the Colab notebook.

    Models hosted on Colab: led-large-book, flan-t5-large
    """

    def __init__(self):
        self.base_url = os.environ.get("COLAB_GPU_URL", "").rstrip("/")
        self._cached_models = None
        self._cache_checked = False

    def is_available(self) -> bool:
        """Check if Colab GPU server is configured."""
        return bool(self.base_url)

    def _get_headers(self) -> dict:
        """Headers for ngrok free tier (bypasses browser warning page)."""
        return {
            "Content-Type": "application/json",
            "ngrok-skip-browser-warning": "true",
        }

    def get_available_models(self) -> list:
        """Query the Colab server for available models (cached after first call)."""
        if not self.is_available():
            return []
        if self._cache_checked:
            return self._cached_models or []
        try:
            import httpx
            r = httpx.get(
                f"{self.base_url}/health",
                timeout=10,
                verify=False,
                headers=self._get_headers(),
            )
            self._cache_checked = True
            if r.status_code == 200:
                data = r.json()
                self._cached_models = data.get("models", [])
                return self._cached_models
        except Exception:
            self._cache_checked = True
            self._cached_models = []
        return []

    def summarize(self, text: str, model: str = "led-large-book", max_length: int = 300, min_length: int = 50) -> dict:
        """Call Colab GPU server for summarization."""
        import httpx

        # Generous timeout: a long document chunked across several passes can
        # legitimately take minutes, and the first call to a model may still be
        # loading it from the session's disk cache.
        response = httpx.post(
            f"{self.base_url}/summarize",
            json={"text": text, "model": model, "max_length": max_length, "min_length": min_length},
            timeout=900,
            verify=False,
            headers=self._get_headers(),
        )
        if response.status_code != 200:
            raise Exception(f"Colab server error ({response.status_code}): {response.text[:300]}")
        try:
            data = response.json()
        except Exception:
            # Almost always the tunnel's own HTML error page rather than the
            # server's reply — say so plainly instead of a JSON parse error.
            raise Exception(
                f"Colab server returned a non-JSON response (HTTP {response.status_code}); "
                f"the tunnel or the model load probably timed out. Body: {response.text[:200]}"
            )
        if "error" in data:
            raise Exception(f"Colab model error: {data['error']}")
        summary = data["summary"]
        if "<n>" in summary:
            # Pegasus writes "<n>" for a line break; left in, it is counted as
            # a token by every metric.
            import re as _re

            summary = _re.sub(r"\s*<n>\s*", " ", summary).strip()
        return {
            "summary": summary,
            "method": f"GPU ({model})",
            "model_id": model,
            "compression_ratio": len(summary.split()) / len(text.split()) if text else 0,
            # Language is whatever the caller sent; the server is language-agnostic.
            "language": None,
            "reliability": "high",
            "warning": None,
            # Reported by the v2 server: GPU time excluding network round-trip,
            # plus how many chunks the hierarchical pass needed.
            "gpu_seconds": data.get("gpu_seconds"),
            "chunks": data.get("chunks"),
            "passes": data.get("passes"),
        }

    def summarize_batch(self, text: str, max_length: int = 300, min_length: int = 50) -> dict:
        """Run ALL models on the Colab GPU server."""
        import httpx
        response = httpx.post(
            f"{self.base_url}/summarize/batch",
            json={"text": text, "max_length": max_length, "min_length": min_length},
            timeout=300,
            verify=False,
            headers=self._get_headers(),
        )
        if response.status_code != 200:
            raise Exception(f"Colab server error ({response.status_code}): {response.text}")
        return response.json()["results"]
