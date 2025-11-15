"""Tools for accessing API-based models (OpenAI/litellm or local Ollama)."""

from __future__ import annotations  # noqa FI58

import asyncio
import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional

import aiolimiter
import tiktoken
from aiohttp import ClientSession
from tqdm.asyncio import tqdm_asyncio

# ---- Optional litellm / openai imports (kept for backward compat) ----
# We keep them optional so Ollama-only users don't need these packages configured.
try:
    import openai  # type: ignore
    from litellm import acompletion, completion  # type: ignore
    import litellm.utils  # type: ignore
except Exception:  # pragma: no cover
    openai = None
    acompletion = None
    completion = None

# =========================
# Error + Retry Definitions
# =========================

class _DummyOpenAIError(Exception):
    pass

# Build a permissive set of API error classes (works even if openai is missing)
_APIErrorBase = getattr(openai, "APIError", _DummyOpenAIError) if openai else _DummyOpenAIError
_APITimeoutError = getattr(openai, "APITimeoutError", _DummyOpenAIError) if openai else _DummyOpenAIError
_RateLimitError = getattr(openai, "RateLimitError", _DummyOpenAIError) if openai else _DummyOpenAIError
_BadRequestError = getattr(openai, "BadRequestError", _DummyOpenAIError) if openai else _DummyOpenAIError
_APIStatusError = getattr(openai, "APIStatusError", _DummyOpenAIError) if openai else _DummyOpenAIError
_APIConnectionError = getattr(openai, "APIConnectionError", _DummyOpenAIError) if openai else _DummyOpenAIError

API_ERRORS = (
    _APIErrorBase,
    _APITimeoutError,
    _RateLimitError,
    _BadRequestError,
    _APIStatusError,
    json.decoder.JSONDecodeError,
    AssertionError,
)

ERROR_ERRORS_TO_MESSAGES: Dict[Any, str] = {
    _BadRequestError: "API Invalid Request: Prompt was filtered",
    _RateLimitError: "API rate limit exceeded. Sleeping for 10 seconds.",
    _APIConnectionError: "Error Communicating with API",
    _APITimeoutError: "API Timeout Error: API Timeout",
    _APIStatusError: "API service unavailable error: {e}",
    _APIErrorBase: "API error: {e}",
}
BUFFER_DURATION = 2

class AttrDict(dict):
    """Dict subclass that also allows attribute-style access.

    This is used to adapt plain dict API responses so existing code that expects
    ``resp.choices`` or nested ``.message.content`` continues to work.
    """

    def __getattr__(self, item: str) -> Any:
        try:
            return self[item]
        except KeyError as e:  # pragma: no cover - trivial
            raise AttributeError(item) from e


def _to_attrdict(obj: Any) -> Any:
    """Recursively convert dicts/lists into AttrDict containers."""
    if isinstance(obj, dict):
        return AttrDict({k: _to_attrdict(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_attrdict(v) for v in obj]
    return obj


# =========================
# Utility
# =========================

def count_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    """Count tokens with tiktoken; used as a conservative estimate."""
    try:
        encoding = tiktoken.get_encoding(encoding_name)
        return len(encoding.encode(string))
    except Exception:
        # Fallback: rough char-based estimate
        return max(1, len(string) // 4)

def _wrap_text_as_openai_like(content: str) -> Dict[str, Any]:
    """Wrap a plain string into an OpenAI-like response dict."""
    return {"choices": [{"message": {"content": content}}]}

def handle_api_error(e, backoff_duration=1) -> None:
    """Handle API errors with backoff; raise if not recognized."""
    logging.error(e)
    if not isinstance(e, API_ERRORS):
        raise e

    if isinstance(e, (_APIErrorBase, _APITimeoutError, _RateLimitError)):
        match = re.search(r"Please retry after (\d+) seconds", str(e))
        if match is not None:
            backoff_duration = int(match.group(1)) + BUFFER_DURATION
        logging.info(f"Retrying in {backoff_duration} seconds...")
        time.sleep(backoff_duration)

# =========================
# API Agent
# =========================

class APIAgent:
    """
    Unified agent for:
      - litellm/OpenAI-compatible chat models (needs OPENAI_API_KEY)
      - local Ollama REST models (no Internet needed)

    Auto backend selection:
      1) If P2M_BACKEND=ollama -> use Ollama
      2) Else if OPENAI_API_KEY present and litellm available -> use litellm/OpenAI
      3) Else try Ollama (best-effort)
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        max_tokens: Optional[int] = 4000,
        api_base: Optional[str] = None,
        ollama_base: Optional[str] = None,
    ):
        self.model_name = model_name or os.getenv("P2M_GEN_MODEL", "llama3.1:8b")
        self.max_tokens = max_tokens
        self.api_base = api_base  # for OpenAI-like endpoints (optional)

        # Backend selection
        force_backend = os.getenv("P2M_BACKEND", "").strip().lower()
        has_openai_key = bool(os.getenv("OPENAI_API_KEY"))
        litellm_ready = (completion is not None and acompletion is not None and openai is not None)

        if force_backend == "ollama":
            self.backend = "ollama"
        elif has_openai_key and litellm_ready:
            self.backend = "litellm"
        else:
            # Default to Ollama if no OpenAI key / litellm not configured
            self.backend = "ollama"

        # Ollama settings
        base = ollama_base or os.getenv("P2M_OLLAMA_BASE", "http://localhost:11434")
        self.ollama_base = base.rstrip("/")
        self.ollama_num_predict = int(os.getenv("P2M_OLLAMA_NUM_PREDICT", "512"))

        # Infer max_tokens via litellm if available (nice-to-have)
        if self.backend == "litellm" and self.max_tokens is None:
            try:
                mt = litellm.utils.get_max_tokens(self.model_name)  # type: ignore
                if isinstance(mt, dict):
                    mt = mt.get("max_tokens", None)
                self.max_tokens = mt  # type: ignore
            except Exception:
                pass

        logging.info(f"[APIAgent] backend={self.backend}, model={self.model_name}")

    # ----------- OpenAI/litellm path -----------
    def _completion_litellm(
        self,
        prompt: str,
        temperature: float,
        presence_penalty: float,
        frequency_penalty: float,
        token_buffer: int,
    ) -> Dict[str, Any]:
        assert completion is not None, "litellm not installed"
        num_prompt_tokens = count_tokens_from_string(prompt)
        if self.max_tokens:
            max_tokens = max(64, self.max_tokens - num_prompt_tokens - token_buffer)
        else:
            max_tokens = max(64, 3 * num_prompt_tokens)

        resp = completion(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            api_base=self.api_base,
            temperature=temperature,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            max_tokens=max_tokens,
        )
        # Normalize to OpenAI-like dict (litellm already returns that)
        return resp  # type: ignore

    async def _acompletion_litellm_batch(
        self,
        prompts: List[str],
        temperature: float,
        responses_per_request: int,
        requests_per_minute: int,
        token_buffer: int,
        show_progress: bool = True,
    ) -> List[Dict[str, Any]]:
        assert acompletion is not None, "litellm not installed"
        limiter = aiolimiter.AsyncLimiter(requests_per_minute)

        num_prompt_tokens = max(count_tokens_from_string(p) for p in prompts)
        if self.max_tokens:
            max_tokens = max(64, self.max_tokens - num_prompt_tokens - token_buffer)
        else:
            max_tokens = max(64, 3 * num_prompt_tokens)

        async def call_one(prompt: str):
            async with limiter:
                # 3 retries with backoff
                for _ in range(3):
                    try:
                        return await acompletion(
                            model=self.model_name,
                            messages=[{"role": "user", "content": prompt}],
                            api_base=self.api_base,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            n=responses_per_request,
                            top_p=1,
                        )
                    except tuple(ERROR_ERRORS_TO_MESSAGES.keys()) as e:  # type: ignore
                        msg = ERROR_ERRORS_TO_MESSAGES.get(type(e), f"API error: {e}")
                        logging.warning(msg if "{e}" not in msg else msg.format(e=e))
                        await asyncio.sleep(10)
                return _wrap_text_as_openai_like("")

        tasks = [call_one(p) for p in prompts]
        if show_progress:
            return await tqdm_asyncio.gather(*tasks)
        # plain asyncio.gather 防止在外层循环中刷满控制台
        return await asyncio.gather(*tasks)

    # ----------- Ollama path -----------
    def _completion_ollama(
        self,
        prompt: str,
        temperature: float,
        presence_penalty: float,
        frequency_penalty: float,
    ) -> Dict[str, Any]:
        """
        POST /api/generate
        body: {model, prompt, stream=false, options={temperature, num_predict, ...}}
        resp: {"model": "...", "response": "...", ...}
        """
        import requests

        url = f"{self.ollama_base}/api/generate"
        options = {
            "temperature": float(temperature),
            "num_predict": int(self.ollama_num_predict),
        }
        # presence_penalty/frequency_penalty not directly supported by Ollama -> ignore

        r = requests.post(
            url,
            json={
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": options,
            },
            timeout=120,
        )
        r.raise_for_status()
        data = r.json()
        text = data.get("response", "")
        return _wrap_text_as_openai_like(text)

    async def _acompletion_ollama_batch(
        self,
        prompts: List[str],
        temperature: float,
        responses_per_request: int,
        requests_per_minute: int,
        show_progress: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        For Ollama we call /api/generate once per prompt.
        (responses_per_request is emulated by 1 call; you can loop to sample multiple if needed.)
        """
        limiter = aiolimiter.AsyncLimiter(requests_per_minute)

        async def call_one(session: ClientSession, prompt: str) -> Dict[str, Any]:
            url = f"{self.ollama_base}/api/generate"
            options = {
                "temperature": float(temperature),
                "num_predict": int(self.ollama_num_predict),
            }
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": options,
            }
            # 3 retries
            for _ in range(3):
                async with limiter:
                    try:
                        async with session.post(url, json=payload, timeout=120) as resp:
                            resp.raise_for_status()
                            data = await resp.json()
                            text = data.get("response", "")
                            return _wrap_text_as_openai_like(text)
                    except Exception as e:  # network / HTTP
                        logging.warning(f"Ollama request failed: {e}. Retry in 5s.")
                        await asyncio.sleep(5)
            return _wrap_text_as_openai_like("")

        async with ClientSession() as session:
            tasks = [call_one(session, p) for p in prompts]
            if show_progress:
                return await tqdm_asyncio.gather(*tasks)
            return await asyncio.gather(*tasks)

    # ----------- Public methods (uniform) -----------
    def generate_one_completion(
        self,
        prompt: str,
        temperature: float = 0.0,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        token_buffer: int = 300,
        **_: dict,
    ) -> Dict[str, Any]:
        """Synchronous single completion; returns OpenAI-like dict."""
        if self.backend == "litellm":
            completion_dict = self._completion_litellm(
                prompt, temperature, presence_penalty, frequency_penalty, token_buffer
            )
        else:
            completion_dict = self._completion_ollama(
                prompt, temperature, presence_penalty, frequency_penalty
            )
        # Wrap dict responses so downstream code can use attribute or key access.
        if isinstance(completion_dict, dict):
            return _to_attrdict(completion_dict)
        return completion_dict

    async def generate_batch_completion(
        self,
        prompts: List[str],
        temperature: float = 1.0,
        responses_per_request: int = 5,
        requests_per_minute: int = 80,
        token_buffer: int = 300,
        show_progress: bool = True,
        **_: dict,
    ) -> List[Dict[str, Any]]:
        """Async batch completion; returns list of OpenAI-like dicts."""
        if self.backend == "litellm":
            return await self._acompletion_litellm_batch(
                prompts,
                temperature,
                responses_per_request,
                requests_per_minute,
                token_buffer,
                show_progress=show_progress,
            )
        else:
            # For Ollama we ignore responses_per_request>1, because /api/generate returns one response.
            return await self._acompletion_ollama_batch(
                prompts,
                temperature,
                responses_per_request,
                requests_per_minute,
                show_progress=show_progress,
            )

# Default agent used across the project
default_api_agent = APIAgent(max_tokens=4000)
