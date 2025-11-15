"""Utility file for parsing OpenAI json responses."""
from __future__ import annotations

import json
import re
from typing import Any

import openai

from prompt2model.utils import api_tools, get_formatted_logger
from prompt2model.utils.api_tools import API_ERRORS, handle_api_error

logger = get_formatted_logger("ParseJsonResponses")


def _join_parts(parts: Any) -> str:
    """Join multi-part content (e.g., OpenAI/Anthropic style) into a string."""
    out = []
    for p in parts:
        if isinstance(p, dict) and p.get("type") == "text":
            out.append(p.get("text", ""))
        else:
            out.append(str(p))
    return "\n".join(out)


def extract_content(resp: Any) -> str:
    """Normalize any LLM response object into a plain string."""
    # 1) Already a plain string
    if isinstance(resp, str):
        return resp

    try:
        # 2) Dict-style payload (e.g., litellm/OpenAI-compatible)
        if isinstance(resp, dict):
            choices = resp.get("choices") or []
            if choices:
                first = choices[0]
                if isinstance(first, dict):
                    msg = first.get("message")
                    if isinstance(msg, dict):
                        content = msg.get("content")
                        if isinstance(content, list):
                            return _join_parts(content)
                        if content is not None:
                            return str(content)
                    # completion-style text field
                    if "text" in first and first["text"] is not None:
                        return str(first["text"])

            # Anthropic / generic content at top level
            content = resp.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return _join_parts(content)

        # 3) OpenAI SDK objects (have .choices attribute)
        if hasattr(resp, "choices"):
            choices = getattr(resp, "choices")
            if choices:
                first = choices[0]
                # object-style message
                msg = getattr(first, "message", None)
                if msg is not None:
                    content = getattr(msg, "content", None)
                    if isinstance(content, list):
                        return _join_parts(content)
                    if content is not None:
                        return str(content)
                # completion-style text
                text = getattr(first, "text", None)
                if text is not None:
                    return str(text)
    except Exception:
        # Fall through to generic stringification below
        pass

    # 4) Fallback: JSON-encode or plain str(), to avoid TypeError in regex code
    try:
        return json.dumps(resp, ensure_ascii=False)
    except Exception:
        return str(resp)


def find_and_parse_json(
    response: openai.Completion, required_keys: list, optional_keys: list = []
) -> dict | None:
    """Parse stuctured fields from the API response.

    In case there are multiple JSON objects in the response, take the final one.

    Args:
        response: API response.
        required_keys: Required keys from the response
        optional_keys: Optional keys from the response

    Returns:
        If the API response is a valid JSON object and contains the
        required and optional keys then returns the
        final response as a Dictionary
        Else returns None.
    """
    response_str = extract_content(response)
    correct_json = find_rightmost_brackets(response_str)

    if correct_json is None:
        logger.warning("No valid JSON found in the response.")
        return None

    try:
        response_json = json.loads(correct_json, strict=False)
    except json.decoder.JSONDecodeError:
        logger.warning(f"API response was not a valid JSON: {correct_json}")
        return None

    missing_keys = [key for key in required_keys if key not in response_json]
    if len(missing_keys) != 0:
        logger.warning(f'API response must contain {", ".join(required_keys)} keys')
        return None

    final_response = {}
    for key in required_keys + optional_keys:
        if key not in response_json:
            # This is an optional key, so exclude it from the final response.
            continue
        if type(response_json[key]) == str:
            final_response[key] = response_json[key].strip()
        else:
            final_response[key] = response_json[key]
    return final_response


def find_rightmost_brackets(text: str) -> str | None:
    """Find the rightmost complete set of brackets in a string."""
    stack = []
    for i, char in enumerate(reversed(text)):
        if char == "}":
            stack.append(len(text) - i - 1)
        elif char == "{" and stack:
            start = len(text) - i - 1
            end = stack.pop()
            if not stack:  # Found the rightmost complete set
                return text[start : end + 1]
    return None


def parse_dataset_config_responses(response: openai.ChatCompletion) -> dict:
    """Parse the response to extract relevant information from dataset/configuration.

    LLMs can return the dataset configuration in different formats -
    usually either between ** ** or as a sentence.

    Args:
        response: The response containing the dataset configuration.

    Returns:
        The extracted relevant information from the dataset configuration.
    """
    # Normalize to a string first to handle dicts / SDK objects uniformly.
    response_str = extract_content(response)

    pattern = r"\*\*(.*?)\*\*"

    match = re.search(pattern, response_str)
    dataset_config = ""
    if match:
        dataset_config = match.group(1)
    elif len(response_str.split()) >= 1:
        dataset_config = response_str.split()[-1].replace(".", "")

    # Clean up bracketed options like "[a] small" -> "small".
    cleaned = dataset_config.strip()
    cleaned = re.sub(r"^\s*\[[^\]]+\]\s*", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return {"name": cleaned}


def parse_prompt_to_fields(
    prompt: str,
    required_keys: list = [],
    optional_keys: list = [],
    max_api_calls: int = 5,
    module_name: str = "col_selection",
) -> dict[str, Any]:
    """Parse prompt into specific fields, and return to the calling function.

    This function calls the required api, has the logic for the retrying,
    passes the response to the parsing function, and return the
    response back or throws an error

    Args:
        prompt: User prompt into specific fields
        required_keys: Fields that need to be present in the response
        optional_keys: Field that may/may not be present in the response
        max_api_calls: Max number of retries, defaults to 5 to avoid
                        being stuck in an infinite loop
        module_name: The module this is to be used for. Currently supports
                        rerank and col_selection

    Returns:
        Parsed Response as a dictionary.

    Raises:
        ValueError: If max_api_calls is not greater than 0.
        RuntimeError: If the maximum number of API calls is reached.

    """
    chat_api = api_tools.default_api_agent
    if max_api_calls <= 0:
        raise ValueError("max_api_calls must be > 0.")

    api_call_counter = 0
    last_error = None
    while True:
        api_call_counter += 1
        try:
            response: openai.ChatCompletion | Exception = chat_api.generate_one_completion(  # type: ignore[assignment]
                prompt,
                temperature=0.01,
                presence_penalty=0,
                frequency_penalty=0,
            )
            extraction: dict[str, Any] | None = None
            if module_name == "col_selection":
                extraction = find_and_parse_json(response, required_keys, optional_keys)

            elif module_name == "rerank":
                extraction = parse_dataset_config_responses(response)
            if extraction is not None:
                return extraction
        except API_ERRORS as e:
            last_error = e
            handle_api_error(e, backoff_duration=2**api_call_counter)

        if api_call_counter >= max_api_calls:
            # In case we reach maximum number of API calls, we raise an error.
            logger.error("Maximum number of API calls reached.")
            # Best-effort: log truncated raw response for debugging, if available.
            try:
                raw_text = extract_content(response)  # type: ignore[arg-type]
                logger.debug("Last LLM raw response (truncated 200): %s", raw_text[:200])
            except Exception:
                pass
            raise RuntimeError("Maximum number of API calls reached.") from last_error


def make_single_api_request(prompt: str, max_api_calls: int = 10) -> str:
    """Prompts an LLM using the APIAgent, and returns the response.

    This function calls the required api, has the logic for retrying,
    returns the response back or throws an error
    Args:
        prompt: User prompt into specific fields
        max_api_calls: Max number of retries, defaults to 5 to avoid
                        being stuck in an infinite loop
    Returns:
        Response text or throws error
    """
    chat_api = api_tools.default_api_agent
    if max_api_calls <= 0:
        raise ValueError("max_api_calls must be > 0.")

    api_call_counter = 0
    last_error = None
    while True:
        api_call_counter += 1
        try:
            response: openai.ChatCompletion = chat_api.generate_one_completion(  # type: ignore[assignment]
                prompt=prompt,
                temperature=0.01,
                presence_penalty=0,
                frequency_penalty=0,
            )
            if response is not None:
                return extract_content(response)

        except API_ERRORS as e:
            last_error = e
            handle_api_error(e, backoff_duration=2**api_call_counter)

        if api_call_counter >= max_api_calls:
            # In case we reach maximum number of API calls, we raise an error.
            logger.error("Maximum number of API calls reached.")
            raise RuntimeError("Maximum number of API calls reached.") from last_error
