"""Shared defensive JSON parsing for LLM output.

LLMs occasionally ignore "return ONLY valid JSON, no prose, no markdown
fences" and wrap the JSON in a preamble ("Here is the extracted JSON
data:\\n\\n{...}\\n") or markdown fences anyway, even at temperature=0.
extract_json_object() is used everywhere in this codebase that parses raw
LLM output as JSON (extractor.py, matcher.py's explain_match) so there's
one place to make parsing more forgiving instead of duplicating ad-hoc
fixups per call site.
"""

import json
import re

_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_TRAILING_COMMA_RE = re.compile(r",\s*([\}\]])")


def extract_json_object(text: str) -> dict:
    """Parse a JSON object out of raw LLM text, trying progressively more
    forgiving strategies:
      (a) parse the text as-is
      (b) look for a ```json ... ``` fenced block and parse its contents
      (c) find the first `{` and last `}` in the text (handles prose
          preambles/postambles around the JSON) and parse that slice,
          after stripping trailing commas before closing brackets/braces
          (a separate common LLM artifact, worth guarding against even
          when it isn't what caused a given failure)

    Raises json.JSONDecodeError with a clear message if none of these
    produce valid JSON.
    """
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fence_match = _FENCE_RE.search(text)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except json.JSONDecodeError:
            pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end + 1]
        candidate = _TRAILING_COMMA_RE.sub(r"\1", candidate)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    raise json.JSONDecodeError(
        f"Could not extract a valid JSON object from LLM output: {text[:200]!r}",
        text, 0,
    )
