# Structured output parser - extract JSON from model responses.

import json

class ParsedOutput:
    """Result of parsing a model response for structured output."""
    __slots__ = ("json_blocks", "raw_text", "errors")

    def __init__(self, json_blocks: list[dict], raw_text: str, errors: list[str]):
        self.json_blocks = json_blocks
        self.raw_text = raw_text
        self.errors = errors

class OutputParser:
    """Extract and validate JSON from model responses."""

    @classmethod
    def _extract_fenced(cls, text: str) -> list[dict]:
        """Extract JSON from fenced code blocks (```json ... ```)."""
        results = []
        marker = "```"
        pos = 0
        while True:
            start = text.find(marker, pos)
            if start < 0:
                break
            # Skip optional language tag
            line_end = text.find("\n", start)
            if line_end < 0:
                break
            end = text.find(marker, line_end)
            if end < 0:
                break
            block = text[line_end + 1:end].strip()
            try:
                parsed = json.loads(block)
                if isinstance(parsed, dict):
                    results.append(parsed)
                elif isinstance(parsed, list):
                    results.extend(d for d in parsed if isinstance(d, dict))
            except (json.JSONDecodeError, ValueError):
                pass
            pos = end + len(marker)
        return results

    @classmethod
    def _extract_bare(cls, text: str) -> list[dict]:
        """Extract bare JSON objects from text."""
        results = []
        depth = 0
        start = -1
        for i, ch in enumerate(text):
            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and start >= 0:
                    try:
                        parsed = json.loads(text[start:i + 1])
                        if isinstance(parsed, dict):
                            results.append(parsed)
                    except (json.JSONDecodeError, ValueError):
                        pass
                    start = -1
        return results

    @classmethod
    def extract_json(cls, text: str) -> list[dict]:
        """Extract all JSON blocks from text (fenced first, then bare)."""
        results = cls._extract_fenced(text)
        if results:
            return results
        return cls._extract_bare(text)

    @classmethod
    def parse_response(cls, text: str, expected_keys: list[str] | None = None) -> ParsedOutput:
        """Parse response, optionally validating expected top-level keys."""
        blocks = cls.extract_json(text)
        errors = []
        if expected_keys and blocks:
            for i, block in enumerate(blocks):
                missing = [k for k in expected_keys if k not in block]
                if missing:
                    errors.append(f"Block {i}: missing keys {missing}")
        return ParsedOutput(json_blocks=blocks, raw_text=text, errors=errors)
