"""Cost tracking for API usage."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
import json
import os
import sys
import time
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError


# Pricing per 1M tokens (as of early 2025, approximate)
# Users should update these as prices change
DEFAULT_PRICING = {
    # Claude models
    "claude-opus-4-20250514": {"input": 15.0, "output": 75.0},
    "claude-sonnet-4-20250514": {"input": 3.0, "output": 15.0},
    "claude-haiku-3-5-20241022": {"input": 0.80, "output": 4.0},
    # Aliases
    "opus": {"input": 15.0, "output": 75.0},
    "sonnet": {"input": 3.0, "output": 15.0},
    "haiku": {"input": 0.80, "output": 4.0},
    # OpenAI models
    "gpt-4o": {"input": 2.50, "output": 10.0},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4-turbo": {"input": 10.0, "output": 30.0},
    "gpt-4": {"input": 30.0, "output": 60.0},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    # Gemini (free tier has limits, these are paid tier)
    "gemini-2.5-flash": {"input": 0.075, "output": 0.30},
    "gemini-2.5-pro": {"input": 1.25, "output": 5.0},
    # Local models (free)
    "llama3": {"input": 0.0, "output": 0.0},
    "codellama": {"input": 0.0, "output": 0.0},
    "mistral": {"input": 0.0, "output": 0.0},
}


@dataclass
class UsageRecord:
    """Record of a single API call."""
    timestamp: str
    model: str
    input_tokens: int
    output_tokens: int
    input_cost: float
    output_cost: float
    total_cost: float


@dataclass
class CostTracker:
    """Track API costs for a session."""

    pricing: dict = field(default_factory=lambda: DEFAULT_PRICING.copy())
    records: list[UsageRecord] = field(default_factory=list)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost: float = 0.0

    def record_usage(self, model: str, input_tokens: int, output_tokens: int) -> UsageRecord:
        """Record token usage and calculate cost."""
        # Get pricing for model
        pricing = self.pricing.get(model, {"input": 0.0, "output": 0.0})

        # Calculate costs (pricing is per 1M tokens)
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        total_cost = input_cost + output_cost

        # Create record
        record = UsageRecord(
            timestamp=datetime.now().isoformat(),
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=total_cost
        )

        # Update totals
        self.records.append(record)
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cost += total_cost

        return record

    def get_summary(self) -> dict:
        """Get a summary of costs."""
        return {
            "total_requests": len(self.records),
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "total_cost_usd": round(self.total_cost, 6),
            "cost_formatted": f"${self.total_cost:.4f}"
        }

    def format_status(self) -> str:
        """Format cost status for display."""
        summary = self.get_summary()
        return (
            f"Tokens: {summary['total_input_tokens']:,} in / "
            f"{summary['total_output_tokens']:,} out | "
            f"Cost: {summary['cost_formatted']}"
        )

    def reset(self) -> None:
        """Reset all tracking."""
        self.records.clear()
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "summary": self.get_summary(),
            "records": [
                {
                    "timestamp": r.timestamp,
                    "model": r.model,
                    "input_tokens": r.input_tokens,
                    "output_tokens": r.output_tokens,
                    "input_cost": r.input_cost,
                    "output_cost": r.output_cost,
                    "total_cost": r.total_cost
                }
                for r in self.records
            ]
        }

    def save(self, path: str | Path) -> None:
        """Save cost data to JSON file."""
        path = Path(path)
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "CostTracker":
        """Load cost data from JSON file."""
        path = Path(path)
        data = json.loads(path.read_text())

        tracker = cls()
        for r in data.get("records", []):
            record = UsageRecord(
                timestamp=r["timestamp"],
                model=r["model"],
                input_tokens=r["input_tokens"],
                output_tokens=r["output_tokens"],
                input_cost=r["input_cost"],
                output_cost=r["output_cost"],
                total_cost=r["total_cost"]
            )
            tracker.records.append(record)
            tracker.total_input_tokens += record.input_tokens
            tracker.total_output_tokens += record.output_tokens
            tracker.total_cost += record.total_cost

        return tracker

    def set_pricing(self, model: str, input_price: float, output_price: float) -> None:
        """Set custom pricing for a model (per 1M tokens)."""
        self.pricing[model] = {"input": input_price, "output": output_price}

    def ensure_pricing(self, model: str) -> bool:
        """Ensure pricing exists for a model. Fetch from OpenRouter if needed.

        Returns True if pricing is available (even if zero).
        """
        if model in self.pricing:
            return True
        # Try OpenRouter lookup
        pricing = fetch_openrouter_pricing(model)
        if pricing:
            self.pricing[model] = pricing
            return True
        return False


# --- OpenRouter pricing ---

_OPENROUTER_CACHE_DIR = Path(os.environ.get(
    'AGENT_LOOP_CACHE', os.path.expanduser('~/.agent-loop/cache')
))
_OPENROUTER_CACHE_FILE = _OPENROUTER_CACHE_DIR / 'openrouter_pricing.json'
_OPENROUTER_CACHE_TTL = 86400  # 1 day


def _load_openrouter_cache() -> dict | None:
    """Load cached OpenRouter pricing if fresh enough."""
    try:
        if not _OPENROUTER_CACHE_FILE.exists():
            return None
        age = time.time() - _OPENROUTER_CACHE_FILE.stat().st_mtime
        if age > _OPENROUTER_CACHE_TTL:
            return None
        return json.loads(_OPENROUTER_CACHE_FILE.read_text())
    except Exception:
        return None


def _save_openrouter_cache(pricing: dict) -> None:
    """Save OpenRouter pricing to cache."""
    try:
        _OPENROUTER_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _OPENROUTER_CACHE_FILE.write_text(json.dumps(pricing))
    except Exception:
        pass


def fetch_openrouter_pricing(model_id: str) -> dict | None:
    """Fetch pricing for a model from OpenRouter's API.

    Returns dict with 'input' and 'output' keys (per 1M tokens),
    or None if not found.
    """
    # Check cache first
    cache = _load_openrouter_cache()
    if cache and model_id in cache:
        return cache[model_id]

    # Fetch from API
    try:
        req = Request(
            'https://openrouter.ai/api/v1/models',
            headers={'Content-Type': 'application/json'}
        )
        with urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
    except (URLError, json.JSONDecodeError, OSError) as e:
        print(f"  [OpenRouter pricing fetch failed: {e}]", file=sys.stderr)
        return None

    # Parse all models into cache format: {model_id: {input: X, output: Y}}
    pricing_cache = {}
    for m in data.get('data', []):
        mid = m.get('id', '')
        p = m.get('pricing', {})
        prompt_per_token = float(p.get('prompt', '0') or '0')
        completion_per_token = float(p.get('completion', '0') or '0')
        # Convert per-token to per-1M-tokens
        pricing_cache[mid] = {
            'input': round(prompt_per_token * 1_000_000, 4),
            'output': round(completion_per_token * 1_000_000, 4),
        }

    _save_openrouter_cache(pricing_cache)

    return pricing_cache.get(model_id)
