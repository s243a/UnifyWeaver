#!/usr/bin/env python3
"""llm_cli.py — shared pluggable LLM backend wrapper.

Single source of truth for calling an LLM across CLIs/APIs, so both the filing AGENT
(bookmark_filing_assistant.py) and the RERANKER (llm_reranker.py) use the same backends.

    from llm_cli import call_llm
    text = call_llm(prompt, provider="claude", model="haiku")

Providers: claude (Haiku via `claude -p`, default), gemini, agy (Antigravity/Gemini-backed),
codex (OpenAI `codex exec`), openai, anthropic, ollama.
"""
import subprocess
import sys
from pathlib import Path
from typing import Optional


def call_claude_cli(prompt: str, model: str = "haiku", timeout: int = 60) -> Optional[str]:
    """`claude -p --model <model> <prompt>` — cheapest with a Claude subscription. Default Haiku."""
    try:
        cmd = ["claude", "-p"]
        if model:
            cmd += ["--model", model]
        cmd.append(prompt)
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=str(Path.cwd()))
        if r.returncode == 0:
            return r.stdout.strip()
        print(f"  Claude CLI error: {r.stderr[:200]}", file=sys.stderr)
        return None
    except subprocess.TimeoutExpired:
        print(f"  Timeout after {timeout}s", file=sys.stderr); return None
    except FileNotFoundError:
        print("  Claude CLI not found (npm i -g @anthropic-ai/claude-code)", file=sys.stderr); return None
    except Exception as e:
        print(f"  Exception: {e}", file=sys.stderr); return None


def call_gemini_cli(prompt: str, model: str = "gemini-2.0-flash", timeout: int = 120) -> Optional[str]:
    """`gemini -p <prompt> -m <model> --output-format text`."""
    try:
        cmd = ["gemini", "-p", prompt, "--output-format", "text"]
        if model:
            cmd += ["-m", model]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=str(Path.cwd()))
        if r.returncode == 0:
            return r.stdout.strip()
        print(f"  Gemini CLI error: {r.stderr[:200]}", file=sys.stderr)
        return None
    except subprocess.TimeoutExpired:
        print(f"  Timeout after {timeout}s", file=sys.stderr); return None
    except FileNotFoundError:
        print("  Gemini CLI not found", file=sys.stderr); return None
    except Exception as e:
        print(f"  Exception: {e}", file=sys.stderr); return None


def call_agy_cli(prompt: str, model: str = "", timeout: int = 120) -> Optional[str]:
    """Antigravity (agy) CLI — Gemini-backed. `agy -p <prompt> [--model M]`. Empty model → CLI default.
    SECURITY NOTE: passes --dangerously-skip-permissions (auto-approves tool use) so a rerank/eval batch runs
    unattended — fine for a trusted local ranking prompt, but be aware it changes the security posture."""
    try:
        cmd = ["agy", "-p", prompt, "--dangerously-skip-permissions"]
        if model:
            cmd += ["--model", model]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=str(Path.cwd()))
        if r.returncode == 0:
            return r.stdout.strip()
        print(f"  agy CLI error: {r.stderr[:200]}", file=sys.stderr)
        return None
    except subprocess.TimeoutExpired:
        print(f"  Timeout after {timeout}s", file=sys.stderr); return None
    except FileNotFoundError:
        print("  agy (Antigravity) CLI not found", file=sys.stderr); return None
    except Exception as e:
        print(f"  Exception: {e}", file=sys.stderr); return None


def call_codex_cli(prompt: str, model: str = "", timeout: int = 120) -> Optional[str]:
    """Codex CLI (OpenAI) non-interactive via `codex exec`. NOTE: needs node>=22 on PATH; flag syntax
    varies by version — adjust `cmd` if it errors."""
    try:
        cmd = ["codex", "exec", "--skip-git-repo-check"]
        if model:
            cmd += ["-m", model]
        cmd.append(prompt)
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=str(Path.cwd()))
        if r.returncode == 0:
            return r.stdout.strip()
        print(f"  codex CLI error: {r.stderr[:200]}", file=sys.stderr)
        return None
    except subprocess.TimeoutExpired:
        print(f"  Timeout after {timeout}s", file=sys.stderr); return None
    except FileNotFoundError:
        print("  codex CLI not found", file=sys.stderr); return None
    except Exception as e:
        print(f"  Exception: {e}", file=sys.stderr); return None


def call_openai_api(prompt: str, model: str = "gpt-4o-mini", timeout: int = 60) -> Optional[str]:
    try:
        import openai
        client = openai.OpenAI()
        resp = client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}],
                                              temperature=0.1, timeout=timeout)
        return resp.choices[0].message.content.strip()
    except ImportError:
        print("  OpenAI package not installed (pip install openai)", file=sys.stderr); return None
    except Exception as e:
        print(f"  OpenAI API error: {e}", file=sys.stderr); return None


def call_anthropic_api(prompt: str, model: str = "claude-3-5-haiku-20241022", timeout: int = 60) -> Optional[str]:
    try:
        import anthropic
        client = anthropic.Anthropic()
        msg = client.messages.create(model=model, max_tokens=1024,
                                     messages=[{"role": "user", "content": prompt}])
        return msg.content[0].text.strip()
    except ImportError:
        print("  Anthropic package not installed (pip install anthropic)", file=sys.stderr); return None
    except Exception as e:
        print(f"  Anthropic API error: {e}", file=sys.stderr); return None


def call_ollama(prompt: str, model: str = "llama3.1", timeout: int = 120) -> Optional[str]:
    try:
        r = subprocess.run(["ollama", "run", model, prompt], capture_output=True, text=True, timeout=timeout)
        if r.returncode == 0:
            return r.stdout.strip()
        print(f"  Ollama error: {r.stderr[:200]}", file=sys.stderr); return None
    except subprocess.TimeoutExpired:
        print(f"  Timeout after {timeout}s", file=sys.stderr); return None
    except FileNotFoundError:
        print("  Ollama not found (https://ollama.ai)", file=sys.stderr); return None
    except Exception as e:
        print(f"  Exception: {e}", file=sys.stderr); return None


_DISPATCH = {"claude": call_claude_cli, "gemini": call_gemini_cli, "agy": call_agy_cli,
             "codex": call_codex_cli, "openai": call_openai_api, "anthropic": call_anthropic_api,
             "ollama": call_ollama}
PROVIDERS = list(_DISPATCH)

# ── token accounting ── so a harness can measure total tokens per completed task (tokens-per-correct-filing).
# Estimate = chars/4 (model-agnostic, first-order); good enough to compare configs. Reset per task, read after.
_USAGE = {"calls": 0, "prompt_chars": 0, "completion_chars": 0}

def reset_usage():
    _USAGE.update(calls=0, prompt_chars=0, completion_chars=0)

def get_usage() -> dict:
    """Cumulative since the last reset_usage(): calls + char/token estimates (prompt, completion, total)."""
    pt, ct = _USAGE["prompt_chars"] // 4, _USAGE["completion_chars"] // 4
    return {"calls": _USAGE["calls"], "prompt_tokens_est": pt, "completion_tokens_est": ct,
            "total_tokens_est": pt + ct}


def call_llm(prompt: str, provider: str = "claude", model: str = "haiku", timeout: int = 60) -> Optional[str]:
    """Route to the chosen provider. Returns the model's text, or None on failure. Accumulates token usage
    (see get_usage/reset_usage) so callers can measure cost per task."""
    fn = _DISPATCH.get(provider)
    if fn is None:
        print(f"  Unknown provider: {provider} (choices: {PROVIDERS})", file=sys.stderr)
        return None
    resp = fn(prompt, model, timeout)
    _USAGE["calls"] += 1
    _USAGE["prompt_chars"] += len(prompt or "")
    _USAGE["completion_chars"] += len(resp or "")
    return resp
