#!/usr/bin/env python3
"""llm_cli.py — shared pluggable LLM backend wrapper.

Single source of truth for calling an LLM across CLIs/APIs, so both the filing AGENT
(bookmark_filing_assistant.py) and the RERANKER (llm_reranker.py) use the same backends.

    from llm_cli import call_llm
    text = call_llm(prompt, provider="claude", model="haiku")

Providers: claude (Haiku via `claude -p`, default), gemini, agy (Antigravity/Gemini-backed),
codex (OpenAI `codex exec`), openai, anthropic, ollama, ollama-json.
"""
import json
import ipaddress
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional
from urllib import request
from urllib.parse import urlsplit


class _NoRedirectHandler(request.HTTPRedirectHandler):
    """Reject redirects so private prompts never leave the validated origin."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


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


def resolve_ollama_cli() -> str:
    """Resolve native Ollama or the Windows CLI exposed to WSL.

    ``OLLAMA_CLI`` is the explicit override.  The Windows executable fallback
    lets WSL connect to the Windows Ollama service through loopback instead of
    requiring that service to listen on the WSL network interface.  Loopback
    transport does not prove that the selected Ollama tag executes locally;
    private-data callers must separately disable Ollama cloud features.
    """

    configured = os.environ.get("OLLAMA_CLI")
    if configured:
        return configured
    native = shutil.which("ollama")
    if native:
        return native
    local_app_data = os.environ.get("LOCALAPPDATA")
    if os.environ.get("WSL_DISTRO_NAME") and local_app_data:
        windows_cli = Path(local_app_data) / "Programs" / "Ollama" / "ollama.exe"
        if windows_cli.is_file():
            return str(windows_cli)
    return "ollama"


def resolve_windows_curl() -> Optional[str]:
    """Return Windows curl for WSL→Windows-loopback API calls, if available."""

    configured = os.environ.get("WINDOWS_CURL_CLI")
    if configured:
        return configured
    candidate = Path("/mnt/c/Windows/System32/curl.exe")
    if os.environ.get("WSL_DISTRO_NAME") and candidate.is_file():
        return str(candidate)
    return None


def local_ollama_endpoint() -> str:
    """Return a canonical loopback-only Ollama endpoint.

    Reject userinfo, proxies disguised as paths, and non-loopback hosts rather
    than relying on substring parsing of ``OLLAMA_HOST``.  This constrains only
    the client-to-server hop; Ollama cloud models can still be proxied by a
    loopback server unless cloud features are disabled on that server.
    """

    configured = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").strip()
    if "://" not in configured:
        configured = "http://" + configured
    parsed = urlsplit(configured)
    if (
        parsed.scheme.casefold() != "http"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("Ollama JSON endpoint must be a plain loopback HTTP origin")
    try:
        hostname = parsed.hostname
    except ValueError as exc:
        raise ValueError("Ollama JSON endpoint has an invalid hostname") from exc
    if not hostname:
        raise ValueError("Ollama JSON endpoint has no hostname")
    normalized_host = hostname.casefold()
    if normalized_host == "localhost":
        # Do not leave loopback isolation dependent on DNS or a hosts-file
        # mapping.  The transport always connects to a numeric loopback.
        normalized_host = "127.0.0.1"
    else:
        try:
            address = ipaddress.ip_address(normalized_host)
            if not address.is_loopback:
                raise ValueError("Ollama JSON endpoint is not loopback")
            normalized_host = str(address)
        except ValueError as exc:
            raise ValueError("Ollama JSON endpoint is not loopback") from exc
    try:
        port = parsed.port or 11434
    except ValueError as exc:
        raise ValueError("Ollama JSON endpoint has an invalid port") from exc
    rendered_host = f"[{normalized_host}]" if ":" in normalized_host else normalized_host
    return f"http://{rendered_host}:{port}"


def _require_local_ollama_model(model: str, timeout: int) -> Optional[str]:
    cli = resolve_ollama_cli()
    available = subprocess.run(
        [cli, "show", model],
        capture_output=True,
        text=True,
        timeout=min(timeout, 30),
    )
    if available.returncode != 0:
        print(
            f"  Ollama model is not installed locally: {model!r}",
            file=sys.stderr,
        )
        return None
    return cli


def call_ollama(prompt: str, model: str = "llama3.1", timeout: int = 120) -> Optional[str]:
    try:
        cli = _require_local_ollama_model(model, timeout)
        if cli is None:
            return None
        # `ollama run` pulls an absent model. Privacy/evaluation callers must
        # instead name a model that was deliberately installed beforehand.
        r = subprocess.run(
            [cli, "run", model, prompt],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if r.returncode == 0:
            return r.stdout.strip()
        print(f"  Ollama error: {r.stderr[:200]}", file=sys.stderr); return None
    except subprocess.TimeoutExpired:
        print(f"  Timeout after {timeout}s", file=sys.stderr); return None
    except FileNotFoundError:
        print("  Ollama not found (https://ollama.ai)", file=sys.stderr); return None
    except Exception as e:
        print(f"  Exception: {e}", file=sys.stderr); return None


def call_ollama_json(
    prompt: str,
    model: str = "llama3.1",
    timeout: int = 120,
    *,
    response_schema: Optional[dict] = None,
    max_output_tokens: int = 2048,
    think: bool = False,
    metadata_out: Optional[dict] = None,
) -> Optional[str]:
    """Call Ollama's non-streaming JSON API.

    The Windows CLI emits terminal animation bytes when captured from WSL.
    Calling the Windows-loopback API through Windows curl avoids that transport
    corruption and asks Ollama to constrain the model response to valid JSON.
    """

    try:
        if metadata_out is not None:
            if not isinstance(metadata_out, dict):
                raise ValueError("metadata_out must be a dict")
            metadata_out.clear()
        if (
            not isinstance(max_output_tokens, int)
            or isinstance(max_output_tokens, bool)
            or max_output_tokens <= 0
        ):
            raise ValueError("max_output_tokens must be a positive integer")
        endpoint = local_ollama_endpoint() + "/api/generate"
        ollama_cli = _require_local_ollama_model(model, timeout)
        if ollama_cli is None:
            return None
        payload = json.dumps(
            {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "format": response_schema if response_schema is not None else "json",
                "think": bool(think),
                "options": {
                    "temperature": 0,
                    "seed": 0,
                    "num_predict": max_output_tokens,
                },
            }
        ).encode("utf-8")
        windows_curl = (
            resolve_windows_curl()
            if str(ollama_cli).casefold().endswith(".exe")
            else None
        )
        if windows_curl:
            result = subprocess.run(
                [
                    windows_curl,
                    "--disable",
                    "--silent",
                    "--show-error",
                    "--fail-with-body",
                    "--noproxy",
                    "*",
                    "--max-redirs",
                    "0",
                    endpoint,
                    "-H",
                    "Content-Type: application/json",
                    "--data-binary",
                    "@-",
                ],
                input=payload,
                capture_output=True,
                timeout=timeout,
            )
            if result.returncode != 0:
                error = result.stderr.decode("utf-8", "replace")[:200]
                print(f"  Ollama JSON API error: {error}", file=sys.stderr)
                return None
            outer_text = result.stdout.decode("utf-8")
        else:
            api_request = request.Request(
                endpoint,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            opener = request.build_opener(
                request.ProxyHandler({}),
                _NoRedirectHandler(),
            )
            with opener.open(api_request, timeout=timeout) as response:
                outer_text = response.read().decode("utf-8")
        outer = json.loads(outer_text)
        if not isinstance(outer, dict):
            raise ValueError("Ollama response is not an object")
        if outer.get("error"):
            raise ValueError(f"Ollama returned an error: {outer['error']}")
        if outer.get("done") is not True:
            raise ValueError("Ollama response did not finish")
        done_reason = outer.get("done_reason")
        if done_reason == "length":
            raise ValueError("Ollama response exhausted the output-token budget")
        model_response = outer.get("response")
        if not isinstance(model_response, str):
            raise ValueError("Ollama response field is missing")
        json.loads(model_response)
        if metadata_out is not None:
            metadata_out.update(
                {
                    key: outer[key]
                    for key in (
                        "done_reason",
                        "eval_count",
                        "eval_duration",
                        "prompt_eval_count",
                        "prompt_eval_duration",
                        "load_duration",
                        "total_duration",
                    )
                    if key in outer
                }
            )
        return model_response.strip()
    except subprocess.TimeoutExpired:
        print(f"  Timeout after {timeout}s", file=sys.stderr); return None
    except FileNotFoundError:
        print("  Ollama/Windows curl bridge not found", file=sys.stderr); return None
    except Exception as e:
        print(f"  Ollama JSON exception: {e}", file=sys.stderr); return None


_DISPATCH = {"claude": call_claude_cli, "gemini": call_gemini_cli, "agy": call_agy_cli,
             "codex": call_codex_cli, "openai": call_openai_api, "anthropic": call_anthropic_api,
             "ollama": call_ollama, "ollama-json": call_ollama_json}
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
