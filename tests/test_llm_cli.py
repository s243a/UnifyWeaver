#!/usr/bin/env python3
"""Focused tests for the shared local-LLM bridge."""

from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import subprocess
import sys
import threading


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import llm_cli


def test_resolve_ollama_cli_uses_windows_executable_from_wsl(tmp_path, monkeypatch):
    local_app_data = tmp_path / "AppData" / "Local"
    windows_cli = local_app_data / "Programs" / "Ollama" / "ollama.exe"
    windows_cli.parent.mkdir(parents=True)
    windows_cli.write_bytes(b"placeholder")
    monkeypatch.delenv("OLLAMA_CLI", raising=False)
    monkeypatch.setenv("WSL_DISTRO_NAME", "Ubuntu")
    monkeypatch.setenv("LOCALAPPDATA", str(local_app_data))
    monkeypatch.setattr(llm_cli.shutil, "which", lambda _name: None)

    assert llm_cli.resolve_ollama_cli() == str(windows_cli)


def test_call_ollama_requires_preinstalled_model(monkeypatch):
    calls = []

    def fake_run(command, **_kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 1, "", "model not found")

    monkeypatch.setattr(llm_cli, "resolve_ollama_cli", lambda: "/fake/ollama.exe")
    monkeypatch.setattr(llm_cli.subprocess, "run", fake_run)

    assert llm_cli.call_ollama("synthetic prompt", "phi3:missing", 60) is None
    assert calls == [["/fake/ollama.exe", "show", "phi3:missing"]]


def test_call_ollama_runs_installed_model(monkeypatch):
    calls = []

    def fake_run(command, **_kwargs):
        calls.append(command)
        if command[1] == "show":
            return subprocess.CompletedProcess(command, 0, "installed", "")
        return subprocess.CompletedProcess(command, 0, '{"classification":"public"}', "")

    monkeypatch.setattr(llm_cli, "resolve_ollama_cli", lambda: "/fake/ollama.exe")
    monkeypatch.setattr(llm_cli.subprocess, "run", fake_run)

    assert (
        llm_cli.call_ollama("synthetic prompt", "phi3:latest", 60)
        == '{"classification":"public"}'
    )
    assert calls == [
        ["/fake/ollama.exe", "show", "phi3:latest"],
        ["/fake/ollama.exe", "run", "phi3:latest", "synthetic prompt"],
    ]


def test_call_ollama_json_uses_nonstreaming_windows_bridge(monkeypatch):
    calls = []
    inner = '[{"qid":"q1","interpretation":"topical","reason":"synthetic"}]'
    outer = json.dumps({"response": inner, "done": True}).encode()

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        if command[1] == "show":
            return subprocess.CompletedProcess(command, 0, b"installed", b"")
        return subprocess.CompletedProcess(command, 0, outer, b"")

    monkeypatch.setattr(llm_cli, "resolve_ollama_cli", lambda: "/fake/ollama.exe")
    monkeypatch.setattr(llm_cli, "resolve_windows_curl", lambda: "/fake/curl.exe")
    monkeypatch.setattr(llm_cli.subprocess, "run", fake_run)

    assert llm_cli.call_ollama_json("synthetic prompt", "qwen3:4b", 60) == inner
    assert calls[0][0] == ["/fake/ollama.exe", "show", "qwen3:4b"]
    curl_command, curl_kwargs = calls[1]
    assert curl_command == [
        "/fake/curl.exe",
        "--disable",
        "--silent",
        "--show-error",
        "--fail-with-body",
        "--noproxy",
        "*",
        "--max-redirs",
        "0",
        "http://127.0.0.1:11434/api/generate",
        "-H",
        "Content-Type: application/json",
        "--data-binary",
        "@-",
    ]
    payload = json.loads(curl_kwargs["input"])
    assert payload == {
        "model": "qwen3:4b",
        "prompt": "synthetic prompt",
        "stream": False,
        "format": "json",
        "think": False,
        "options": {"temperature": 0, "seed": 0, "num_predict": 2048},
    }
    assert curl_kwargs["timeout"] == 60


def test_call_ollama_json_binds_reasoning_and_output_budget(monkeypatch):
    calls = []
    outer = json.dumps({"response": "[]", "done": True}).encode()

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        if command[1] == "show":
            return subprocess.CompletedProcess(command, 0, b"installed", b"")
        return subprocess.CompletedProcess(command, 0, outer, b"")

    monkeypatch.setattr(llm_cli, "resolve_ollama_cli", lambda: "/fake/ollama.exe")
    monkeypatch.setattr(llm_cli, "resolve_windows_curl", lambda: "/fake/curl.exe")
    monkeypatch.setattr(llm_cli.subprocess, "run", fake_run)

    assert (
        llm_cli.call_ollama_json(
            "synthetic prompt",
            "deepseek-r1:8b",
            60,
            max_output_tokens=4096,
            think=True,
        )
        == "[]"
    )
    payload = json.loads(calls[1][1]["input"])
    assert payload["think"] is True
    assert payload["options"]["num_predict"] == 4096


def test_call_ollama_json_exposes_completion_metadata(monkeypatch):
    outer = json.dumps(
        {
            "response": "[]",
            "done": True,
            "done_reason": "stop",
            "eval_count": 17,
            "eval_duration": 23,
            "prompt_eval_count": 41,
            "prompt_eval_duration": 43,
            "load_duration": 47,
            "total_duration": 53,
            "context": [1, 2, 3],
        }
    ).encode()

    def fake_run(command, **_kwargs):
        if command[1] == "show":
            return subprocess.CompletedProcess(command, 0, b"installed", b"")
        return subprocess.CompletedProcess(command, 0, outer, b"")

    monkeypatch.setattr(llm_cli, "resolve_ollama_cli", lambda: "/fake/ollama.exe")
    monkeypatch.setattr(llm_cli, "resolve_windows_curl", lambda: "/fake/curl.exe")
    monkeypatch.setattr(llm_cli.subprocess, "run", fake_run)
    metadata = {"stale": True}

    assert (
        llm_cli.call_ollama_json(
            "synthetic prompt", "qwen3:4b", 60, metadata_out=metadata
        )
        == "[]"
    )
    assert metadata == {
        "done_reason": "stop",
        "eval_count": 17,
        "eval_duration": 23,
        "prompt_eval_count": 41,
        "prompt_eval_duration": 43,
        "load_duration": 47,
        "total_duration": 53,
    }


def test_call_ollama_json_rejects_length_truncation(monkeypatch):
    outer = json.dumps(
        {"response": "[]", "done": True, "done_reason": "length"}
    ).encode()

    def fake_run(command, **_kwargs):
        if command[1] == "show":
            return subprocess.CompletedProcess(command, 0, b"installed", b"")
        return subprocess.CompletedProcess(command, 0, outer, b"")

    monkeypatch.setattr(llm_cli, "resolve_ollama_cli", lambda: "/fake/ollama.exe")
    monkeypatch.setattr(llm_cli, "resolve_windows_curl", lambda: "/fake/curl.exe")
    monkeypatch.setattr(llm_cli.subprocess, "run", fake_run)
    metadata = {"stale": True}

    assert (
        llm_cli.call_ollama_json(
            "synthetic prompt", "qwen3:4b", 60, metadata_out=metadata
        )
        is None
    )
    assert metadata == {}


def test_call_ollama_json_rejects_invalid_output_budget(monkeypatch):
    monkeypatch.setattr(
        llm_cli,
        "_require_local_ollama_model",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("model lookup must not run")
        ),
    )
    assert (
        llm_cli.call_ollama_json(
            "synthetic prompt",
            "qwen3:4b",
            60,
            max_output_tokens=0,
        )
        is None
    )


def test_call_ollama_json_rejects_malformed_inner_json(monkeypatch):
    outer = json.dumps({"response": "not-json", "done": True}).encode()

    def fake_run(command, **_kwargs):
        if command[1] == "show":
            return subprocess.CompletedProcess(command, 0, b"installed", b"")
        return subprocess.CompletedProcess(command, 0, outer, b"")

    monkeypatch.setattr(llm_cli, "resolve_ollama_cli", lambda: "/fake/ollama.exe")
    monkeypatch.setattr(llm_cli, "resolve_windows_curl", lambda: "/fake/curl.exe")
    monkeypatch.setattr(llm_cli.subprocess, "run", fake_run)

    assert llm_cli.call_ollama_json("synthetic prompt", "qwen3:4b", 60) is None


def test_local_ollama_endpoint_rejects_remote_and_userinfo(monkeypatch):
    bad_hosts = [
        "http://example.com:11434",
        "http://127.0.0.1:11434@evil.example",
        "https://127.0.0.1:11434",
        "http://127.0.0.1:11434/api",
        "http://127.0.0.1:11434/",
        "http://127.0.0.1:11434?proxy=1",
        "http://127.0.0.1:11434#fragment",
    ]
    for configured in bad_hosts:
        monkeypatch.setenv("OLLAMA_HOST", configured)
        try:
            llm_cli.local_ollama_endpoint()
        except ValueError:
            pass
        else:
            raise AssertionError(f"unsafe endpoint was accepted: {configured}")


def test_local_ollama_endpoint_accepts_loopback_forms(monkeypatch):
    expected = {
        "127.0.0.1": "http://127.0.0.1:11434",
        "localhost:22000": "http://127.0.0.1:22000",
        "http://LOCALHOST:22001": "http://127.0.0.1:22001",
        "http://[::1]:11435": "http://[::1]:11435",
    }
    for configured, canonical in expected.items():
        monkeypatch.setenv("OLLAMA_HOST", configured)
        assert llm_cli.local_ollama_endpoint() == canonical


def test_call_ollama_json_native_transport_disables_proxy(monkeypatch):
    inner = '[{"qid":"q1","interpretation":"topical","reason":"synthetic"}]'
    outer = json.dumps({"response": inner, "done": True}).encode()
    opened = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return outer

    class FakeOpener:
        def open(self, api_request, timeout):
            opened["url"] = api_request.full_url
            opened["timeout"] = timeout
            return FakeResponse()

    def fake_build_opener(*handlers):
        opened["handlers"] = handlers
        return FakeOpener()

    monkeypatch.setattr(llm_cli, "_require_local_ollama_model", lambda *_args: "/usr/bin/ollama")
    monkeypatch.setattr(llm_cli, "resolve_windows_curl", lambda: "/fake/curl.exe")
    monkeypatch.setattr(llm_cli.request, "build_opener", fake_build_opener)

    assert llm_cli.call_ollama_json("synthetic prompt", "qwen3:4b", 60) == inner
    assert opened["url"] == "http://127.0.0.1:11434/api/generate"
    assert opened["timeout"] == 60
    assert any(
        isinstance(handler, llm_cli.request.ProxyHandler)
        for handler in opened["handlers"]
    )
    assert any(
        isinstance(handler, llm_cli._NoRedirectHandler)
        for handler in opened["handlers"]
    )


def test_call_ollama_json_native_transport_denies_redirect(monkeypatch):
    redirected = []

    class RedirectingHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.send_response(302)
            self.send_header("Location", "/redirected")
            self.end_headers()

        def do_GET(self):
            redirected.append(self.path)
            body = json.dumps({"response": "[]", "done": True}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _format, *_args):
            return

    server = HTTPServer(("127.0.0.1", 0), RedirectingHandler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    monkeypatch.setenv("OLLAMA_HOST", f"http://127.0.0.1:{server.server_port}")
    monkeypatch.setattr(
        llm_cli,
        "_require_local_ollama_model",
        lambda *_args: "/usr/bin/ollama",
    )
    try:
        assert llm_cli.call_ollama_json("synthetic prompt", "qwen3:4b", 60) is None
        assert redirected == []
    finally:
        server.shutdown()
        worker.join(timeout=5)
        server.server_close()


def test_call_ollama_json_rejects_incomplete_outer_response(monkeypatch):
    outer = json.dumps({"response": "[]", "done": False}).encode()

    def fake_run(command, **_kwargs):
        if command[1] == "show":
            return subprocess.CompletedProcess(command, 0, b"installed", b"")
        return subprocess.CompletedProcess(command, 0, outer, b"")

    monkeypatch.setattr(llm_cli, "resolve_ollama_cli", lambda: "/fake/ollama.exe")
    monkeypatch.setattr(llm_cli, "resolve_windows_curl", lambda: "/fake/curl.exe")
    monkeypatch.setattr(llm_cli.subprocess, "run", fake_run)

    assert llm_cli.call_ollama_json("synthetic prompt", "qwen3:4b", 60) is None
