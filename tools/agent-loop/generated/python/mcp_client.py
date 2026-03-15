# MCP (Model Context Protocol) client - stdio JSON-RPC 2.0 transport.

import json
import subprocess
import os
import sys

class MCPClient:
    """Connect to an MCP server over stdio and call tools."""

    def __init__(self, name: str, command: list[str], env: dict[str, str] | None = None):
        self.name = name
        self.command = command
        self.env = env
        self._process = None
        self._request_id = 0

    def connect(self, timeout: float = 10.0) -> bool:
        """Start the MCP server subprocess."""
        try:
            merged_env = {**os.environ, **(self.env or {})}
            self._process = subprocess.Popen(
                self.command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=merged_env,
            )
            # Send initialize request
            resp = self._send_request("initialize", {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "uwsal", "version": "0.2.0"}
            })
            return resp is not None and "error" not in resp
        except (OSError, FileNotFoundError) as e:
            print(f"MCP [{self.name}] connect failed: {e}", file=sys.stderr)
            return False

    def _send_request(self, method: str, params: dict | None = None) -> dict | None:
        """Send a JSON-RPC 2.0 request and read the response."""
        if not self._process or self._process.stdin is None:
            return None
        self._request_id += 1
        request = {
            "jsonrpc": "2.0",
            "id": self._request_id,
            "method": method,
        }
        if params is not None:
            request["params"] = params
        try:
            line = json.dumps(request) + "\n"
            self._process.stdin.write(line.encode())
            self._process.stdin.flush()
            resp_line = self._process.stdout.readline()
            if not resp_line:
                return None
            return json.loads(resp_line.decode())
        except (BrokenPipeError, json.JSONDecodeError, OSError):
            return None

    def discover_tools(self) -> list[dict]:
        """Call tools/list and return tool definitions."""
        resp = self._send_request("tools/list", {})
        if resp and "result" in resp:
            return resp["result"].get("tools", [])
        return []

    def call_tool(self, name: str, arguments: dict) -> dict:
        """Call tools/call and return the result."""
        resp = self._send_request("tools/call", {"name": name, "arguments": arguments})
        if resp and "result" in resp:
            return resp["result"]
        error = resp.get("error", {}) if resp else {"message": "No response"}
        return {"error": error.get("message", str(error))}

    def disconnect(self) -> None:
        """Terminate the MCP server subprocess."""
        if self._process:
            try:
                self._process.stdin.close()
                self._process.terminate()
                self._process.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                self._process.kill()
            self._process = None

    def __del__(self):
        self.disconnect()


class MCPManager:
    """Manage multiple MCP server connections."""

    def __init__(self, server_configs: list[dict] | None = None):
        self.clients: dict[str, MCPClient] = {}
        self._tools: dict[str, tuple[str, dict]] = {}  # tool_name -> (server_name, spec)
        if server_configs:
            for cfg in server_configs:
                self.add_server(cfg)

    def add_server(self, config: dict) -> bool:
        """Add and connect an MCP server from config dict."""
        name = config.get("name", "")
        command = config.get("command", [])
        if isinstance(command, str):
            command = command.split()
        env = config.get("env")
        client = MCPClient(name, command, env)
        if client.connect():
            self.clients[name] = client
            # Discover tools
            for tool in client.discover_tools():
                tool_name = f"mcp:{name}:{tool["name"]}"
                self._tools[tool_name] = (name, tool)
            return True
        return False

    def list_tools(self) -> list[dict]:
        """List all discovered MCP tools."""
        return [
            {"name": name, "server": server, "description": spec.get("description", "")}
            for name, (server, spec) in self._tools.items()
        ]

    def dispatch(self, tool_name: str, arguments: dict) -> dict | None:
        """Dispatch a tool call to the appropriate MCP server."""
        entry = self._tools.get(tool_name)
        if not entry:
            return None
        server_name, spec = entry
        client = self.clients.get(server_name)
        if not client:
            return None
        # Extract the real tool name (strip mcp:server: prefix)
        real_name = spec.get("name", tool_name.split(":")[-1])
        return client.call_tool(real_name, arguments)

    def disconnect_all(self) -> None:
        """Disconnect all MCP servers."""
        for client in self.clients.values():
            client.disconnect()
        self.clients.clear()
        self._tools.clear()
