"""Configuration system for agent loop variants."""

import os
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any


def read_config_cascade(no_fallback: bool = False) -> dict:
    """Read config from uwsal.json, falling back to coro.json."""
    candidates = [
        'uwsal.json',
        os.path.expanduser('~/uwsal.json'),
    ]
    if not no_fallback:
        candidates += [
            'coro.json',
            os.path.expanduser('~/coro.json'),
        ]
    for path in candidates:
        try:
            with open(path) as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            continue
    return {}



def resolve_api_key(backend_type: str, cli_key: str | None = None,
                    no_fallback: bool = False) -> str | None:
    """Resolve API key with full priority chain."""
    if cli_key:
        return cli_key

    env_vars = {
        'openrouter': 'OPENROUTER_API_KEY',
        'claude': 'ANTHROPIC_API_KEY',
        'openai': 'OPENAI_API_KEY',
        'gemini': 'GEMINI_API_KEY',
    }
    env_var = env_vars.get(backend_type)
    if env_var:
        val = os.environ.get(env_var)
        if val:
            return val

    config = read_config_cascade(no_fallback)
    provider_key = config.get('keys', {}).get(backend_type)
    if provider_key:
        return provider_key
    if config.get('api_key'):
        return config['api_key']

    file_locations = {
        'claude': '~/.anthropic/api_key',
        'openai': '~/.openai/api_key',
    }
    loc = file_locations.get(backend_type)
    if loc:
        path = os.path.expanduser(loc)
        try:
            with open(path) as f:
                return f.read().strip()
        except FileNotFoundError:
            pass

    return None


# Try to import yaml, fall back to JSON only
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


@dataclass
class AgentConfig:
    """Configuration for an agent variant."""
    name: str
    backend: str  # coro, claude-code, gemini, claude, ollama-api, ollama-cli
    model: str | None = None
    host: str | None = None  # For network backends (ollama-api)
    port: int | None = None
    api_key: str | None = None  # Or env var name like $ANTHROPIC_API_KEY
    command: str | None = None  # For CLI backends
    system_prompt: str | None = None
    agent_md: str | None = None  # Path to agent.md file
    tools: list[str] = field(default_factory=lambda: ['bash', 'read', 'write', 'edit'])
    auto_tools: bool = False  # Skip confirmation
    context_mode: str = "continue"  # continue, fresh, sliding
    max_context_tokens: int = 100000
    max_messages: int = 50
    skills: list[str] = field(default_factory=list)  # Paths to skill files
    max_iterations: int = 0  # 0 = unlimited, N = pause after N tool iterations
    timeout: int = 300
    show_tokens: bool = True
    extra: dict = field(default_factory=dict)


@dataclass
class Config:
    """Root configuration with multiple agent variants."""
    default: str = "default"
    agents: dict[str, AgentConfig] = field(default_factory=dict)

    # Global settings
    config_dir: str = ""
    skills_dir: str = ""


def _resolve_env_var(value: str) -> str:
    """Resolve environment variable references like $VAR or ${VAR}."""
    if not isinstance(value, str):
        return value
    if value.startswith('$'):
        var_name = value[1:]
        if var_name.startswith('{') and var_name.endswith('}'):
            var_name = var_name[1:-1]
        return os.environ.get(var_name, value)
    return value


def _load_agent_config(name: str, data: dict) -> AgentConfig:
    """Load an agent config from a dictionary."""
    if 'api_key' in data:
        data['api_key'] = _resolve_env_var(data['api_key'])

    return AgentConfig(
        name=name,
        backend=data.get('backend', 'coro'),
        model=data.get('model'),
        host=data.get('host'),
        port=data.get('port'),
        api_key=data.get('api_key'),
        command=data.get('command'),
        system_prompt=data.get('system_prompt'),
        agent_md=data.get('agent_md'),
        tools=data.get('tools', ['bash', 'read', 'write', 'edit']),
        auto_tools=data.get('auto_tools', False),
        context_mode=data.get('context_mode', 'continue'),
        max_context_tokens=data.get('max_context_tokens', 100000),
        max_messages=data.get('max_messages', 50),
        skills=data.get('skills', []),
        max_iterations=data.get('max_iterations', 0),
        timeout=data.get('timeout', 300),
        show_tokens=data.get('show_tokens', True),
        extra=data.get('extra', {})
    )


def load_config(path: str | Path) -> Config:
    """Load configuration from a YAML or JSON file."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    content = path.read_text()

    if path.suffix in ('.yaml', '.yml'):
        if not HAS_YAML:
            raise ImportError("PyYAML not installed. Use JSON config or: pip install pyyaml")
        data = yaml.safe_load(content)
    else:
        data = json.loads(content)

    config = Config(
        default=data.get('default', 'default'),
        config_dir=str(path.parent),
        skills_dir=data.get('skills_dir', '')
    )

    for name, agent_data in data.get('agents', {}).items():
        config.agents[name] = _load_agent_config(name, agent_data)

    if 'default' not in config.agents:
        config.agents['default'] = AgentConfig(name='default', backend='coro')

    return config


def load_config_from_dir(dir_path: str | Path = None) -> Config | None:
    """Load config from standard locations in a directory."""
    if dir_path is None:
        dir_path = Path.cwd()
    else:
        dir_path = Path(dir_path)

    for name in ['agents.yaml', 'agents.yml', 'agents.json',
                 '.agents.yaml', '.agents.yml', '.agents.json']:
        config_path = dir_path / name
        if config_path.exists():
            return load_config(config_path)

    return None


def get_default_config() -> Config:
    """Get a default configuration with common presets."""
    config = Config()

    config.agents['default'] = AgentConfig(
        name='default',
        backend='coro',
        command='claude',
    )

    config.agents['claude-sonnet'] = AgentConfig(
        name='claude-sonnet',
        backend='claude-code',
        model='sonnet',
    )

    config.agents['claude-opus'] = AgentConfig(
        name='claude-opus',
        backend='claude-code',
        model='opus',
    )

    config.agents['claude-haiku'] = AgentConfig(
        name='claude-haiku',
        backend='claude-code',
        model='haiku',
    )

    config.agents['yolo'] = AgentConfig(
        name='yolo',
        backend='claude-code',
        model='haiku',
        auto_tools=True,
        system_prompt='Be concise and take action. Execute tools without asking.',
    )

    config.agents['gemini'] = AgentConfig(
        name='gemini',
        backend='gemini',
        model='gemini-3-flash-preview',
    )

    config.agents['ollama'] = AgentConfig(
        name='ollama',
        backend='ollama-api',
        model='llama3',
        host='localhost',
        port=11434,
    )

    config.agents['ollama-cli'] = AgentConfig(
        name='ollama-cli',
        backend='ollama-cli',
        model='llama3',
    )

    config.agents['openai'] = AgentConfig(
        name='openai',
        backend='openai',
        model='gpt-4o',
    )

    config.agents['gpt-4o-mini'] = AgentConfig(
        name='gpt-4o-mini',
        backend='openai',
        model='gpt-4o-mini',
    )

    return config


def save_example_config(path: str | Path):
    """Save an example configuration file."""
    path = Path(path)

    example = {
        "default": "claude-sonnet",
        "skills_dir": "./skills",
        "agents": {
            "claude-sonnet": {
                "backend": "claude-code",
                "model": "sonnet",
                "tools": ["bash", "read", "write", "edit"],
                "context_mode": "continue"
            },
            "claude-opus": {
                "backend": "claude-code",
                "model": "opus",
                "system_prompt": "You are a senior software engineer. Be thorough."
            },
            "yolo": {
                "backend": "claude-code",
                "model": "haiku",
                "auto_tools": True,
                "system_prompt": "Be fast and take action without asking."
            },
            "gemini": {
                "backend": "gemini",
                "model": "gemini-2.5-flash"
            },
            "ollama-local": {
                "backend": "ollama-api",
                "model": "llama3",
                "host": "localhost",
                "port": 11434
            },
            "ollama-remote": {
                "backend": "ollama-api",
                "model": "codellama",
                "host": "192.168.1.100",
                "port": 11434
            },
            "claude-api": {
                "backend": "claude",
                "model": "claude-sonnet-4-20250514",
                "api_key": "$ANTHROPIC_API_KEY"
            },
            "openai": {
                "backend": "openai",
                "model": "gpt-4o",
                "api_key": "$OPENAI_API_KEY"
            },
            "openai-mini": {
                "backend": "openai",
                "model": "gpt-4o-mini",
                "api_key": "$OPENAI_API_KEY"
            },
            "coding-assistant": {
                "backend": "claude-code",
                "model": "sonnet",
                "agent_md": "./agents/coding.md",
                "skills": ["./skills/git.md", "./skills/testing.md"],
                "tools": ["bash", "read", "write", "edit"],
                "system_prompt": "You are a coding assistant focused on clean, tested code."
            }
        }
    }

    if path.suffix in ('.yaml', '.yml'):
        if not HAS_YAML:
            raise ImportError("PyYAML not installed for YAML output")
        content = yaml.dump(example, default_flow_style=False, sort_keys=False)
    else:
        content = json.dumps(example, indent=2)

    path.write_text(content)

