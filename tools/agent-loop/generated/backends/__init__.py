# Agent loop backends
from .base import AgentBackend, AgentResponse, ToolCall
from .coro import CoroBackend
from .claude_code import ClaudeCodeBackend
from .gemini import GeminiBackend
from .ollama_api import OllamaAPIBackend
from .ollama_cli import OllamaCLIBackend

__all__ = [
    'AgentBackend', 'AgentResponse', 'ToolCall',
    'CoroBackend', 'ClaudeCodeBackend', 'GeminiBackend',
    'OllamaAPIBackend', 'OllamaCLIBackend'
]

# Claude API backend (optional - requires anthropic package)
try:
    from .claude_api import ClaudeAPIBackend
    __all__.append('ClaudeAPIBackend')
except ImportError:
    pass
