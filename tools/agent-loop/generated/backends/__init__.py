# Agent loop backends
from .base import AgentBackend, AgentResponse, ToolCall
from .coro import CoroBackend

# Claude API backend (optional - requires anthropic package)
try:
    from .claude_api import ClaudeAPIBackend
    __all__ = ['AgentBackend', 'AgentResponse', 'ToolCall', 'CoroBackend', 'ClaudeAPIBackend']
except ImportError:
    __all__ = ['AgentBackend', 'AgentResponse', 'ToolCall', 'CoroBackend']
