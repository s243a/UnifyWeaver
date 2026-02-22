"""Auto-generated tool definitions from Prolog specs."""

TOOL_SPECS = {
    "bash": {
        "description": "Execute a bash command",
        "parameters": [
            {"name": "command", "type": "string", "required": True, "description": "The command to execute"},
        ]
    },
    "read": {
        "description": "Read a file",
        "parameters": [
            {"name": "path", "type": "string", "required": True, "description": "Path to file"},
        ]
    },
    "write": {
        "description": "Write content to a file",
        "parameters": [
            {"name": "path", "type": "string", "required": True, "description": "Path to file"},
            {"name": "content", "type": "string", "required": True, "description": "Content to write"},
        ]
    },
    "edit": {
        "description": "Edit a file with search/replace",
        "parameters": [
            {"name": "path", "type": "string", "required": True, "description": "Path to file"},
            {"name": "old_string", "type": "string", "required": True, "description": "Text to find"},
            {"name": "new_string", "type": "string", "required": True, "description": "Replacement text"},
        ]
    },
}

DESTRUCTIVE_TOOLS = {
    "bash",
    "write",
    "edit",
}
