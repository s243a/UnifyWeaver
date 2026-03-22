"""Auto-generated tool definitions from Prolog specs."""

# Binding: tool_handler/2 -> TOOL_HANDLERS[name](name) [dict_lookup]
# Binding: destructive_tool/1 -> DESTRUCTIVE_TOOLS(tool_name) [set_membership]

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
            {"name": "file_path", "type": "string", "required": True, "description": "Path to file"},
        ]
    },
    "write": {
        "description": "Write content to a file",
        "parameters": [
            {"name": "file_path", "type": "string", "required": True, "description": "Path to file"},
            {"name": "content", "type": "string", "required": True, "description": "Content to write"},
        ]
    },
    "edit": {
        "description": "Edit a file with search/replace",
        "parameters": [
            {"name": "file_path", "type": "string", "required": True, "description": "Path to file"},
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



_TOOL_SCHEMAS_CACHE = None

def get_tool_schemas() -> list[dict]:
    """Return cached tool schemas in JSON Schema format for API backends."""
    global _TOOL_SCHEMAS_CACHE
    if _TOOL_SCHEMAS_CACHE is not None:
        return _TOOL_SCHEMAS_CACHE
    schemas = []
    for name, spec in TOOL_SPECS.items():
        props = {}
        required = []
        for p in spec.get("parameters", []):
            props[p["name"]] = {"type": p.get("param_type", "string")}
            if p.get("description"):
                props[p["name"]]["description"] = p["description"]
            if p.get("required", False):
                required.append(p["name"])
        schema = {
            "type": "function",
            "function": {
                "name": name,
                "description": spec.get("description", ""),
                "parameters": {
                    "type": "object",
                    "properties": props,
                    "required": required
                }
            }
        }
        schemas.append(schema)
    _TOOL_SCHEMAS_CACHE = schemas
    return schemas
