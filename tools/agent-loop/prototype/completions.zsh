#compdef agent_loop.py agentloop

# Zsh completion for agent_loop.py
# Install: Copy to a directory in $fpath, or source directly
# Example: source completions.zsh

_agent_loop() {
    local -a opts backends context_modes context_formats models

    backends=(
        'coro:Coro CLI backend'
        'claude:Claude API backend'
        'claude-code:Claude Code CLI backend'
        'gemini:Gemini CLI backend'
        'openai:OpenAI API backend'
        'ollama-api:Ollama REST API backend'
        'ollama-cli:Ollama CLI backend'
    )

    context_modes=(
        'continue:Continue conversation with full context'
        'fresh:Each query is independent'
        'sliding:Keep only recent messages'
    )

    context_formats=(
        'plain:Simple text with role prefixes'
        'markdown:Markdown formatted'
        'json:JSON array of messages'
        'xml:XML structure'
    )

    models=(
        'sonnet:Claude Sonnet'
        'opus:Claude Opus'
        'haiku:Claude Haiku'
        'gpt-4o:GPT-4o'
        'gpt-4o-mini:GPT-4o Mini'
        'gpt-4:GPT-4'
        'llama3:Llama 3'
        'codellama:Code Llama'
        'gemini-2.5-flash:Gemini 2.5 Flash'
    )

    _arguments -s \
        '(-h --help)'{-h,--help}'[Show help message]' \
        '(-a --agent)'{-a,--agent}'[Agent variant from config]:agent:->agents' \
        '(-C --config)'{-C,--config}'[Path to config file]:config file:_files -g "*.{json,yaml,yml}"' \
        '--list-agents[List available agent variants]' \
        '--init-config[Create example config file]:path:_files -g "*.{json,yaml,yml}"' \
        '(-b --backend)'{-b,--backend}'[Backend to use]:backend:(($backends))' \
        '(-c --command)'{-c,--command}'[Command for CLI backends]:command:_command_names' \
        '(-m --model)'{-m,--model}'[Model to use]:model:(($models))' \
        '--host[Host for network backends]:host:_hosts' \
        '--port[Port for network backends]:port:()' \
        '--api-key[API key]:key:()' \
        '--no-tokens[Hide token usage information]' \
        '--context-mode[Context behavior mode]:mode:(($context_modes))' \
        '--auto-tools[Auto-execute tools without confirmation]' \
        '--no-tools[Disable tool execution]' \
        '--system-prompt[System prompt to use]:prompt:()' \
        '(-i --max-iterations)'{-i,--max-iterations}'[Max tool iterations]:iterations:(0 1 3 5 10)' \
        '(-s --session)'{-s,--session}'[Load a saved session]:session:->sessions' \
        '--list-sessions[List saved sessions]' \
        '--sessions-dir[Directory for session files]:directory:_directories' \
        '--context-format[Format for context]:format:(($context_formats))' \
        '--stream[Enable streaming output]' \
        '--no-cost-tracking[Disable cost tracking]' \
        '--search[Search across saved sessions]:query:()' \
        '*:prompt:'

    case "$state" in
        agents)
            local -a agents
            # Try to get agents from config
            if [[ -f "agents.json" ]]; then
                agents=(${(f)"$(python3 -c "import json; print('\n'.join(json.load(open('agents.json')).get('agents', {}).keys()))" 2>/dev/null)"})
            elif [[ -f "agents.yaml" ]]; then
                agents=(${(f)"$(python3 -c "import yaml; print('\n'.join(yaml.safe_load(open('agents.yaml')).get('agents', {}).keys()))" 2>/dev/null)"})
            fi
            # Fallback
            if [[ -z "$agents" ]]; then
                agents=(default claude-sonnet claude-opus claude-haiku yolo gemini ollama openai)
            fi
            _describe 'agent' agents
            ;;
        sessions)
            local sessions_dir="${HOME}/.agent-loop/sessions"
            if [[ -d "$sessions_dir" ]]; then
                local -a sessions
                sessions=(${sessions_dir}/*.json(:t:r))
                _describe 'session' sessions
            fi
            ;;
    esac
}

_agent_loop "$@"

# Create alias
alias agentloop='python3 agent_loop.py'
