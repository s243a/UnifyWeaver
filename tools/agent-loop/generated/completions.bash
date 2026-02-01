# Bash completion for agent_loop.py
# Source this file: source completions.bash
# Or add to ~/.bashrc: source /path/to/completions.bash

_agent_loop_completions() {
    local cur prev opts
    COMPREPLY=()
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"

    # Main options
    opts="--help --agent --config --list-agents --init-config --backend --command --model --host --port --api-key --no-tokens --context-mode --auto-tools --no-tools --system-prompt --max-iterations --session --list-sessions --sessions-dir --context-format --stream --no-cost-tracking --search"
    opts="$opts -h -a -C -b -c -m -i -s"

    # Backend choices
    local backends="coro claude claude-code gemini openai ollama-api ollama-cli"

    # Context modes
    local context_modes="continue fresh sliding"

    # Context formats
    local context_formats="plain markdown json xml"

    case "${prev}" in
        --backend|-b)
            COMPREPLY=( $(compgen -W "${backends}" -- ${cur}) )
            return 0
            ;;
        --context-mode)
            COMPREPLY=( $(compgen -W "${context_modes}" -- ${cur}) )
            return 0
            ;;
        --context-format)
            COMPREPLY=( $(compgen -W "${context_formats}" -- ${cur}) )
            return 0
            ;;
        --agent|-a)
            # Try to get agents from config file
            local agents=""
            if [[ -f "agents.json" ]]; then
                agents=$(python3 -c "import json; print(' '.join(json.load(open('agents.json')).get('agents', {}).keys()))" 2>/dev/null)
            elif [[ -f "agents.yaml" ]]; then
                agents=$(python3 -c "import yaml; print(' '.join(yaml.safe_load(open('agents.yaml')).get('agents', {}).keys()))" 2>/dev/null)
            fi
            # Fallback to common presets
            if [[ -z "$agents" ]]; then
                agents="default claude-sonnet claude-opus claude-haiku yolo gemini ollama openai"
            fi
            COMPREPLY=( $(compgen -W "${agents}" -- ${cur}) )
            return 0
            ;;
        --config|-C|--init-config)
            # File completion for config files
            COMPREPLY=( $(compgen -f -X '!*.@(json|yaml|yml)' -- ${cur}) )
            return 0
            ;;
        --session|-s)
            # Try to list sessions
            local sessions_dir="${HOME}/.agent-loop/sessions"
            if [[ -d "$sessions_dir" ]]; then
                local sessions=$(ls "$sessions_dir" 2>/dev/null | sed 's/\.json$//')
                COMPREPLY=( $(compgen -W "${sessions}" -- ${cur}) )
            fi
            return 0
            ;;
        --sessions-dir)
            # Directory completion
            COMPREPLY=( $(compgen -d -- ${cur}) )
            return 0
            ;;
        --model|-m)
            # Common model names
            local models="sonnet opus haiku gpt-4o gpt-4o-mini gpt-4 llama3 codellama gemini-2.5-flash"
            COMPREPLY=( $(compgen -W "${models}" -- ${cur}) )
            return 0
            ;;
        --host)
            COMPREPLY=( $(compgen -W "localhost 127.0.0.1" -- ${cur}) )
            return 0
            ;;
        --port)
            COMPREPLY=( $(compgen -W "11434 8080 3000" -- ${cur}) )
            return 0
            ;;
        --max-iterations|-i)
            COMPREPLY=( $(compgen -W "0 1 3 5 10" -- ${cur}) )
            return 0
            ;;
        *)
            ;;
    esac

    # Default to options if starting with -
    if [[ ${cur} == -* ]]; then
        COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
        return 0
    fi
}

# Register completion
complete -F _agent_loop_completions agent_loop.py
complete -F _agent_loop_completions python3\ agent_loop.py
complete -F _agent_loop_completions ./agent_loop.py

# Also create an alias for convenience
alias agentloop='python3 agent_loop.py'
complete -F _agent_loop_completions agentloop
