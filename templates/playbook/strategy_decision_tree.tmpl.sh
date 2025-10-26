# =============================================================================
# Strategy Selection Logic
# =============================================================================

function select_strategy() {
    local strategy_choice="$1"
    # The user's query and other arguments can be passed as $2, $3, etc.

    echo "[INFO] Attempting to execute strategy: $strategy_choice"

    case "$strategy_choice" in
{{ strategy_case_block }}
        *)
            echo "[ERROR] Unknown strategy: '$strategy_choice'" >&2
            echo "Available strategies are: {{ available_strategies }}" >&2
            exit 1
            ;;
    esac
}

# =============================================================================
# Main Executor
# =============================================================================

# Default to showing help if no strategy is provided.
if [ -z "$1" ]; then
    echo "Usage: $0 <strategy_name> [args...]"
    echo "Available strategies: {{ available_strategies }}"
    exit 0
fi

# Run the selected strategy with all provided arguments.
select_strategy "$@"
