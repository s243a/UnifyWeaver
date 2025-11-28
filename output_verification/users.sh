#!/bin/bash
# users - CSV source (arity 3)
# Columns: id, name, age

users() {
    local target_key="$1"
    
    if [[ -z "$target_key" ]]; then
        # No key provided, stream all rows
        awk -F"," '
        NR > 1 {
            gsub(/"/, "", $0)
            if (NF >= 3) print $1":"$2":"$3
        }
        ' test_data/test_users.csv
    else
        # Lookup mode: find rows where first column matches key
        awk -F"," -v key="$target_key" '
        NR > 1 {
            gsub(/"/, "", $0)
            if (NF >= 3 && $1 == key) print $1":"$2":"$3
        }
        ' test_data/test_users.csv
    fi
}

users_stream() {
    users
}

users_all() {
    users
}

users_check() {
    local key="$1"
    [[ -n $(users "$key") ]] && echo "$key exists"
}

# Auto-execute when run directly (not when sourced)
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    users "$@"
fi
