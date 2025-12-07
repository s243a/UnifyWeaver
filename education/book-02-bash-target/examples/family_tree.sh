# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2025 John William Creighton (s243a)
#
# This file is part of UnifyWeaver.
# Licensed under either MIT or Apache-2.0 at your option.

#!/bin/bash
# ancestor - transitive closure of parent

# Check for base stream function
parent_get_stream() {
    if declare -f parent_stream >/dev/null 2>&1; then
        parent_stream
    elif declare -f parent >/dev/null 2>&1; then
        parent
    else
        echo "Error: parent not found" >&2
        return 1
    fi
}

# Main function
ancestor() {
    local start="$1"
    local target="$2"
    
    if [[ -z "$target" ]]; then
        ancestor_all "$start"
    else
        ancestor_check "$start" "$target"
    fi
}

# Find all reachable using BFS
ancestor_all() {
    local start="$1"
    declare -A visited
    local queue_file="/tmp/ancestor_queue_$"
    local next_queue="/tmp/ancestor_next_$"
    
    trap "rm -f $queue_file $next_queue" EXIT PIPE
    
    echo "$start" > "$queue_file"
    visited["$start"]=1
    
    while [[ -s "$queue_file" ]]; do
        > "$next_queue"
        
        while IFS= read -r current; do
            # Use process substitution to keep while loop in current shell
            while IFS=":" read -r from to; do
                if [[ "$from" == "$current" && -z "${visited[$to]}" ]]; then
                    visited["$to"]=1
                    echo "$to" >> "$next_queue"
                    echo "$start:$to"
                fi
            done < <(parent_get_stream | grep "^$current:")
        done < "$queue_file"
        
        mv "$next_queue" "$queue_file"
    done
    
    rm -f "$queue_file" "$next_queue"
}

# Check specific relationship
ancestor_check() {
    local start="$1"
    local target="$2"
    ancestor_all "$start" | grep -q "^$start:$target$" && echo "$start:$target"
}

# Stream function
ancestor_stream() {
    ancestor_all "$1"
}