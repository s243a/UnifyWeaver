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

# Main logic to find all descendants
ancestor_all() {
    local start="$1"
    declare -A visited
    local queue_file="/tmp/ancestor_queue_$$"
    local next_queue="/tmp/ancestor_next_$$"
    
    trap "rm -f $queue_file $next_queue" EXIT PIPE
    
    echo "$start" > "$queue_file"
    visited["$start"]=1
    
    while [[ -s "$queue_file" ]]; do
        > "$next_queue"
        
        while IFS= read -r current; do
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
    local tmpflag="/tmp/ancestor_found_$$"
    
    # Use tee to prevent SIGPIPE when grep exits early (suppress all errors)
    ancestor_all "$start" 2>/dev/null | 
    tee >(grep -q "^$start:$target$" && touch "$tmpflag") >/dev/null 2>&1
    
    if [[ -f "$tmpflag" ]]; then
        rm -f "$tmpflag"
        return 0
    else
        rm -f "$tmpflag"
        return 1
    fi
}

# Main entry point
ancestor() {
    local start="$1"
    local target="$2"

    if [[ -z "$target" ]]; then
        # One-argument call: find all descendants
        if [[ "sort_u" == "sort_u" ]]; then
            ancestor_all "$start" | sort -u
        elif [[ "sort_u" == "hash_dedup" ]]; then
            declare -A seen
            ancestor_all "$start" | while IFS= read -r line; do
                if [[ -z "${seen[$line]}" ]]; then
                    seen[$line]=1
                    echo "$line"
                fi
            done
        else
            ancestor_all "$start"
        fi
    else
        # Two-argument call: check relationship
        if ancestor_check "$start" "$target"; then
            echo "$start:$target"
            return 0
        else
            return 1
        fi
    fi
}